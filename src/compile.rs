use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Result, bail};

use log::{warn, debug};

use crate::ast::{self, Expr};
use crate::interpret::{Op, Pattern, MenuItem, Script, TaskLabel};
use crate::eval::Placement;

#[derive(Clone, Debug)]
pub enum GlobalType {
    Int,

    Enum {
        variants: Vec<Arc<str>>,
    },
}

#[derive(Default)]
pub(crate) struct Compiler {
    /// Stores all generated opcodes.
    body: Vec<Op>,

    /// Maps page names to LabelId placeholders.
    pages: HashMap<Arc<str>, PageInfo>,

    /// Globals used by this script, with optional default values.
    globals: HashMap<Arc<str>, Option<Expr>>,

    /// Contains all labels that have been previously defined.
    known_labels: HashMap<LabelId, TaskLabel>,

    /// Maps unvisited labels to the indices of opcodes that use them.
    unknown_labels: HashMap<LabelId, HashSet<usize>>,

    next_label: LabelId,

    continue_target: ContinueTarget,
}

struct PageInfo {
    entry_point: LabelId,

    params: Vec<Arc<str>>,
}

#[derive(Clone, Debug, Default)]
enum ContinueTarget {
    #[default]
    CannotContinue,

    Page(Arc<str>),
}

/// Placeholder for a `TaskLabel` that may or may not have been defined.
#[derive(Copy, Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct LabelId(usize);

impl ast::Script {
    pub fn compile(&self) -> Result<Script> {
        let mut compiler = Compiler::default();

        let header: Vec<_> = self.header.clone();

        for decl in header {
            match decl {
                ast::Decl::Enum { name, variants } => {
                    let variants = variants.iter()
                    .map(|s| Arc::<str>::from(s.as_str()))
                    .collect();

                    compiler.define_global(&name, GlobalType::Enum { variants })?;
                },

                ast::Decl::Flag { name } => {
                    compiler.define_global(&name, GlobalType::Int)?;
                },
            }
        }

        // Create LabelIds for all pages
        for ast::Page { label: name, .. } in self.pages.clone() {
            let label = compiler.alloc_label();

            if compiler.pages.contains_key(&name) {
                bail!("Redefined page name {name}");
            }

            compiler.pages.insert(name, PageInfo {
                entry_point: label,
                params: vec![], // TODO: Get params from AST
            });
        }

        if let Some(first) = self.pages.first() {
           let name = first.label.clone();
           compiler.continue_target = ContinueTarget::Page(name);
        }

        for stmt in self.setup.clone() {
            compiler.tr_stmt(stmt)?;
        }

        // Use peekable iterator to be able to set continue points
        let mut pages = self.pages.clone().into_iter().peekable();

        while let Some(ast::Page { label, body }) = pages.next() {
            let Some(page) = compiler.pages.get(&label) else {
                bail!("Internal error: Page entry point {label:?} was never bookmarked");
            };

            compiler.define_label(page.entry_point)?;

            if let Some(ast::Page { label: next, .. }) = pages.peek() {
                compiler.continue_target = ContinueTarget::Page(next.clone());
            }

            for stmt in body {
                compiler.tr_stmt(stmt)?;
            }
        }

        compiler.finish()
    }
}

impl Compiler {
    fn define_global(&mut self, name: &str, _gtype: GlobalType) -> Result<()> {
        warn!("TODO: Typed globals");

        if self.globals.contains_key(name) {
            bail!("Global ${name} redefined");
        }

        self.globals.insert(name.into(), None);

        Ok(())
    }

    fn alloc_label(&mut self) -> LabelId {
        let label = self.next_label.postincrement();
        self.unknown_labels.entry(label).or_default();
        label
    }

    fn use_label(&mut self, label: LabelId) -> Result<TaskLabel> {
        if let Some(&label) = self.known_labels.get(&label) {
            return Ok(label);
        }

        // Bookmark current opcode for fixups
        self.unknown_labels.entry(label)
        .or_default()
        .insert(self.body.len());

        Ok(label.as_task_label())
    }

    fn jump_to(&mut self, label: LabelId) -> Result<usize> {
        if self.known_labels.contains_key(&label) {
            bail!("Cannot jump forward to previously defined label {label:?}");
        }

        self.unknown_labels.entry(label)
        .or_default()
        .insert(self.body.len());

        Ok(label.as_task_label().0)
    }

    fn define_label(&mut self, label: LabelId) -> Result<()> {
        if let Some(&old) = self.known_labels.get(&label) {
            bail!("Redefined {label:?}; was previously {old:?}");
        }

        let Some(fixups) = self.unknown_labels.remove(&label) else {
            bail!("Tried to define an unallocated label: {label:?}");
        };

        // Disjoint borrow
        let Compiler {
            known_labels,
            body,
            ..
        } = self;

        let jump_target = body.len();

        known_labels.insert(label, TaskLabel(jump_target));

        for index in fixups {
            match &mut body[index] {
                Op::EnterBlock { offset } |
                Op::Jump { offset } |
                Op::Jz { offset, .. } => {
                    // Target is encoded as offset
                    // Check validity first
                    if *offset < LabelId::MAGIC {
                        warn!("Bad fixup at index {index}");
                        continue;
                    }

                    let bookmark = LabelId(*offset - LabelId::MAGIC);
                    let Some(&label) = known_labels.get(&bookmark) else {
                        bail!("No such label {bookmark:?} in known_labels");
                    };

                    let Some(fixup) = label.0.checked_sub(index) else {
                        bail!("Attempted backward jump to {label:?}");
                    };

                    debug!("Fixup: {offset} -> {fixup}");

                    // Is this an off-by-one error?
                    // TODO: Check against Actor::resume()
                    *offset = fixup;
                },

                other => other.visit_labels(|label| {
                    let TaskLabel(addr) = *label;

                    if addr >= LabelId::MAGIC {
                        let bookmark = LabelId(addr - LabelId::MAGIC);
                        if let Some(dst) = known_labels.get(&bookmark) {
                            *label = *dst;
                        }
                    }

                    Ok(())
                })?,
            }
        }

        Ok(())
    }

    fn finish(mut self) -> Result<Script> {
        if !self.unknown_labels.is_empty() {
            let count = self.unknown_labels.len();
            bail!("Failed to define {count} labels");
        }

        let body_len = self.body.len();

        for (index, op) in self.body.iter_mut().enumerate() {
            op.visit_offsets(|offset| {
                let target = index + *offset;

                if target > body_len {
                    warn!("Offset {offset} jumps outside script body");
                }

                Ok(())
            })?;

            op.visit_labels(|label| {
                if label.0 >= LabelId::MAGIC {
                    bail!("Forgot to fixup {label:?}");
                } else if label.0 >= body_len {
                    warn!("{label:?} jumps outside of script body");
                }

                Ok(())
            })?;
        }

        let body = self.body.into();
        let pages = self.pages.into_iter().map(|(name, info)| {
            use crate::interpret;

            let entry_point = self.known_labels[&info.entry_point];
            let params = info.params.into();

            (name, interpret::PageInfo {
                entry_point,
                params,
            })
        }).collect();

        let globals = self.globals;

        Ok(Script { body, pages, globals })
    }

    /// Translate the body of a nested scope that is evaluated immediately.
    fn in_block(&mut self, f: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        let after = self.alloc_label();

        let offset = self.jump_to(after)?;
        self.emit(Op::EnterBlock { offset })?;

        let t = f(self)?;
        self.emit(Op::Return)?;

        self.define_label(after)?;
        Ok(t)
    }

    /// Translate the body of a callback with a specified label.
    fn in_callback(&mut self, start: LabelId, f: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        let after = self.alloc_label();

        let offset = self.jump_to(after)?;
        self.emit(Op::Jump { offset })?;

        self.define_label(start)?;
        f(self)?;
        self.emit(Op::Return)?;

        self.define_label(after)?;
        Ok(())
    }

    fn bind_local(&mut self, name: &str) {
        warn!("Not implemented: Binding local name {name}");
        //let name = name.into();
        //let parent = None;
        //let refcount = 0;
        //self.current_block().last_binding = Some(self.bindings.len());
        //self.bindings.push(Binding { name, parent, refcount });
    }

    fn use_local(&mut self, name: &str) -> Result<()> {
        warn!("TODO: Consume binding {name}");
        Ok(())
    }

    fn emit(&mut self, op: Op) -> Result<()> {
        self.body.push(op);
        Ok(())
    }

    fn tr_stmt(&mut self, stmt: ast::Stmt) -> Result<()> {
        match stmt {
            ast::Stmt::Let { name, value } => {
                let value = self.tr_expr(value)?.into();
                self.bind_local(&name);

                self.emit(Op::Eval {
                    expr: value,
                    dst: Placement::CreateLocal { name },
                })?;
            },

            ast::Stmt::If { cases, fallback } => {
                let exit = self.alloc_label();

                for ast::IfThen { guard, body } in cases {
                    let or_else = self.alloc_label();

                    let guard = self.tr_expr(guard)?.into();
                    let offset = self.jump_to(or_else)?;
                    self.emit(Op::Jz { guard, offset })?;

                    self.in_block(|this| {
                        for stmt in body {
                            this.tr_stmt(stmt)?;
                        }

                        Ok(())
                    })?;

                    let offset = self.jump_to(exit)?;
                    self.emit(Op::Jump { offset })?;

                    self.define_label(or_else)?;
                }

                self.in_block(|this| {
                    for stmt in fallback {
                        this.tr_stmt(stmt)?;
                    }

                    Ok(())
                })?;

                self.define_label(exit)?;
            },

            ast::Stmt::BareExpr { expr } => {
                let expr = self.tr_expr(expr)?.into();

                self.emit(Op::Eval { expr, dst: Placement::Discard })?;
            },

            ast::Stmt::Assign { lhs, op, mut rhs } => {
                if let Some(op) = op {
                    let lhs = lhs.clone().into();
                    rhs = Expr::Infix { lhs, op, rhs: rhs.into() };
                }

                match lhs {
                    Expr::Global { name } => {
                        let rhs = self.tr_expr(rhs)?.into();
                        let name = name.as_str().into();
                        self.emit(Op::Eval {
                            expr: rhs,
                            dst: Placement::Global { name },
                        })?;
                    },

                    Expr::Local { name } => {
                        self.use_local(&name)?;
                        let value = self.tr_expr(rhs)?.into();
                        self.emit(Op::Eval {
                            expr: value,
                            dst: Placement::UpdateLocal { name },
                        })?;
                    },

                    lhs => bail!("Cannot assign to {:?}", lhs),
                }
            },

            ast::Stmt::On { pattern, body } => {
                let pattern = self.tr_pattern(pattern)?;
                let handler_body = self.alloc_label();

                let label = self.use_label(handler_body)?;
                self.emit(Op::PushHandler {
                    pattern,
                    cancel: false,
                    label,
                })?;

                self.in_callback(handler_body, |this| {
                    for stmt in body {
                        this.tr_stmt(stmt)?;
                    }

                    Ok(())
                })?;
            },

            ast::Stmt::Listen { pattern } => {
                let label = TaskLabel(0);
                let pattern = self.tr_pattern(pattern)?;
                self.emit(Op::PushHandler { pattern, label, cancel: true })?;
                self.emit(Op::Return)?;
                // TODO: Fixup
            },

            ast::Stmt::Menu { choices } => {
                let mut items = Vec::<MenuItem>::new();
                let mut bodies = Vec::new();

                for ast::MenuItem { prompt, actions } in choices {
                    let prompt = prompt.into();
                    let pc = self.alloc_label();

                    items.push(MenuItem {
                        prompt,
                        guard: None,
                        target: self.use_label(pc)?,
                    });

                    bodies.push((pc, actions));
                }

                let choices = Arc::<[_]>::from(items);
                self.emit(Op::Menu { choices })?;

                for (pc, actions) in bodies {
                    self.in_callback(pc, |this| {
                        for stmt in actions {
                            this.tr_stmt(stmt)?;
                        }

                        Ok(())
                    })?;
                }
            },

            ast::Stmt::Quote { speaker, text } => {
                let speaker = match speaker {
                    Some(expr) => Some(self.tr_expr(expr)?.into()),
                    None => None,
                };

                let lines = text.into_iter().map(|s| {
                    s.map_exprs(&mut |expr| self.tr_expr(expr))
                }).collect::<Result<Vec<_>>>()?.into();

                self.emit(Op::Quote { speaker, lines })?;
            },

            ast::Stmt::Music { path } => {
                self.emit(Op::Eval {
                    expr: Expr::FnCall {
                        lhs: Expr::Local { name: "music".into() }.into(),
                        args: vec![
                            Expr::String { value: path },
                        ].into(),
                    }.into(),
                    dst: Placement::Discard,
                })?;
            },

            ast::Stmt::Trace { on } => {
                self.emit(Op::Trace { on })?;
            },

            ast::Stmt::Goto { label } => {
                let path = label.into();
                self.emit(Op::Tailcall {
                    path,
                    args: vec![].into(),
                })?;
            },

            ast::Stmt::Wait { amount } => {
                let amount = self.tr_expr(amount)?.into();
                self.emit(Op::Wait { amount })?;
            },

            ast::Stmt::Continue => match &self.continue_target {
                ContinueTarget::CannotContinue => {
                    bail!("Cannot continue from here");
                },

                ContinueTarget::Page(name) => {
                    if let Some(page) = self.pages.get(name) {
                        if page.params.len() != 0 {
                            bail!("Cannot continue to a page that takes params");
                        }
                    }

                    self.emit(Op::Tailcall {
                        path: name.clone(),
                        args: Vec::new().into(),
                    })?;
                },

                //other => bail!("Not yet implemented: {other:?}"),
            },

            ast::Stmt::Hibernate => {
                self.emit(Op::Return)?;
            },

            ast::Stmt::Return => {
                self.emit(Op::Return)?;
            },

            ast::Stmt::Bye => {
                self.emit(Op::Retire)?;
            },
        }

        Ok(())
    }

    fn tr_pattern(&mut self, pattern: ast::Pattern) -> Result<Arc<Pattern>> {
        let ast::Pattern { variant, params, wildcard, guard } = pattern;

        let subject = variant.as_str().into();

        let mut used = HashSet::<Arc<str>>::new();

        let params = params.into_iter().map(|name| {
            let name = Arc::from(name.as_str());

            self.bind_local(&name);

            if used.contains(&name) {
                bail!("Duplicate field name in pattern: {}", name);
            }

            used.insert(name.clone());

            Ok(name)
        }).collect::<Result<Vec<_>>>()?;

        let guard = match guard {
            Some(expr) => Some(self.tr_expr(expr)?.into()),
            None => None,
        };

        Ok(Arc::new(Pattern { subject, params, wildcard, guard }))
    }

    fn tr_expr(&mut self, expr: Expr) -> Result<Expr> {
        expr.map_locals(&mut |name| {
            self.use_local(&name)?;
            Ok(name)
        })
    }
}

impl LabelId {
    const MAGIC: usize = usize::MAX / 2;

    fn postincrement(&mut self) -> Self {
        let current = *self;
        self.0 += 1;
        current
    }

    fn as_task_label(self) -> TaskLabel {
        TaskLabel(self.0 + Self::MAGIC)
    }
}

impl Op {
    /// Convenience method for performing fixups
    fn visit_labels(&mut self, mut f: impl FnMut(&mut TaskLabel) -> Result<()>) -> Result<()> {
        match self {
            Op::PushHandler { label, .. } => {
                f(label)
            },

            Op::CancelHandler { label } => {
                f(label)
            },

            Op::Menu { choices } => {
                // Can't use Arc::make_mut due to missing/broken [_]: Clone impl
                // Instead, clone the contents into a Vec as a workaround
                let mut working_copy = Vec::from(choices.as_ref());

                for MenuItem { target, .. } in working_copy.iter_mut() {
                    f(target)?;
                }

                *choices = working_copy.into();

                Ok(())
            },

            Op::EnterBlock { .. } |
            Op::Jz { .. } |
            Op::Jump { .. } |
            Op::Quote { .. } |
            Op::Tailcall { .. } |
            Op::Trace { .. } |
            Op::Eval { .. } |
            Op::Wait { .. } |
            Op::Return |
            Op::Retire |
            Op::Hcf { .. } => Ok(()),
        }
    }

    fn visit_offsets(&mut self, mut f: impl FnMut(&mut usize) -> Result<()>) -> Result<()> {
        match self {
            Op::EnterBlock { offset } |
            Op::Jz { offset, .. } |
            Op::Jump { offset } => {
                f(offset)
            },

            _ => Ok(()),
        }
    }
}
