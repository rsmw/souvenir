use std::{fmt, sync::Arc};
use std::iter::Peekable;

use anyhow::{Context, Result, anyhow, bail};

use super::token::*;

#[derive(Debug, Default, Eq, PartialEq)]
pub struct Script {
    pub header: Vec<Decl>,
    pub setup: Vec<Stmt>,
    pub pages: Vec<Page>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Page {
    pub label: Arc<str>,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Decl {
    Flag {
        name: String,
    },

    Enum {
        name: String,
        variants: Vec<String>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Stmt {
    Let {
        name: Arc<str>,
        value: Expr,
    },

    If {
        cases: Vec<IfThen>,
        fallback: Vec<Stmt>,
    },

    On {
        pattern: Pattern,
        body: Vec<Stmt>,
    },

    Listen {
        pattern: Pattern,
    },

    BareExpr {
        expr: Expr,
    },

    Assign {
        lhs: Expr,
        op: Option<Binop>,
        rhs: Expr,
    },

    Menu {
        choices: Vec<MenuItem>,
    },

    Quote {
        speaker: Option<Expr>,
        text: Vec<Splice>,
    },

    Music {
        path: String,
    },

    Trace {
        on: bool,
    },

    Goto {
        label: Arc<str>,
    },

    Wait {
        amount: Expr,
    },

    Return,

    Continue,

    Hibernate,

    Bye,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Expr<L=Arc<str>> {
    Local {
        name: L,
    },

    Global {
        name: String,
    },

    Atom {
        name: String,
    },

    Member {
        lhs: Box<Self>,
        name: String,
    },

    FnCall {
        lhs: Box<Self>,
        args: Vec<Self>,
    },

    Int {
        value: u32,
    },

    String {
        value: String,
    },

    Infix {
        lhs: Box<Self>,
        op: Binop,
        rhs: Box<Self>,
    },

    Neg {
        rhs: Box<Self>,
    },

    Not {
        rhs: Box<Self>,
    },

    Paren {
        value: Box<Self>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Splice<L=Arc<str>> {
    Verbatim {
        value: String,
    },

    Expr {
        expr: Expr<L>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Pattern {
    pub variant: String,
    pub params: Vec<String>,
    pub wildcard: bool,
    pub guard: Option<Expr>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Binop {
    Add,
    Sub,
    Div,
    Mul,
    Stitch,
    Less,
    Equal,
    Greater,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IfThen {
    pub guard: Expr,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MenuItem {
    pub prompt: Vec<Splice>,
    pub actions: Vec<Stmt>,
}

pub(crate) struct Parser<'src, I: Iterator<Item=Token<'src>>> {
    pub(crate) src: &'src str,
    pub(crate) input: Peekable<I>,
    pub(crate) last_page: Option<Arc<str>>,
}

#[derive(Copy, Clone, Debug)]
pub struct ErrLocation {
    pub line: usize,
    pub column: usize,
}

enum In {
    ScriptHeader,

    PageHeader {
        after_label: Option<Arc<str>>,
    },

    PageBody {
        label: Arc<str>,
    },

    MenuArm {
        number: usize,
    },
}

impl<'src, I: Iterator<Item=Token<'src>>> Parser<'src, I> {
    pub(crate) fn parse(mut self) -> Result<Script> {
        let (header, setup) = In::ScriptHeader.try_parse(|| {
            self.parse_header()
        })?;

        let mut script = Script {
            header,
            setup,
            pages: Vec::new(),
        };

        while self.input.peek().is_some() {
            let after_label = self.last_page.clone();

            let label = In::PageHeader { after_label }.try_parse(|| {
                self.expect(Tok::LabelMarker)?;
                let label = self.parse_label()?;
                self.expect(Tok::LineBreak)?;
                Ok(label)
            })?;

            let page = In::PageBody { label: label.clone() }.try_parse(|| {
                let label = label.clone();
                let body = self.parse_stmts()?;
                Ok(Page { label, body })
            })?;

            self.last_page = Some(label.clone());
            script.pages.push(page);
        }

        Ok(script)
    }

    /// Consume one token of the expected kind, or raise an error.
    fn expect(&mut self, expected: Tok) -> Result<()> {
        let tok = self.expect_token()?;

        if expected != tok.kind {
            let loc = self.line_and_column(tok.offset);
            bail!("{loc}: Expected \"{}\", got {:?}", expected, tok.as_ref());
        }

        Ok(())
    }

    /// Consume one token and return it. Returns an error on EOF.
    fn expect_token(&mut self) -> Result<Token<'src>> {
        self.input.next().ok_or_else(|| anyhow!("Unexpected end of input"))
    }

    /// If the next token is of the expected kind, consume it and return true.
    /// Otherwise, return false.
    fn munch_one(&mut self, expected: Tok) -> bool {
        let Some(tok) = self.input.peek() else {
            return false;
        };

        if tok.kind == expected {
            self.input.next().unwrap();
            true
        } else {
            false
        }
    }

    fn line_and_column(&self, offset: usize) -> ErrLocation {
        let [mut line, mut column] = [0; 2];

        for (index, ch) in self.src.char_indices() {
            if index >= offset {
                break;
            }

            if let '\n' = ch {
                column = 0;
                line += 1;
            } else {
                column += 1;
            }
        }

        line += 1;
        column += 1;
        ErrLocation { line, column }
    }

    fn parse_header(&mut self) -> Result<(Vec<Decl>, Vec<Stmt>)> {
        let mut header = Vec::<Decl>::new();
        while let Some(tok) = self.input.peek() {
            let decl = match tok.kind {
                Tok::LineBreak => {
                    self.expect_token()?;
                    continue;
                },

                Tok::KwdGlobal => {
                    self.expect_token()?;

                    let next = self.expect_token()?;
                    let Tok::GlobalName(name) = next.kind else {
                        let loc = self.line_and_column(next.offset);
                        bail!("{loc}: Expected name of global");
                    };

                    Decl::Flag { name }
                },

                Tok::KwdEnum => {
                    self.expect_token()?;

                    let next = self.expect_token()?;
                    let Tok::GlobalName(name) = next.kind else {
                        let loc = self.line_and_column(next.offset);
                        bail!("{loc}: Expected name of global");
                    };

                    self.expect(Tok::LineBreak)?;

                    let mut variants = vec![];

                    while let Some(tok) = self.input.peek() {
                        match &tok.kind {
                            Tok::Semicolons => break,

                            Tok::LineBreak => {
                                self.expect(Tok::LineBreak)?;
                                continue;
                            },

                            Tok::Atom(e) => {
                                variants.push(e.clone());
                                self.input.next().unwrap();
                            },

                            _ => {
                                let offset = tok.offset;
                                let loc = self.line_and_column(offset);
                                bail!("{loc}: Expected atom or semicolons")
                            },
                        }
                    }

                    self.expect(Tok::Semicolons)?;

                    Decl::Enum { name, variants }
                },

                _ => break,
            };

            header.push(decl);
        }

        let setup = self.parse_stmts()?;

        Ok((header, setup))
    }

    fn parse_label(&mut self) -> Result<Arc<str>> {
        let next = self.expect_token()?;
        let loc = self.line_and_column(next.offset);
        match next.kind {
            Tok::Ident(w) => Ok(w.into()),
            tok => bail!("{loc}: Expected label, got {}", tok),
        }
    }

    fn parse_stmts(&mut self) -> Result<Vec<Stmt>> {
        let mut stmts = Vec::new();

        loop {
            let Some(next) = self.input.peek() else {
                break;
            };

            match &next.kind {
                Tok::LabelMarker | Tok::Semicolons | Tok::OptionPipe | Tok::KwdElse => break,

                Tok::LineBreak => {
                    self.input.next();
                },

                Tok::GlobalName(_) | Tok::Ident(_) => {
                    let expr = self.parse_expr(0)?;

                    match self.input.peek().map(|t| &t.kind) {
                        Some(&Tok::Assignment(op)) => {
                            let lhs = expr;
                            self.expect(Tok::Assignment(op))?;
                            let rhs = self.parse_expr(0)?;
                            stmts.push(Stmt::Assign { lhs, op, rhs });
                            self.expect(Tok::LineBreak)?;
                        },

                        Some(Tok::QuoteStart) => {
                            let speaker = Some(expr);

                            self.expect(Tok::QuoteStart)?;

                            let text = self.parse_quote()?;

                            stmts.push(Stmt::Quote { speaker, text });
                            self.expect(Tok::LineBreak)?;
                        },

                        // TODO: Multi-line paragraphs

                        _ => {
                            stmts.push(Stmt::BareExpr { expr });
                            self.expect(Tok::LineBreak)?;
                        },
                    }
                },

                _ => {
                    stmts.push(self.parse_stmt()?);
                    self.expect(Tok::LineBreak)?;
                },
            }
        }

        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt> {
        let next = self.expect_token()?;

        Ok(match next.kind {
            Tok::QuoteStart => {
                let text = self.parse_quote()?;
                Stmt::Quote { text, speaker: None }
            },

            Tok::GotoArrow => {
                let label = self.parse_label()?;
                Stmt::Goto { label }
            },

            Tok::KwdLet => {
                let next = self.expect_token()?;

                let Tok::Ident(name) = next.kind else {
                    let loc = self.line_and_column(next.offset);
                    bail!("{loc}: Expected identifier, got {}", next.kind);
                };

                self.expect(Tok::Assignment(None))?;

                let value = self.parse_expr(0)?;

                let name = name.as_str().into();

                Stmt::Let { name, value }
            },

            Tok::KwdIf => {
                let (cases, fallback) = self.parse_if()?;
                Stmt::If { cases, fallback }
            },

            Tok::KwdOn => {
                let pattern = self.parse_pattern()?;

                self.expect(Tok::LineBreak)?;

                let body = self.parse_stmts()?;

                self.expect(Tok::Semicolons)?;

                Stmt::On { pattern, body }
            },

            Tok::KwdListen => {
                let pattern = self.parse_pattern()?;
                Stmt::Listen { pattern }
            },

            Tok::KwdMenu => {
                let choices = self.parse_menu()?;
                Stmt::Menu { choices }
            },

            Tok::KwdMusic => {
                let next = self.expect_token()?;

                let Tok::LitString(path) = &next.kind else {
                    let loc = self.line_and_column(next.offset);
                    bail!("{loc}: Expected string, found {:?}", &next.kind);
                };

                let path = path.clone();

                Stmt::Music { path }
            },

            Tok::KwdTrace => {
                self.expect(Tok::KwdOn)?;
                Stmt::Trace { on: true }
            },

            Tok::KwdWait => {
                let amount = self.parse_expr(0)?;
                Stmt::Wait { amount }
            },

            Tok::KwdReturn => {
                Stmt::Return
            },

            Tok::KwdContinue => {
                Stmt::Continue
            },

            Tok::KwdHibernate => {
                Stmt::Hibernate
            },

            Tok::KwdBye => {
                Stmt::Bye
            },

            other => {
                let loc = self.line_and_column(next.offset);
                bail!("{loc}: Unexpected {}", other)
            },
        })
    }

    fn parse_expr(&mut self, min_bp: u32) -> Result<Expr> {
        let mut lhs = self.parse_pre_expr()?;

        while self.input.peek().is_some() {
            if let Some(bp) = self.peek_postfix_power() {
                if bp < min_bp {
                    break;
                }

                let tok = self.expect_token()?;
                let loc = self.line_and_column(tok.offset);

                lhs = match tok.kind {
                    Tok::Dot => {
                        let tok = self.expect_token()?;

                        let Tok::Ident(name) = tok.kind else {
                            bail!("Expected field name");
                        };

                        let lhs = lhs.into();

                        Expr::Member { lhs, name }
                    },

                    Tok::ParenOpen => {
                        let mut args = vec![];

                        loop {
                            let Some(next) = self.input.peek() else {
                                bail!("{loc}: Unexpected end of input");
                            };

                            if let Tok::ParenClose = &next.kind {
                                break;
                            }

                            if !args.is_empty() {
                                self.expect(Tok::LineBreak)?;
                            }

                            args.push(self.parse_expr(0)?);
                        }

                        self.expect(Tok::ParenClose)?;

                        let lhs = lhs.into();

                        Expr::FnCall { lhs, args }
                    },

                    other => bail!("{loc}: Unrecognized operator {}", other),
                };
            } else if let Some((lbp, rbp)) = self.peek_infix_power() {
                if lbp < min_bp {
                    break;
                }

                let tok = self.expect_token()?;
                let rhs = self.parse_expr(rbp)?;

                lhs = match tok.kind {
                    Tok::Plus => Binop::Add.with(lhs, rhs),

                    Tok::Minus => Binop::Sub.with(lhs, rhs),

                    Tok::Splat => Binop::Mul.with(lhs, rhs),

                    Tok::Slash => Binop::Div.with(lhs, rhs),

                    Tok::Stitch => Binop::Stitch.with(lhs, rhs),

                    Tok::Less => Binop::Less.with(lhs, rhs),

                    Tok::Equal => Binop::Equal.with(lhs, rhs),

                    Tok::Greater => Binop::Greater.with(lhs, rhs),

                    other => {
                        let loc = self.line_and_column(tok.offset);
                        bail!("{loc}: Unknown operator {:?}", other)
                    },
                };
            } else {
                break;
            }
        }

        Ok(lhs)
    }

    fn parse_pre_expr(&mut self) -> Result<Expr> {
        if let Some(rbp) = self.peek_prefix_power() {
            let tok = self.expect_token()?;

            let rhs = self.parse_expr(rbp)?.into();

            return Ok(match tok.kind {
                Tok::KwdNot => Expr::Not { rhs },
                Tok::Minus => Expr::Neg { rhs },
                other => {
                    let loc = self.line_and_column(tok.offset);
                    bail!("{loc}: Unexpected {:?}", other)
                },
            })
        }

        self.parse_atomic_expr()
    }

    fn parse_atomic_expr(&mut self) -> Result<Expr> {
        let tok = self.expect_token()?;

        let lhs = match tok.kind {
            Tok::Ident(name) => {
                let name = name.as_str().into();
                Expr::Local { name }
            },

            Tok::GlobalName(name) => {
                let name = name.clone();
                Expr::Global { name }
            },

            Tok::Atom(name) => {
                let name = name.clone();
                Expr::Atom { name }
            },

            Tok::LitInt(value) => {
                Expr::Int { value }
            },

            Tok::LitString(value) => {
                Expr::String { value }
            },

            Tok::ParenOpen => {
                let value = self.parse_expr(0)?.into();
                self.expect(Tok::ParenClose)?;
                Expr::Paren { value }
            },

            _ => {
                let loc = self.line_and_column(tok.offset);
                bail!("{loc}: Expected expression, got {}", tok.kind)
            },
        };

        Ok(lhs)
    }

    fn peek_prefix_power(&mut self) -> Option<u32> {
        Some(match self.input.peek()?.kind {
            Tok::KwdNot => 90,
            Tok::Minus => 100,
            _ => return None,
        })
    }

    fn peek_infix_power(&mut self) -> Option<(u32, u32)> {
        Some(match self.input.peek()?.kind {
            Tok::Less | Tok::Equal | Tok::Greater => (30, 31),
            Tok::Plus | Tok::Minus => (40, 41),
            Tok::Splat | Tok::Slash => (50, 51),
            Tok::Stitch => (20, 21),
            _ => return None,
        })
    }

    fn peek_postfix_power(&mut self) -> Option<u32> {
        Some(match self.input.peek()?.kind {
            Tok::Dot => 60,
            Tok::ParenOpen => 100,
            _ => return None,
        })
    }

    fn parse_if(&mut self) -> Result<(Vec<IfThen>, Vec<Stmt>)> {
        let mut cases = vec![];
        let mut fallback = vec![];

        loop { // KwdIf already eaten by this point
            let guard = self.parse_expr(0)?;
            self.expect(Tok::LineBreak)?;

            let body = self.parse_stmts()?;

            cases.push(IfThen { guard, body });

            if !self.munch_one(Tok::KwdElse) {
                break;
            }

            if self.munch_one(Tok::KwdIf) {
                continue;
            }

            fallback = self.parse_stmts()?;
            break;
        }

        self.expect(Tok::Semicolons)?;

        Ok((cases, fallback))
    }

    fn parse_pattern(&mut self) -> Result<Pattern> {
        let variant = self.expect_token()?;

        let Tok::Atom(variant) = variant.kind else {
            let loc = self.line_and_column(variant.offset);
            bail!("{loc}: Expected atom, found {}", variant.kind);
        };

        let mut params = Vec::new();
        let mut wildcard = false;
        let mut guard = None;

        if self.munch_one(Tok::CurlyOpen) {
            while let Some(delim) = self.input.peek() {
                if delim.kind == Tok::CurlyClose {
                    break;
                }

                if !params.is_empty() {
                    self.expect(Tok::LineBreak)?;

                    while self.munch_one(Tok::LineBreak) {
                        continue;
                    }
                }

                let tok = self.expect_token()?;

                if tok.kind == Tok::Ellipsis {
                    if wildcard {
                        let loc = self.line_and_column(tok.offset);
                        bail!("{loc}: Multiple wildcards in pattern");
                    }

                    wildcard = true;
                    continue;
                }

                let Tok::Ident(name) = &tok.kind else {
                    let loc = self.line_and_column(tok.offset);
                    bail!("{loc}: Expected identifier, found {}", &tok.kind)
                };

                params.push(name.clone());
            }

            self.expect(Tok::CurlyClose)?;

            if self.munch_one(Tok::KwdIf) {
                guard = Some(self.parse_expr(0)?);
            }
        }

        Ok(Pattern { variant, params, wildcard, guard })
    }

    fn parse_menu(&mut self) -> Result<Vec<MenuItem>> {
        let mut items = vec![];

        loop {
            let next = self.expect_token()?;
            match next.kind {
                Tok::Semicolons => break,

                Tok::LineBreak => continue,

                Tok::OptionPipe => (),

                other => {
                    let loc = self.line_and_column(next.offset);
                    bail!("{loc}: Unexpected {}", other)
                },
            }

            In::MenuArm { number: items.len() }.try_parse(|| {
                let next = self.expect_token()?;
                match next.kind {
                    Tok::QuoteStart => {
                        let prompt = self.parse_quote()?;
                        self.expect(Tok::LineBreak)?;

                        let actions = self.parse_stmts()?;

                        items.push(MenuItem { prompt, actions })
                    },

                    other => {
                        let loc = self.line_and_column(next.offset);
                        bail!("{loc}: Unexpected {}", other)
                    },
                }

                Ok(())
            })?;

            while let Some(Tok::LineBreak) = self.input.peek().map(|t| &t.kind) {
                self.input.next();
            }
        }

        Ok(items)
    }

    fn parse_quote(&mut self) -> Result<Vec<Splice>> {
        // Assume QuoteStart is already consumed

        let mut splices = Vec::<Splice>::new();

        while let Some(tok) = self.input.peek() {
            match &tok.kind {
                Tok::LineBreak => break,

                Tok::QuoteVerbatim(value) => {
                    let value = value.clone();
                    splices.push(Splice::Verbatim { value });
                    self.input.next();
                },

                Tok::CurlyOpen => {
                    self.expect(Tok::CurlyOpen)?;

                    splices.push(Splice::Expr {
                        expr: self.parse_expr(0)?,
                    });

                    self.expect(Tok::CurlyClose)?;
                },

                _ => break,
            }
        }

        Ok(splices)
    }
}

impl Binop {
    fn with(self, lhs: Expr, rhs: Expr) -> Expr {
        let op = self;
        let [lhs, rhs] = [lhs.into(), rhs.into()];
        Expr::Infix { lhs, op, rhs }
    }
}

impl<L> Expr<L> {
    pub fn map_locals<M>(self, f: &mut impl FnMut(L) -> Result<M>) -> Result<Expr<M>> {
        use Expr::*;

        Ok(match self {
            Local { name } => {
                let name = f(name)?;
                Local { name }
            },

            Member { lhs, name } => {
                let lhs = lhs.map_locals(f)?.into();
                Member { lhs, name }
            },

            Infix { lhs, op, rhs } => {
                let lhs = lhs.map_locals(f)?.into();
                let rhs = rhs.map_locals(f)?.into();
                Infix { lhs, op, rhs }
            },

            Neg { rhs } => {
                let rhs = rhs.map_locals(f)?.into();
                Neg { rhs }
            },

            Not { rhs } => {
                let rhs = rhs.map_locals(f)?.into();
                Not { rhs }
            },

            Paren { value } => {
                let value = value.map_locals(f)?.into();
                Paren { value }
            },

            Global { name } => Global { name },
            Atom { name } => Atom { name },
            Int { value } => Int { value },
            String { value } => String { value },

            FnCall { .. } => {
                bail!("Unimplemented: Function calls in expressions");
            },
        })
    }
}

impl<L> Splice<L> {
    pub fn map_exprs<M>(self, f: &mut impl FnMut(Expr<L>) -> Result<Expr<M>>) -> Result<Splice<M>> {
        use Splice::*;

        Ok(match self {
            Expr { expr } => {
                let expr = f(expr)?;
                Expr { expr }
            },

            Verbatim { value } => Verbatim { value },
        })
    }
}

impl In {
    fn try_parse<T>(self, mut f: impl FnMut() -> Result<T>) -> Result<T> {
        f().with_context(move || self)
    }
}

impl fmt::Display for In {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            In::ScriptHeader => write!(f, "At the beginning of the script"),

            In::PageHeader { after_label } => match after_label {
                Some(label) => write!(f, "In the header after page {:?}", label),
                None => write!(f, "In the first page header"),
            },

            In::PageBody { label } => {
                write!(f, "In the body of page {:?}", label)
            },

            In::MenuArm { number } => {
                write!(f, "In arm {} of a menu statement", number + 1)
            },
        }
    }
}

impl fmt::Display for ErrLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ErrLocation { line, column } = self;
        write!(f, "Line {line}, column {column}")
    }
}

impl fmt::Display for Binop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", match self {
            Binop::Add => "+",
            Binop::Sub => "-",
            Binop::Mul => "*",
            Binop::Div => "/",
            Binop::Stitch => "<>",
            Binop::Less => "?<",
            Binop::Equal => "?=",
            Binop::Greater => "?>",
        })
    }
}

impl<'a> From<&'a str> for Splice {
    fn from(value: &'a str) -> Self {
        let value = value.into();
        Self::Verbatim { value }
    }
}

impl From<Expr> for Splice {
    fn from(expr: Expr) -> Self {
        Self::Expr { expr }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn example_stmts() {
        "Player > This chair;".should_be(Stmt::Quote {
            speaker: Some(Expr::Local { name: "Player".into() }),
            text: vec![
                Splice::Verbatim {
                    value: "This chair;".into(),
                },
            ],
        });

        "Foo = Bar -- Comments should be ignored".should_be(Stmt::Assign {
            lhs: Expr::Local { name: "Foo".into() },
            op: None,
            rhs: Expr::Local { name: "Bar".into() },
        });

        "$foo += $bar".should_be(Stmt::Assign {
            lhs: Expr::Global { name: "foo".into() },
            op: Some(Binop::Add),
            rhs: Expr::Global { name: "bar".into() },
        });

        "$status = #rekt".should_be(Stmt::Assign {
            lhs: Expr::Global { name: "status".into() },
            op: None,
            rhs: Expr::Atom { name: "rekt".into() },
        });

        "let Status = #rekt".should_be(Stmt::Let {
            name: "Status".into(),
            value: Expr::Atom { name: "rekt".into() },
        });

        "X = 2 + 2".should_be(Stmt::Assign {
            lhs: Expr::Local { name: "X".into() },
            op: None,
            rhs: Expr::Infix {
                lhs: Expr::Int { value: 2 }.into(),
                op: Binop::Add,
                rhs: Expr::Int { value: 2 }.into(),
            }
        });

        "X /= (1 + (1))".should_be(Stmt::Assign {
            lhs: Expr::Local{ name: "X".into() },
            op: Some(Binop::Div),
            rhs: Expr::Paren {
                value: Expr::Infix {
                    lhs: Expr::Int { value: 1 }.into(),
                    op: Binop::Add,
                    rhs: Expr::Paren {
                        value: Expr::Int { value: 1 }.into(),
                    }.into(),
                }.into(),
            }.into(),
        });

        "X + - 2".should_be(Stmt::BareExpr {
            expr: Expr::Infix {
                lhs: Expr::Local { name: "X".into() }.into(),
                op: Binop::Add,
                rhs: Expr::Neg {
                    rhs: Expr::Int { value: 2 }.into(),
                }.into(),
            },
        });

        "X * (2 + 2)".should_be(Stmt::BareExpr {
            expr: Expr::Infix {
                lhs: Expr::Local { name: "X".into() }.into(),
                op: Binop::Mul,
                rhs: Expr::Paren {
                    value: Expr::Infix {
                        lhs: Expr::Int { value: 2 }.into(),
                        op: Binop::Add,
                        rhs: Expr::Int { value: 2 }.into(),
                    }.into(),
                }.into(),
            },
        });

        "foo()".should_be(Stmt::BareExpr {
            expr: Expr::FnCall {
                lhs: Expr::Local { name: "foo".into() }.into(),
                args: vec![],
            }.into(),
        });

        "foo(Bar)".should_be(Stmt::BareExpr {
            expr: Expr::FnCall {
                lhs: Expr::Local { name: "foo".into() }.into(),
                args: vec![
                    Expr::Local { name: "Bar".into() }.into(),
                ],
            }.into(),
        });
    }

    trait ParseExample<T> {
        fn should_be(&self, expected: T);
    }

    impl ParseExample<Script> for str {
        fn should_be(&self, expected: Script) {
            let thing = crate::parse(self).expect("Failed to parse example");
            assert_eq!(expected, thing);
        }
    }

    impl ParseExample<Stmt> for str {
        fn should_be(&self, expected: Stmt) {
            let expected = Script {
                header: vec![],
                setup: vec![expected],
                pages: vec![],
            };
            let source = format!("{}\n\n", self);
            let parsed = crate::parse(&source).expect("Failed to parse example");
            assert_eq!(expected, parsed);
        }
    }
}
