//! Evaluation module. Defines expression evaluation and the structure of the
//! lexical environment.
//! 
//! Evaluation is currently implemented as recursive AST walking. In the future,
//! this can be optimized using closures for partial evaluation, or (on
//! applicable platforms) full JIT compilation, as long as the state of a
//! suspended program remains amenable to platform-independent serialization.

use std::collections::{HashMap, BTreeMap};
use std::sync::{Arc, RwLock};

use anyhow::{self, bail, Result};

use crate::ast::{Expr, Splice};
use crate::interpret::Pattern;
use crate::value::Value;

//pub type Result<T, E=EvalErr> = std::result::Result<T, E>;

#[derive(Clone, Debug)]
pub enum EvalErr {

}

#[derive(Debug)]
pub(crate) enum GlobalValue {
    Int(u64),

    // TODO: Other variants
}

/// Container for a single node in the lexical environment graph.
#[derive(Debug)]
pub(crate) struct LocalEnv {
    locals: HashMap<Arc<str>, Value>,

    parent: Option<EnvHandle>,

    // TODO: Refcount
}

/// Handle to shared global state.
#[derive(Clone)]
pub struct GlobalHandle(Arc<RwLock<GlobalEnv>>);

#[derive(Debug)]
struct GlobalEnv {
    bindings: HashMap<Arc<str>, GlobalValue>,
}

/// Container for the entire lexical environment for one Actor.
#[derive(Debug)]
pub(crate) struct EnvHeap {
    envs: BTreeMap<EnvHandle, LocalEnv>,

    next: usize,

    globals: Arc<RwLock<GlobalEnv>>,
}

/// Handle for a `LocalDict` owned by EnvStack.
#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct EnvHandle(usize);

/// Temporary context for expression evaluation
pub(crate) struct EvalContext<'a> {
    heap: &'a EnvHeap,
    handle: EnvHandle,
}

/// Like `EvalContext`, but mutable, for creating/mutating bindings
pub(crate) struct BindContext<'a> {
    heap: &'a mut EnvHeap,
    handle: &'a mut EnvHandle,
}

/// Description of where to place the result of evaluating an expression.
#[derive(Clone, Debug)]
pub enum Placement {
    /// Discard the result instead of saving it.
    Discard,

    /// An existing local variable already visible in the current scope.
    UpdateLocal {
        name: Arc<str>,
    },

    /// A new local variable; may shadow variables with the same name.
    CreateLocal {
        name: Arc<str>,
    },

    /// Actor property. Not yet implemented.
    Prop {
        name: Arc<str>,
    },

    /// Persistent global variable.
    Global {
        name: Arc<str>,
    },
}

impl EnvHeap {
    pub(crate) fn with_globals(globals: GlobalHandle) -> Self {
        Self {
            envs: Default::default(),
            next: 0,
            globals: globals.0,
        }
    }

    pub(crate) fn create(&mut self) -> EnvHandle {
        let handle = EnvHandle(self.next);

        self.next += 1;

        self.envs.insert(handle, LocalEnv {
            locals: Default::default(),
            parent: None,
        });

        handle
    }

    pub(crate) fn create_child(&mut self, parent: EnvHandle) -> EnvHandle {
        self.retain(parent);

        let handle = self.create();

        self.get_mut(handle).parent = Some(parent);

        handle
    }

    pub(crate) fn retain(&mut self, handle: EnvHandle) {
        let _env = self.get_mut(handle);
        //env.refcount += 1;
    }

    pub(crate) fn release(&mut self, handle: EnvHandle) {
        // Sketch of refcount tracing

        // loop {
        //     let env = self.get_mut(handle);
        //     env.refcount = env.refcount.saturating_sub(1);
        //     if env.refcount > 0 {
        //         break;
        //     }
        //     parent = env.parent;
        //     self.envs.remove(&handle);
        //     let Some(parent) = parent else {
        //         break;
        //     };
        //     handle = parent;
        // }

        self.envs.remove(&handle);
    }

    pub(crate) fn get_mut(&mut self, env: EnvHandle) -> &mut LocalEnv {
        self.envs.get_mut(&env).unwrap()
    }

    pub(crate) fn clear(&mut self) {
        self.envs.clear();
        self.next = 0;
    }

    pub(crate) fn lookup_global(&self, name: &str) -> Result<Value> {
        let globals = self.globals.read().or_else(|_| {
            bail!("Cannot access global dict");
        })?;

        let Some(value) = globals.bindings.get(name) else {
            bail!("No such global: ${name}");
        };

        match value {
            &GlobalValue::Int(n) if n >= (i64::MAX as u64) => {
                bail!("Value of global ${name} is too large to read: {n}");
            },

            &GlobalValue::Int(n) => Ok(Value::Int(n as i64)),
        }
    }

    pub(crate) fn eval(&self, expr: &Expr, env: EnvHandle) -> Result<Value> {
        self.ctx(env).eval(expr)
    }

    pub(crate) fn stitch(&self, splices: &[Splice], env: EnvHandle) -> Result<String> {
        self.ctx(env).stitch(splices)
    }

    pub(crate) fn ctx<'a>(&'a self, env: EnvHandle) -> EvalContext<'a> {
        EvalContext { heap: self, handle: env }
    }

    pub(crate) fn ctx_mut<'a>(&'a mut self, env: &'a mut EnvHandle) -> BindContext<'a> {
        BindContext { heap: self, handle: env }
    }
}

impl<'a> EvalContext<'a> {
    pub(crate) fn lookup(&self, name: &str) -> Result<Value> {
        let mut env = &self.heap.envs[&self.handle];

        loop {
            if let Some(value) = env.locals.get(name) {
                return Ok(value.clone());
            }

            if let Some(parent) = env.parent {
                env = &self.heap.envs[&parent];
                continue;
            }

            bail!("No such variable: {name}")
        }
    }

    pub(crate) fn eval(&self, expr: &Expr) -> Result<Value> {
        Ok(match expr {
            Expr::Local { name } => self.lookup(name)?,

            Expr::Global { name } => self.heap.lookup_global(name)?,

            Expr::Atom { name } => Value::Atom(name.as_str().into()),

            Expr::Member { .. } => bail!("TODO: Field access"),

            Expr::FnCall { .. } => bail!("TODO: Pure functions"),

            &Expr::Int { value } => Value::Int(value.into()),

            Expr::String { value } => Value::String(value.clone()),

            Expr::Infix { lhs, op, rhs } => {
                let lhs = self.eval(lhs)?;
                let rhs = self.eval(rhs)?;

                use crate::ast::Binop::*;
                use self::Value::*;

                match (op, lhs, rhs) {
                    (Add, Int(lhs), Int(rhs)) => Int(lhs + rhs),

                    (Sub, Int(lhs), Int(rhs)) => Int(lhs - rhs),

                    (Mul, Int(lhs), Int(rhs)) => Int(lhs * rhs),

                    (Div, lhs, Int(0)) => bail!("Tried to divide {lhs:?} by zero"),

                    (Less, Int(lhs), Int(rhs)) => Value::from(lhs < rhs),

                    (Greater, Int(lhs), Int(rhs)) => Value::from(lhs > rhs),

                    (Equal, lhs, rhs) => Value::from(lhs == rhs),

                    (Stitch, lhs, rhs) => {
                        use std::string::String;
                        let mut lhs = String::try_from(lhs)?;
                        lhs.push_str(&String::try_from(rhs)?);
                        Value::String(lhs)
                    },

                    _ => bail!("Can't eval {expr}"),
                }
            },

            Expr::Neg { rhs } => {
                match self.eval(rhs)? {
                    Value::Int(n) => Value::Int(-n),
                    other => bail!("Not negatable: {other:?}"),
                }
            },

            Expr::Not { rhs } => {
                let value = self.eval(rhs)?;
                Value::from(!value.is_truthy())
            },

            Expr::Paren { value } => self.eval(value)?,
        })
    }

    /// Evaluate a sequence of `Splice`s and concatenate the results
    pub fn stitch(&self, splices: &[Splice]) -> Result<String> {
        let mut buf = String::new();

        for splice in splices {
            match splice {
                Splice::Verbatim { value } => {
                    buf.push_str(value.as_str());
                },

                Splice::Expr { expr } => {
                    let s: String = self.eval(expr)?.try_into()?;
                    buf.push_str(&s);
                },
            }
        }

        Ok(buf)
    }
}

impl<'a> BindContext<'a> {
    pub(crate) fn as_eval<'b>(&'b self) -> EvalContext<'b> {
        EvalContext {
            heap: &self.heap,
            handle: *self.handle,
        }
    }

    pub(crate) fn bind(&mut self, dst: Placement, value: Value) -> Result<()> {
        match dst {
            Placement::Discard => (),

            Placement::CreateLocal { name } => {
                let parent = *self.handle;
                *self.handle = self.heap.create_child(parent);

                // Skip rebind check because the child env is empty

                let env = self.heap.get_mut(*self.handle);
                env.locals.insert(name, value);
            },

            Placement::Global { name } => {
                self.update_global(&name, value)?;
            },

            Placement::UpdateLocal { name } => {
                self.update_local(&name, value)?;
            },

            Placement::Prop { name } => {
                bail!("TODO: Set actor-local property {name} to {value:?}")
            },
        }

        Ok(())
    }

    pub(crate) fn bind_pattern(&mut self, pat: &Pattern, args: HashMap<Arc<str>, Value>) -> Result<bool> {
        let Pattern { ref params, wildcard, ref guard, .. } = pat;

        let mut args = args.clone();

        for name in params.iter() {
            let Some(value) = args.remove(name) else {
                bail!("Argument {name} is missing");
            };

            let env = self.heap.get_mut(*self.handle);

            if let Some(old) = env.locals.insert(name.clone(), value) {
                bail!("Binding {name:?} already existed with value {old:?}");
            }
        }

        if args.len() > 0 && !wildcard {
            bail!("Extra fields not caught by wildcard: {args:?}");
        }

        if let Some(expr) = guard {
            let result = self.as_eval().eval(expr)?;

            if !result.is_truthy() {
                bail!("Guard expression returned {result:?}");
            }
        }

        Ok(true)
    }

    pub(crate) fn update_local(&mut self, name: &str, value: Value) -> Result<()> {
        let mut handle = *self.handle;

        loop {
            let env = self.heap.get_mut(handle);

            if let Some(slot) = env.locals.get_mut(name) {
                *slot = value;
                return Ok(());
            } else if let Some(parent) = env.parent {
                handle = parent;
            } else {
                bail!("No such local: {name}");
            }
        }
    }

    pub(crate) fn update_global(&mut self, name: &str, value: Value) -> Result<()> {
        let value = GlobalValue::try_from(value)?;

        let mut globals = self.heap.globals.write().unwrap();

        let Some(slot) = globals.bindings.get_mut(name) else {
            bail!("No such global: ${name}");
        };

        *slot = value;

        Ok(())
    }
}

impl LocalEnv {
    pub(crate) fn reset(&mut self, parent: EnvHandle) {
        self.locals.clear();
        self.parent = Some(parent);
    }
}

impl GlobalHandle {
    pub fn with_values(values: &[(&str, Value)]) -> Result<Self> {
        let mut bindings = HashMap::default();

        for (name, value) in values.iter().cloned() {
            let value = GlobalValue::try_from(value)?;
            bindings.insert(name.into(), value);
        }

        let env = GlobalEnv { bindings };
        Ok(Self(Arc::new(env.into())))
    }
}

impl TryFrom<Value> for GlobalValue {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self> {
        match value {
            Value::Int(n) if n >= 0 => {
                Ok(GlobalValue::Int(n as _))
            },

            other => bail!("Cannot store {other:?} in a global"),
        }
    }
}

#[cfg(test)]
pub mod dsl {
    use super::*;

    use crate::ast::Binop;

    pub fn infix(op: Binop, lhs: impl Into<Box<Expr>>, rhs: impl Into<Box<Expr>>) -> Expr {
        let lhs = lhs.into();
        let rhs = rhs.into();
        Expr::Infix { op, lhs, rhs }
    }

    pub fn add(lhs: impl Into<Box<Expr>>, rhs: impl Into<Box<Expr>>) -> Expr {
        infix(Binop::Add, lhs, rhs)
    }

    pub fn sub(lhs: impl Into<Box<Expr>>, rhs: impl Into<Box<Expr>>) -> Expr {
        infix(Binop::Sub, lhs, rhs)
    }

    pub fn div(lhs: impl Into<Box<Expr>>, rhs: impl Into<Box<Expr>>) -> Expr {
        infix(Binop::Div, lhs, rhs)
    }

    pub fn mul(lhs: impl Into<Box<Expr>>, rhs: impl Into<Box<Expr>>) -> Expr {
        infix(Binop::Mul, lhs, rhs)
    }

    pub fn neg(rhs: impl Into<Box<Expr>>) -> Expr {
        let rhs = rhs.into();
        Expr::Neg { rhs }
    }

    pub fn local(name: impl Into<Arc<str>>) -> Expr {
        let name = name.into();
        Expr::Local { name }
    }

    pub fn global(name: impl Into<String>) -> Expr {
        let name = name.into();
        Expr::Global { name }
    }

    impl From<u32> for Expr {
        fn from(value: u32) -> Self {
            Expr::Int { value }
        }
    }

    impl From<u32> for Box<Expr> {
        fn from(value: u32) -> Self {
            Box::new(value.into())
        }
    }
}

#[test]
fn hello_world() {
    use dsl::*;

    let globals = GlobalHandle::with_values(&[("ONE", Value::Int(3))]).unwrap();

    let mut heap = EnvHeap::with_globals(globals.clone());

    let mut locals = heap.create();

    heap.ctx_mut(&mut locals).update_global("ONE", Value::Int(1)).unwrap();

    heap.get_mut(locals).locals.insert("X".into(), Value::Int(7));

    assert_eq!(heap.eval(&global("ONE"), locals).unwrap(), Value::Int(1));

    let example = {
        mul(neg(global("ONE")), add(2, 2))
    };

    assert_eq!(heap.eval(&example, locals).unwrap(), Value::Int(-4));

    let example2 = {
        sub(mul(3, 3), 1)
    };

    assert_eq!(heap.eval(&example2, locals).unwrap(), Value::Int(8));

    let example3 = {
        div(1, 0)
    };

    assert_eq!(heap.eval(&example3, locals).expect_err("Must fail").to_string(), "Tried to divide Int(1) by zero");

    assert_eq!(heap.eval(&local("X"), locals).unwrap(), Value::Int(7));
}

#[test]
fn compare_ints() {
    use dsl::*;
    use crate::ast::Binop;

    let example = {
        infix(Binop::Less, 2, 3)
    };

    let globals = GlobalHandle::with_values(&[]).unwrap();
    let mut heap = EnvHeap::with_globals(globals);
    let locals = heap.create();

    assert_eq!(heap.eval(&example, locals).unwrap(), Value::Int(1));
}