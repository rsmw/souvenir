//! Core interpreter module. Handles evaluation for one actor. Current
//! architecture revolves around async scheduling.
//! 
//! Lexical environments are dynamically managed and allocated, replacing prior
//! attempts to allocate them statically. This is slower, but removes numerous
//! corner cases that broke in earlier implementations.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Result};

use log::debug;

use crate::ast::{Expr, Splice};

use crate::eval::{EnvHandle, EnvHeap, GlobalHandle, Placement};
use crate::value::{ActorRef, Value};

/// An Actor is a self-contained Souvenir script interpreter, containing all
/// state necessary for evaluating expressions and handling incoming messages.
/// The host environment is responsible for passing messages between Actors,
/// fulfilling FFI calls, and all other system IO.
/// 
/// Internally, the state of an ongoing computation is represented by a `Task`
/// and kept in a FILO stack. When an Actor is prompted to continue by the
/// `tick` method, only the topmost task will run, and only if it is not waiting
/// for the completion of an IO request.
#[derive(Debug)]
pub struct Actor {
    liveness: ActorLiveness,

    script: Arc<Script>,

    /// All extant tasks, whether running or suspended.
    /// Uses Vec as a FILO stack. Only the top task will run.
    tasks: Vec<Task>,

    /// The entire lexical environment, including globals
    heap: EnvHeap,

    handlers: HashMap<Arc<str>, Vec<HandlerClosure>>,

    /// Queue for pending IO requests.
    outbox: VecDeque<IoReq>,

    /// Generates secrets for IO requests.
    secrets: Secrets,

    /// Controls verbose print debugging.
    trace: bool,
}

/// Compiled script.
#[derive(Debug)]
pub struct Script {
    pub(crate) pages: HashMap<Arc<str>, PageInfo>,

    /// Stores all opcodes in a single linear array.
    /// `TaskLabel` is an index into this array.
    pub(crate) body: Arc<[Op]>,
}

#[derive(Debug)]
pub struct PageInfo {
    pub(crate) entry_point: TaskLabel,

    #[allow(unused)]
    pub(crate) params: Arc<[Arc<str>]>,
}

/// Target of a jump within a script body.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct TaskLabel(pub(crate) usize);

/// Represents an in-progress IO request for the host environment.
/// 
/// **WARNING:** Must not implement Copy or Clone.
#[derive(Debug)]
pub struct IoReq {
    /// Index of the task that sent the request.
    task_index: usize,

    /// Opaque token identifying the individual request.
    secret: usize,

    payload: IoPayload,
}

#[derive(Debug)]
pub enum IoPayload {
    /// Text to display
    Quote {
        speaker: Option<ActorRef>,
        line: String,
    },

    Menu {
        items: Vec<String>,
    },

    /// Any other call to a host-defined function.
    HostFfi {
        fn_name: Arc<str>,

        // TODO: Named params?
        args: Vec<Value>,
    },
}

/// Flattened instruction format.
/// 
/// Includes instructions that perform control flow. The basic block model does
/// not work here. I tried.
#[derive(Clone, Debug)]
pub enum Op {
    /// Evaluate an expression and optionally save the result.
    Eval {
        expr: Arc<Expr>,
        dst: Placement,
    },

    /// Execute the next instruction in a child Task and jump forward by the
    /// specified offset.
    EnterBlock {
        offset: usize,
    },

    /// Install a message handler. See `Actor::handle_message`.
    PushHandler {
        /// Pattern to try binding.
        pattern: Arc<Pattern>,

        /// Label of body to execute if binding succeeds.
        label: TaskLabel,

        /// If true, automatically remove this handler the first time it fires.
        cancel: bool,
    },

    /// Remove any installed handler with a matching label.
    CancelHandler {
        label: TaskLabel,
    },

    // Listen -- replaced with PushHandler { cancel: true, .. }

    /// Evaluate `guard`; if not truthy, jump by `offset`. Always jumps forward.
    Jz {
        guard: Arc<Expr>,
        offset: usize,
    },

    /// Unconditional forward jump.
    Jump {
        offset: usize,
    },

    /// Prompt the player for input.
    Menu {
        choices: Arc<[MenuItem]>,
    },

    /// Display text to the player.
    Quote {
        speaker: Option<Arc<Expr>>,
        lines: Arc<[Splice]>,
    },

    // Music -- replaced with FFI

    /// Enable or disable verbose debugging for this actor
    Trace {
        on: bool,
    },

    /// Restart this actor with a different script
    Tailcall {
        path: Arc<str>,

        args: Arc<[Expr]>,
    },

    /// Put this task to sleep
    Wait {
        amount: Arc<Expr>,
    },

    /// End this Task and resume execution of the next one, if ready.
    Return,

    /// Shut down this actor
    Retire,

    /// Throw an error and self destruct
    Hcf {
        reason: Arc<[Splice]>,
    },
}

#[derive(Clone, Debug)]
pub struct Pattern {
    pub subject: Arc<str>,

    /// Fields to bind
    pub params: Vec<Arc<str>>,

    /// Allow extra fields in message to be ignored
    pub wildcard: bool,

    /// Expression to filter incoming messages
    pub guard: Option<Arc<Expr>>,
}

#[derive(Clone, Debug)]
pub struct MenuItem {
    /// Test to decide whether to discard this option.
    pub guard: Option<Arc<Expr>>,

    /// Text to show the player
    pub prompt: Arc<[Splice]>,

    /// Where to jump if the player picks this option.
    pub target: TaskLabel,
}

/// Publicly visible state of this Actor and its topmost Task.
#[derive(Debug)]
pub enum ActorStatus<'a> {
    Loading {
        path: &'a str,

        args: &'a [Value],
    },

    Running,

    Sleeping {
        time_left: Duration,
    },

    Blocked,

    Retiring,

    Killed {
        error: &'a str,
    },

    /// There are no running Tasks
    Hibernating,
}

/// Internal representation of liveness
#[derive(Debug)]
enum ActorLiveness {
    Loading {
        path: Arc<str>,

        args: Vec<Value>,
    },

    Running,

    Killed {
        error: String,
    },

    Retiring,

    // Hibernating -- redundant. An Actor is hibernating if no tasks are ready.
}

/// Internal representation of the status of a Task.
#[derive(Clone, Debug)]
enum TaskStatus {
    Ready,

    Sleeping {
        time_left: Duration,
    },

    AwaitFfi {
        dst: Placement,

        // OK to clone; this type is private
        secret: usize,
    },

    AwaitMenu {
        labels: Vec<TaskLabel>,

        // OK to clone; this type is private
        secret: usize,
    },
}

#[derive(Debug)]
struct Task {
    /// Index of next instruction to execute.
    pc: usize,

    env: EnvHandle,

    /// When a task is Ready, it executes immediately.
    status: TaskStatus,
}

#[derive(Clone, Debug)]
struct HandlerClosure {
    pattern: Arc<Pattern>,
    label: TaskLabel,
    cancel: bool,
    env: EnvHandle,
}

#[derive(Debug)]
struct Secrets {
    next: usize,
}

impl Actor {
    /// Create an actor with the given script in the given global environment.
    pub fn from_script(script: Arc<Script>, globals: GlobalHandle) -> Self {
        let mut heap = EnvHeap::with_globals(globals);

        Actor {
            liveness: ActorLiveness::Running,
            script: script.clone(),
            tasks: vec![Task {
                env: heap.create(),
                pc: 0,
                status: TaskStatus::Ready,
            }],
            heap,
            handlers: Default::default(),
            outbox: Default::default(),
            secrets: Secrets { next: 0 },
            trace: false,
        }
    }

    /// Advance the timer. May unblock a sleeping task.
    pub fn tick(&mut self, dt: Duration) {
        for task in self.tasks.iter_mut() {
            if let TaskStatus::Sleeping { time_left } = &mut task.status {
                *time_left = time_left.saturating_sub(dt);

                if time_left.is_zero() {
                    task.status = TaskStatus::Ready;
                }
            }
        }

        // Resume the topmost task if ready.
        if let Err(error) = self.resume() {
            let error = format!("{error:#?}");
            self.liveness = ActorLiveness::Killed { error };
        }
    }

    /// Fetch the next `IoReq`, if one is queued.
    /// The host is responsible for retaining the return value
    /// in order to pass it to `fulfill`.
    pub fn poll_io(&mut self) -> Option<IoReq> {
        self.outbox.pop_front()
    }

    /// Check whether an `IoReq` is still pending (true) or cancelled (false).
    pub fn pending(&self, req: &IoReq) -> bool {
        let Some(task) = self.tasks.get(req.task_index) else {
            return false;
        };

        task.status.secret() == Some(req.secret)
    }

    /// Unblock a pending IO request, and if possible, resume execution of the
    /// relevant task. Returns `Ok(true)` if this succeeds or `Ok(false)` if the
    /// request has expired. Kills the actor if an error is raised.
    pub fn fulfill(&mut self, req: IoReq, response: Value) -> Result<bool> {
        if !self.is_alive() {
            bail!("Actor is dead");
        }

        if !self.pending(&req) {
            return Ok(false);
        }

        let IoReq { task_index, .. } = req;

        let top = task_index + 1 >= self.tasks.len();

        let Some(task) = self.tasks.get_mut(task_index) else {
            return Ok(false);
        };

        match &task.status {
            &TaskStatus::AwaitMenu { secret, ref labels } => {
                if req.secret != secret {
                    return Ok(false);
                }

                let index = usize::try_from(response)?;

                let Some(label) = labels.get(index) else {
                    bail!("Menu item index {index} outside bounds of list {labels:?}");
                };

                task.pc = label.0;
                task.status = TaskStatus::Ready;
            },

            &TaskStatus::AwaitFfi { secret, ref dst, .. } => {
                if req.secret != secret {
                    return Ok(false);
                }

                let IoPayload::HostFfi { .. } = req.payload else {
                    return Ok(false)
                };

                self.heap.ctx_mut(&mut task.env).bind(dst.clone(), response)?;

                task.status = TaskStatus::Ready;
            },

            _ => (),
        }

        if top {
            if let Err(err) = self.resume() {
                let error = format!("{}", &err);
                self.liveness = ActorLiveness::Killed { error };
                return Err(err);
            }
        }

        Ok(true)
    }

    /// Deliver a message to this actor. Returns `Ok(true)` if it was handled,
    /// `Ok(false)` if not, and `Err` if delivery caused an error.
    /// (TODO: Maybe we don't even need Err)
    pub fn handle_message(&mut self, subject: &str) -> Result<bool> {
        if !self.is_alive() {
            bail!("Actor is dead");
        }

        // TODO: Actual payloads
        let payload = HashMap::<Arc<str>, Value>::default();

        let Some(handlers) = self.handlers.get_mut(subject) else {
            return Ok(false);
        };

        let mut env = self.heap.create();

        for (index, closure) in handlers.iter().enumerate().rev() {
            // Reset env for our next attempt
            self.heap.get_mut(env).reset(closure.env);

            // NOTE: Can't do disjoint borrows
            match self.heap.ctx_mut(&mut env).bind_pattern(&closure.pattern, payload.clone()) {
                Err(err) if self.trace => {
                    debug!("Pattern match failed: {err:#?}");
                    continue;
                },

                Ok(true) => {
                    self.tasks.push(Task {
                        pc: closure.label.0,
                        env,
                        status: TaskStatus::Ready,
                    });

                    // Borrow checker: OK to mutate here if we don't continue

                    if closure.cancel {
                        handlers.remove(index);
                    }

                    self.resume()?;

                    return Ok(true);
                },

                _ => continue,
            }
        }

        // Successful binding returns early from inside loop
        // If we get here, message has not been delivered

        self.heap.release(env);

        Ok(false)
    }

    /// Query the actor's current state.
    pub fn status(&self) -> ActorStatus {
        match &self.liveness {
            ActorLiveness::Killed { error } => {
                ActorStatus::Killed { error }
            },

            ActorLiveness::Loading { path, args } => {
                ActorStatus::Loading { path, args }
            },

            ActorLiveness::Retiring => {
                ActorStatus::Retiring
            },

            ActorLiveness::Running => {
                let Some(task) = &self.tasks.last() else {
                    return ActorStatus::Hibernating;
                };

                match &task.status {
                    TaskStatus::Ready => ActorStatus::Running,

                    &TaskStatus::Sleeping { time_left } => {
                        ActorStatus::Sleeping { time_left }
                    },

                    TaskStatus::AwaitFfi { .. } | TaskStatus::AwaitMenu { .. } => {
                        ActorStatus::Blocked
                    },
                }
            },
        }
    }

    /// Query whether the actor is "alive" or "dead" (eg. killed by an error).
    pub fn is_alive(&self) -> bool {
        match self.liveness {
            ActorLiveness::Running | ActorLiveness::Loading { .. } => true,
            ActorLiveness::Retiring | ActorLiveness::Killed { .. } => false,
        }
    }

    /// Destroy an Actor. If it died due to an error, return the error.
    pub fn reap(self) -> Option<String> {
        match self.liveness {
            ActorLiveness::Killed { error } => Some(error),
            _ => None,
        }
    }

    /// Unblock the topmost `Task` and resume execution.
    /// The caller **must** first check that the task is ready to resume.
    /// The caller **must** catch and handle any returned errors.
    fn resume(&mut self) -> Result<()> {
        // Needs to be on the stack to call methods on self
        // or manipulate the task stack
        // NOTE: Control flow is tricky here. Double check everything!
        let Some(mut task) = self.tasks.pop() else {
            // Hibernating
            return Ok(());
        };

        // Changed from `while let ...` to an unconditional loop.
        // This allows a blocked task to remain in the stack until
        // the host has replied with `fulfill()`
        loop {
            // Return by default; rely on correct handling below for cleanup
            let op = self.script.body.get(task.pc).unwrap_or(&Op::Return);

            if self.trace {
                debug!("Next instruction: {op:#?}");
            }

            let mut next_jump = 1;

            match op {
                Op::Eval { expr, dst } => {
                    // For now, assume all FnCalls are FFI
                    if let Expr::FnCall { lhs, args } = expr.as_ref() {
                        let Expr::Local { name } = lhs.as_ref() else {
                            bail!("Non-identifier used as function: {lhs:?}");
                        };

                        let args = args.iter().map(|expr| {
                            self.heap.ctx(task.env).eval(expr)
                        }).collect::<Result<Vec<Value>>>()?;

                        let dst = dst.clone();

                        let fn_name = name.clone();

                        let secret = self.secrets.next();

                        task.status = TaskStatus::AwaitFfi {
                            dst: dst.clone(),
                            secret,
                        };

                        self.outbox.push_back(IoReq {
                            task_index: self.tasks.len(),
                            secret,
                            payload: IoPayload::HostFfi {
                                fn_name,
                                args,
                            },
                        });

                        task.pc += next_jump;
                        break;
                    }

                    let value = self.heap.ctx(task.env).eval(expr)?;
                    self.heap.ctx_mut(&mut task.env).bind(dst.clone(), value)?;
                },

                &Op::EnterBlock { offset } => {
                    let parent = task.env;
                    let pc = task.pc;
                    task.pc += offset;

                    self.tasks.push(task);

                    task = Task {
                        pc, // Will be incremented by next_jump
                        env: self.heap.create_child(parent),
                        status: TaskStatus::Ready,
                    };
                },

                &Op::PushHandler { ref pattern, label, cancel } => {
                    let subject = pattern.subject.clone();
                    let pattern = pattern.clone();
                    let env = task.env;

                    self.heap.retain(env);

                    self.handlers.entry(subject).or_default().push(HandlerClosure {
                        pattern,
                        label,
                        cancel,
                        env,
                    });
                },

                &Op::CancelHandler { label } => {
                    for list in self.handlers.values_mut() {
                        list.retain(|h| {
                            if h.label != label {
                                return true;
                            }

                            self.heap.release(h.env);

                            false
                        });
                    }
                },

                &Op::Jz { ref guard, offset } => {
                    if !self.heap.eval(guard, task.env)?.is_truthy() {
                        next_jump = offset;
                    }
                },

                &Op::Jump { offset } => {
                    next_jump = offset;
                },

                Op::Menu { choices } => {
                    let mut items = Vec::with_capacity(choices.len());
                    let mut labels = Vec::with_capacity(choices.len());

                    for choice in choices.as_ref().iter() {
                        if let Some(guard) = &choice.guard {
                            let true = self.heap.eval(&guard, task.env)?.is_truthy() else {
                                continue;
                            };
                        }

                        items.push(self.heap.stitch(choice.prompt.as_ref(), task.env)?);
                        labels.push(choice.target);
                    }

                    if let &[label] = labels.as_slice() {
                        // Only one option, so just execute immediately
                        // Works the same as CallLabel
                        let parent = task.env;
                        self.tasks.push(task);
                        task = Task {
                            env: self.heap.create_child(parent),
                            pc: label.0,
                            status: TaskStatus::Ready,
                        };
                        continue;
                    } else if items.is_empty() {
                        // TODO: Other cleanup?
                        task.pc += next_jump;
                        continue;
                    }

                    // If there are multiple options, await player input
                    let secret = self.secrets.next();

                    self.outbox.push_back(IoReq {
                        secret,
                        task_index: self.tasks.len(),
                        payload: IoPayload::Menu {
                            items,
                        },
                    });

                    task.status = TaskStatus::AwaitMenu {
                        labels,
                        secret,
                    };

                    task.pc += next_jump;

                    // Task is pushed below when we break
                    break;
                },

                Op::Quote { speaker, lines } => {
                    let speaker = speaker.as_ref().map(|expr| -> Result<ActorRef> {
                        ActorRef::try_from(self.heap.eval(expr.as_ref(), task.env)?)
                    }).transpose()?;

                    let line = self.heap.stitch(lines.as_ref(), task.env)?;

                    let secret = self.secrets.next();

                    self.outbox.push_back(IoReq {
                        task_index: self.tasks.len(),
                        secret,
                        payload: IoPayload::Quote {
                            speaker,
                            line,
                        },
                    });

                    task.status = TaskStatus::AwaitFfi {
                        dst: Placement::Discard,
                        secret: 0,
                    };

                    task.pc += next_jump;
                    break;
                },

                &Op::Trace { on } => {
                    self.trace = on;
                },

                Op::Tailcall { path, args } => {
                    let path = path.clone();

                    let args = args.iter().map(|expr| {
                        self.heap.eval(expr, task.env)
                    }).collect::<Result<Vec<Value>>>()?;

                    self.liveness = ActorLiveness::Loading {
                        path,
                        args,
                    };

                    // Prevent all pending tasks from completing IO reqs
                    self.tasks.clear();

                    // Ignore all messages until new script is loaded
                    self.handlers.clear();

                    return Ok(());
                },

                Op::Wait { amount } => {
                    let time_left = self.heap.eval(&amount, task.env)?.try_into()?;
                    task.status = TaskStatus::Sleeping { time_left };
                    break;
                },

                Op::Return => {
                    self.heap.release(task.env);

                    // TODO: Any other cleanup?

                    if let Some(next) = self.tasks.pop() {
                        task = next;

                        let &TaskStatus::Ready = &task.status else {
                            // Call tasks.push() below, but stop execution
                            break;
                        };
                    } else {
                        // Time to hibernate
                        // Skip the call to tasks.push()
                        return Ok(());
                    }
                },

                Op::Retire => {
                    self.liveness = ActorLiveness::Retiring;
                    self.handlers.clear();
                    self.heap.clear();
                    self.tasks.clear();
                    return Ok(());
                },

                Op::Hcf { reason } => {
                    let reason = self.heap.stitch(reason, task.env)?;
                    // self.liveness to be set by caller
                    bail!("{reason}");
                },
            }

            task.pc += next_jump;
        }

        self.tasks.push(task);

        Ok(())
    }
}

impl TaskStatus {
    fn secret(&self) -> Option<usize> {
        match self {
            &TaskStatus::AwaitFfi { secret, .. } => Some(secret),
            &TaskStatus::AwaitMenu { secret, .. } => Some(secret),
            _ => None,
        }
    }
}

impl IoReq {
    pub fn payload(&self) -> &IoPayload {
        &self.payload
    }
}

impl Secrets {
    fn next(&mut self) -> usize {
        let secret = self.next;
        self.next += 1;
        secret
    }
}

#[cfg(test)]
mod dsl {
    use super::*;

    pub use crate::eval::dsl::*;

    pub fn eval(expr: impl Into<Arc<Expr>>) -> Op {
        let expr = expr.into();
        let dst = Placement::Discard;
        Op::Eval { expr, dst }
    }

    pub fn let_local(name: &str, expr: impl Into<Arc<Expr>>) -> Op {
        let dst = Placement::CreateLocal { name: name.into() };
        let expr = expr.into();
        Op::Eval { expr, dst }
    }

    pub fn mut_local(name: &str, expr: impl Into<Arc<Expr>>) -> Op {
        let dst = Placement::UpdateLocal { name: name.into() };
        let expr = expr.into();
        Op::Eval { expr, dst }
    }

    pub fn mut_global(name: &str, expr: impl Into<Arc<Expr>>) -> Op {
        let dst = Placement::Global { name: name.into() };
        let expr = expr.into();
        Op::Eval { expr, dst }
    }

    impl From<Vec<Op>> for Script {
        fn from(value: Vec<Op>) -> Self {
            let body = value.into();
            let pages = HashMap::new();
            Script { body, pages }
        }
    }
}

#[test]
fn two_plus_two() {
    use dsl::*;

    let globals = init_globals(&[
        ("ONE", Value::Int(1)),
        ("TWO", Value::Int(2)),
        ("BAR", Value::Int(0)),
    ]);

    let mut actor = Actor::from_script(Arc::new({
        vec![ 
            let_local("Four", add(global("TWO"), global("TWO"))),
            let_local("Bar", local("Four")),
            mut_local("Bar", sub(local("Four"), 2)),
            eval(local("Bar")),
            mut_global("BAR", local("Bar")),
            Op::Return,
        ].into()
    }), globals);

    actor.resume().unwrap();

    let ActorStatus::Hibernating = actor.status() else {
        let status = actor.status();
        panic!("Unexpected actor status: {status:?}");
    };

    let heap = actor.heap;

    let two = heap.lookup_global("BAR").unwrap();

    assert_eq!(two, Value::Int(2));
}

#[test]
fn eval_stitch() {
    use dsl::*;

    let script = Arc::new(vec![
        eval(Expr::FnCall {
            lhs: local("print").into(),
            args: vec![
                Expr::Infix {
                    lhs: Expr::String { value: "Car".into() }.into(),
                    op: crate::ast::Binop::Stitch,
                    rhs: Expr::String { value: "pet".into() }.into(),
                },
            ],
        }),
    ].into());
    
    let globals = init_globals(&[]);
    let mut actor = Actor::from_script(script, globals);

    actor.resume().unwrap();

    let req = actor.poll_io().unwrap();

    let IoPayload::HostFfi { fn_name, args } = req.payload() else {
        panic!("Expected FFI call")
    };

    assert_eq!(fn_name.as_ref(), "print");
    assert_eq!(args, &[Value::String("Carpet".into())]);
}

#[test]
fn ffi_meet_and_greet() {
    use dsl::*;

    let globals = init_globals(&[]);

    let mut actor = Actor::from_script(Arc::new(vec![
        let_local("Neighbor", Expr::FnCall {
            lhs: local("meet").into(),
            args: vec![],
        }),
        eval(Expr::FnCall {
            lhs: local("greet").into(),
            args: vec![local("Neighbor")],
        }),
    ].into()), globals);

    actor.resume().unwrap();

    let req = actor.poll_io().unwrap();
    assert!(actor.poll_io().is_none());

    assert!(actor.fulfill(req, Value::String("world".into())).unwrap());

    let req2 = actor.poll_io().unwrap();
    assert!(actor.poll_io().is_none());

    let IoPayload::HostFfi { fn_name, args } = req2.payload() else {
        panic!("Wrong payload: {:#?}", req2.payload());
    };

    assert_eq!(fn_name.as_ref(), "greet");
    assert_eq!(args, &[Value::String("world".into())]);

    assert!(actor.fulfill(req2, Value::Int(0)).unwrap());

    assert!(actor.tasks.is_empty());
}

#[test]
fn menu_single() {
    use dsl::*;

    let script: Script = vec![
        Op::Menu {
            choices: vec![
                MenuItem {
                    prompt: vec![
                        "Only one option".into(),
                    ].into(),
                    guard: None,
                    target: TaskLabel(2),
                },
            ].into(),
        },

        Op::Hcf { reason: vec!["BAM!".into()].into() },

        Op::Retire,
    ].into();

    let globals = init_globals(&[]);

    let mut actor = Actor::from_script(script.into(), globals);

    actor.resume().unwrap();

    let ActorStatus::Retiring = actor.status() else {
        eprintln!("Wrong actor status: {:?}", actor.status());

        panic!("{actor:#?}");
    };
}

#[test]
fn menu_multiple() {
    use dsl::*;

    let item = |index: usize, prompt: &str| MenuItem {
        prompt: vec![
            prompt.into(),
        ].into(),
        guard: None,
        target: TaskLabel(index),
    };

    let die = |message: &str| Op::Hcf {
        reason: vec![
            message.into(),
        ].into(),
    };

    let script: Script = vec![
        Op::Menu {
            choices: vec![
                item(1, "Genuflect"),
                item(2, "Put it in"),
                item(3, "Hold hands"),
            ].into(),
        },

        die("Died at 1"),
        die("Died at 2"),

        eval(Expr::FnCall {
            lhs: local("hold").into(),
            args: vec![
                Expr::String { value: "hands".into() },
            ],
        }),
    ].into();

    let globals = init_globals(&[]);

    let mut actor = Actor::from_script(Arc::new(script), globals);

    actor.resume().unwrap();

    let menu_req = actor.poll_io().unwrap();

    actor.fulfill(menu_req, Value::Int(2)).unwrap();

    let hold_req = actor.poll_io().unwrap();

    let IoPayload::HostFfi { fn_name, args } = hold_req.payload() else {
        panic!("Wrong IO payload in {hold_req:#?}");
    };

    assert_eq!(fn_name.as_ref(), "hold");
    assert_eq!(args.as_slice(), &[Value::String("hands".into())]);
}
