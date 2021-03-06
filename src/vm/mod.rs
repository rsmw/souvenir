mod pretty_print;

use std::collections::{HashMap, VecDeque};

use string_interner::{StringInterner, NonNegative};

use vecmap::*;

/// Entry point to the interpreter API.
pub struct Scheduler {
    program: Program,

    /// Processes which are alive and ready to run immediately.
    queue: RunQueue,

    /// Buffer of processes presently being executed.
    workspace: VecDeque<Task>,

    /// Buffered output from execution.
    outbuf: VecDeque<OutSignal>,

    env_table: VecMap<EnvId, Value>,

    global_heap: Heap,

    next_pid: u32,

    next_event: u32,
}

/// Organizes processes by current status.
struct RunQueue {
    running: HashMap<ActorId, Box<Process>>,
    sleeping: HashMap<ActorId, (Tag, Box<Process>)>,
    dead: VecDeque<Box<Process>>,
}

/// Program data marshalled for use by the host environment.
#[derive(Clone, Debug)]
pub enum RawValue {
    ActorId(ActorId),
    Atom(String),
    Int(i32),
    Str(String),
    List(Vec<RawValue>),
}

/// Signals sent into the interpreter by the host environment. Cannot be cloned.
pub enum InSignal {
    Kill(ActorId),
    EndSay(SayReplyToken),
    EndAsk(AskReplyToken),
}

/// Signals sent from the interpreter to the host environment. Cannot be cloned.
pub enum OutSignal {
    Exit(ActorId),
    Hcf(ActorId, RunErr),
    Say(SayToken),
    Ask(AskToken),
    Trace(ActorId, RawValue),
}

/// Opaque key into the supervisor's list of processes.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct ActorId(u32);

struct Task {
    id: ActorId,
    process: Box<Process>,
    status: Ret<RunState>,
}

#[derive(Debug, Eq, Hash, PartialEq)]
struct Tag(ActorId, u32);

// NB. No Copy, no Clone!
pub struct SayToken(Tag, RawValue);
pub struct SayReplyToken(Tag);
pub struct AskToken(Tag, Vec<(i32, RawValue)>, Reg);
pub struct AskReplyToken(Tag, i32, Reg);

/// Executable program
#[derive(Clone, Debug)]
pub struct Program {
    /// Instructions from all blocks.
    pub code: VecMap<InstrAddr, Instr>,

    /// Lookup table for the destinations of jump instructions.
    pub jump_table: VecMap<Label, InstrAddr>,

    /// Interned atoms.
    pub atom_table: StringInterner<AtomId>,

    /// Interned (global) string constants.
    pub str_table: StringInterner<StrId>,

    /// Sparse map of env IDs for labels that need them.
    pub env_table: EnvTable,

    /// Map of (qualified) scene names to their corresponding entry points.
    pub scene_table: SceneTable,
}

/// Unencoded (immediately executable) VM instructions.
///
/// Multi-argument operations follow the convention of `input -> output` in
/// their arguments. So, for example, `Add(a, b)` reads a value from `a` and
/// adds it to `b`.
#[derive(Copy, Clone, Debug)]
pub enum Instr {
    Cpy(Reg, Reg),
    Add(Reg, Reg),
    Sub(Reg, Reg),
    Div(Reg, Reg),
    Mul(Reg, Reg),
    Eql(Reg, Reg, Flag),
    Gte(Reg, Reg, Flag),
    Lte(Reg, Reg, Flag),
    Gt(Reg, Reg, Flag),
    Lt(Reg, Reg, Flag),
    And(Flag, Flag),
    Or(Flag, Flag),
    Set(Flag, Flag),
    Not(Flag),
    True(Flag),
    False(Flag),
    Reify(Flag, Reg),
    Nonzero(Reg, Flag),
    CheckSize(ListLen, Reg, Flag),
    LoadLit(Value, Reg),
    Alloc(ListLen, Reg),
    Read(Ptr, Reg),
    Write(Reg, Ptr),
    Jump(Label),
    JumpIf(Flag, Label),
    Arm(Reg, Label),
    Disarm(Label),
    Return(bool),
    Blocking(Io),
    Nop,
    Bye,
    Hcf,
}

/// Instructions representing blocking IO operations.
#[derive(Copy, Clone, Debug)]
pub enum Io {
    Export(Reg, EnvId),
    Recur(Reg, Label),
    Spawn(Reg, Label, Reg),
    GetPid(Reg),
    SendMsg(Reg, Reg),
    Roll(Reg, Reg),
    Sleep(f32),
    ArmAtomic(Reg, Label),
    Trace(Reg),
    Native(Reg, NativeFn, Reg),
    Say(Reg),
    Ask(Reg, Reg),
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Value {
    Int(i32),
    Atom(AtomId),
    ActorId(ActorId),
    StrConst(StrId),
    StrAddr(u32),
    ListAddr(HeapAddr),
    Capacity(u32),
    Undefined,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TypeTag {
    Int,
    Atom,
    Actor,
    Str,
    List,
}

#[derive(Copy, Clone, Debug)]
pub struct NativeFn(u32);

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ListLen(pub u32);

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Ptr {
    pub addr: Reg,
    pub offset: u32,
}

pub struct LocalValue<'a> {
    value: Value,
    heap: &'a Heap,
}

#[derive(Clone, Debug)]
pub struct SceneDef {
    pub label: Label,
    pub argc: u32,
}

pub type JumpTable = VecMap<Label, InstrAddr>;

pub type EnvTable = HashMap<Label, EnvId>;

pub type SceneTable = HashMap<String, SceneDef>;

pub struct StackFrame {
    gpr: [Value; REG_COUNT],
    flag: [bool; REG_COUNT],
}

/// Prototype for a message handler.
#[derive(Copy, Clone, Debug)]
pub struct Trap {
    label: Label,
    env: HeapAddr,
}

/// State of a handler invocation.
pub struct Continuation {
    /// Code position to return to once there are no handlers left to execute.
    return_addr: InstrAddr,

    /// Input to running handlers.
    argv: HeapAddr,

    frame: StackFrame,

    /// Sequence of remaining handlers in this invocation.
    queue: Vec<Trap>,
}

pub struct Stack {
    lower: StackFrame,
    upper: Option<Continuation>,
}

#[derive(Clone, Debug, Default)]
pub struct Heap {
    values: Vec<Value>,
    strings: Vec<String>,
}

pub struct Process {
    stack: Stack,
    heap: Heap,
    traps: Vec<Trap>,
    inbox: VecDeque<HeapAddr>,
    op: Instr,
    pc: InstrAddr,
}

#[derive(Copy, Clone, Debug)]
enum RunState {
    Blocked(Io),
    Running,
    Exiting,
}

#[derive(Copy, Clone, Debug)]
pub enum RunErr {
    StackOverflow,
    StackUnderflow,
    NoSuchRegister(Reg),
    NoSuchFlag(Flag),
    NoSuchLabel(Label),
    NoSuchScene(Label),
    FetchOutOfBounds(InstrAddr),
    IllegalInstr(Instr),
    UnallocatedAccess(usize),
    HeapCorrupted(Value),
    ListOutOfBounds(usize, u32),
    TypeMismatch(Value, TypeTag),
    DividedByZero,
    Unrepresentable(usize),
    Uninitialized,
    UnrecognizedAtom,
    UnrecognizedSceneName,
    NoSuchAtom(AtomId),
    NoSuchValue(Value),
    EnvNotInitialized(EnvId),
    EnvExportMismatch { expected: EnvId, found: EnvId, },
    ArgCountMismatch { expected: usize, found: usize, },
    InitFailure,
}

pub type Ret<T> = Result<T, RunErr>;

pub const REG_COUNT: usize = 0x400;

impl Default for Instr {
    fn default() -> Self { Instr::Nop }
}

impl Default for StackFrame {
    fn default() -> Self {
        StackFrame {
            gpr: [Value::Undefined; REG_COUNT],
            flag: [false; REG_COUNT],
        }
    }
}

macro_rules! index_via_u32 {
    ( $name:ident, $( $value:ty ),* ) => {
        #[derive(Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
        pub struct $name(pub u32);

        impl From<$name> for usize {
            fn from($name(u): $name) -> Self {
                u as usize
            }
        }

        impl CheckedFrom<usize> for $name {
            fn checked_from(u: usize) -> Option<Self> {
                if u > u32::max_value() as usize {
                    None
                } else {
                    Some($name(u as u32))
                }
            }
        }

        $( impl IndexFor<$value> for $name {} )*
    };
}

index_via_u32!(Label, InstrAddr, EnvId);
index_via_u32!(InstrAddr, Instr);
index_via_u32!(Reg, Value);
index_via_u32!(HeapAddr, Value);
index_via_u32!(EnvId, Value);
index_via_u32!(Flag, bool);

macro_rules! symbol_via_u32 {
    ( $name:ident ) => {
        #[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(pub u32);

        impl From<$name> for usize {
            fn from($name(u): $name) -> Self {
                u as usize
            }
        }

        impl From<usize> for $name {
            fn from(u: usize) -> Self {
                $name(u as u32)
            }
        }

        impl NonNegative for $name { }

        // Blanket implementation automatically satisfied
        //impl Symbol for $name { }
    };
}

symbol_via_u32!(AtomId);
symbol_via_u32!(StrId);

impl Stack {
    fn current(&mut self) -> &mut StackFrame {
        if let Some(c) = self.upper.as_mut() {
            return &mut c.frame;
        }

        &mut self.lower
    }

    fn push(&mut self, cc: Continuation) -> Ret<()> {
        if self.upper.is_some() {
            Err(RunErr::StackOverflow)
        } else {
            self.upper = Some(cc);
            Ok(())
        }
    }

    fn pop(&mut self) -> Ret<Continuation> {
        self.upper.take().ok_or(RunErr::StackUnderflow)
    }
}

impl StackFrame {
    fn get(&self, r: Reg) -> Ret<Value> {
        let i = r.0 as usize;
        if i < REG_COUNT {
            Ok(self.gpr[i])
        } else {
            Err(RunErr::NoSuchRegister(r))
        }
    }

    fn set(&mut self, r: Reg, v: Value) -> Ret<()> {
        let i = r.0 as usize;
        if i < REG_COUNT {
            self.gpr[i] = v;
            Ok(())
        } else {
            Err(RunErr::NoSuchRegister(r))
        }
    }

    fn get_flag(&mut self, f: Flag) -> Ret<bool> {
        let i = f.0 as usize;
        if i < REG_COUNT {
            Ok(self.flag[i])
        } else {
            Err(RunErr::NoSuchFlag(f))
        }
    }

    fn set_flag(&mut self, f: Flag, v: bool) -> Ret<()> {
        let i = f.0 as usize;
        if i < REG_COUNT {
            self.flag[i] = v;
            Ok(())
        } else {
            Err(RunErr::NoSuchFlag(f))
        }
    }
}

impl Heap {
    fn alloc(&mut self, len: ListLen) -> Ret<HeapAddr> {
        let addr = HeapAddr(self.values.len() as u32);
        self.values.push(len.into());
        for _ in 0 .. len.0 {
            self.values.push(Value::Undefined);
        }
        Ok(addr)
    }

    fn size_of(&self, addr: HeapAddr) -> Ret<u32> {
        let addr: usize = addr.into();
        let header = *self.values.get(addr)
            .ok_or(RunErr::UnallocatedAccess(addr))?;

        if let Value::Capacity(size) = header {
            Ok(size)
        } else {
            Err(RunErr::HeapCorrupted(header))
        }
    }

    fn check_bounds(&self, addr: HeapAddr, offset: u32) -> Ret<usize> {
        if self.size_of(addr)? > offset {
            Ok(usize::from(addr) + 1 + offset as usize)
        } else {
            Err(RunErr::ListOutOfBounds(usize::from(addr), offset))
        }
    }

    fn get(&self, addr: HeapAddr, offset: u32) -> Ret<Value> {
        let i = self.check_bounds(addr, offset)?;
        Ok(self.values[i])
    }

    fn set(&mut self, addr: HeapAddr, offset: u32, value: Value) -> Ret<()> {
        let i = self.check_bounds(addr, offset)?;
        self.values[i] = value;
        Ok(())
    }

    fn clear(&mut self) {
        self.values.clear();
        self.strings.clear();
    }

    fn localize(&mut self, item: LocalValue) -> Ret<Value> {
        Ok(match item.value {
            Value::StrAddr(addr) => {
                // FIXME: We should be using a StringInterner here
                let len = self.strings.len();
                let content = item.heap.strings.get(addr as usize)
                    .ok_or(RunErr::UnallocatedAccess(addr as usize))?;
                self.strings.push(content.to_owned());
                Value::StrAddr(len as u32)
            },

            Value::ListAddr(addr) => {
                let len = item.heap.size_of(addr)?;
                let list = self.alloc(ListLen(len))?;
                for i in 0 .. len {
                    let value = self.localize(LocalValue {
                        value: item.heap.get(addr, i)?,
                        heap: item.heap,
                    })?;
                    self.set(list, i, value)?;
                }
                Value::ListAddr(list)
            },

            other => other,
        })
    }
}

impl Process {
    pub fn exec(&mut self, program: &Program) -> Ret<()> {
        match self.op {
            Instr::Nop => (),

            Instr::Cpy(src, dst) => {
                let frame = self.stack.current();
                let value = frame.get(src)?;
                frame.set(dst, value)?;
            },

            Instr::Add(src, dst) => {
                let frame = self.stack.current();
                let lhs = frame.get(dst)?.as_int()?;
                let rhs = frame.get(src)?.as_int()?;
                frame.set(dst, (lhs + rhs).into())?;
            },

            Instr::Sub(src, dst) => {
                let frame = self.stack.current();
                let lhs = frame.get(dst)?.as_int()?;
                let rhs = frame.get(src)?.as_int()?;
                frame.set(dst, (lhs - rhs).into())?;
            },

            Instr::Div(src, dst) => {
                let frame = self.stack.current();
                let lhs = frame.get(dst)?.as_int()?;
                let rhs = frame.get(src)?.as_int()?;
                if rhs == 0 {
                    return Err(RunErr::DividedByZero);
                } else {
                    frame.set(dst, (lhs / rhs).into())?;
                }
            },

            Instr::Mul(src, dst) => {
                let frame = self.stack.current();
                let lhs = frame.get(dst)?.as_int()?;
                let rhs = frame.get(src)?.as_int()?;
                frame.set(dst, (lhs * rhs).into())?;
            },

            Instr::Eql(lhs, rhs, flag) => {
                let frame = self.stack.current();
                let lhs = frame.get(lhs)?;
                let rhs = frame.get(rhs)?;
                frame.set_flag(flag, lhs == rhs)?;
            },

            Instr::Gte(lhs, rhs, flag) => {
                let frame = self.stack.current();
                let lhs = frame.get(lhs)?.as_int()?;
                let rhs = frame.get(rhs)?.as_int()?;
                frame.set_flag(flag, lhs >= rhs)?;
            },

            Instr::Lte(lhs, rhs, flag) => {
                let frame = self.stack.current();
                let lhs = frame.get(lhs)?.as_int()?;
                let rhs = frame.get(rhs)?.as_int()?;
                frame.set_flag(flag, lhs <= rhs)?;
            },

            Instr::Gt(lhs, rhs, flag) => {
                let frame = self.stack.current();
                let lhs = frame.get(lhs)?.as_int()?;
                let rhs = frame.get(rhs)?.as_int()?;
                frame.set_flag(flag, lhs > rhs)?;
            },

            Instr::Lt(lhs, rhs, flag) => {
                let frame = self.stack.current();
                let lhs = frame.get(lhs)?.as_int()?;
                let rhs = frame.get(rhs)?.as_int()?;
                frame.set_flag(flag, lhs < rhs)?;
            },

            Instr::And(src, dst) => {
                let frame = self.stack.current();
                let lhs = frame.get_flag(dst)?;
                let rhs = frame.get_flag(src)?;
                frame.set_flag(dst, lhs && rhs)?;
            },

            Instr::Or(src, dst) => {
                let frame = self.stack.current();
                let lhs = frame.get_flag(dst)?;
                let rhs = frame.get_flag(src)?;
                frame.set_flag(dst, lhs || rhs)?;
            },

            Instr::Not(flag) => {
                let frame = self.stack.current();
                let value = frame.get_flag(flag)?;
                frame.set_flag(flag, !value)?;
            },

            Instr::True(flag) => {
                self.stack.current().set_flag(flag, true)?;
            },

            Instr::False(flag) => {
                self.stack.current().set_flag(flag, false)?;
            },

            Instr::Reify(flag, dst) => {
                let test = self.stack.current().get_flag(flag)?;
                self.stack.current().set(dst, test.into())?;
            },

            Instr::Nonzero(src, flag) => {
                let value = self.stack.current().get(src)?;
                self.stack.current().set_flag(flag, value.as_bool()?)?;
            },

            Instr::Set(src, dst) => {
                let value = self.stack.current().get_flag(src)?;
                self.stack.current().set_flag(dst, value)?;
            },

            Instr::CheckSize(ListLen(expected), reg, flag) => {
                let result = match self.stack.current().get(reg)? {
                    Value::ListAddr(addr) => {
                        let found = self.heap.size_of(addr)?;
                        expected == found
                    },

                    _ => false,
                };

                self.stack.current().set_flag(flag, result)?;
            },

            Instr::LoadLit(value, dst) => {
                self.stack.current().set(dst, value)?;
            },

            Instr::Alloc(len, dst) => {
                let addr = self.heap.alloc(len)?;
                self.stack.current().set(dst, addr.into())?;
            },

            Instr::Read(ptr, dst) => {
                let frame = self.stack.current();
                let addr = frame.get(ptr.addr)?.as_addr()?;
                let value = self.heap.get(addr, ptr.offset)?;
                frame.set(dst, value)?;
            },

            Instr::Write(src, ptr) => {
                let frame = self.stack.current();
                let value = frame.get(src)?;
                let addr = frame.get(ptr.addr)?.as_addr()?;
                self.heap.set(addr, ptr.offset, value)?;
            },

            Instr::Jump(label) => {
                self.pc = *program.jump_table.get(label)?;
            },

            Instr::JumpIf(flag, label) => {
                if self.stack.current().get_flag(flag)? {
                    self.pc = *program.jump_table.get(label)?;
                }
            },

            Instr::Return(finished) => {
                let cc = self.stack.pop()?;
                self.pc = cc.return_addr;

                if !finished {
                    self.call(cc, program)?;
                }

                // If self.call() proceeded with the next trap in the queue,
                // this call will return immediately because there will be an
                // upper stack frame. If self.call() ran out of traps, there
                // may still be messages in the queue -- but if not, it still
                // does nothing and we'll continue with normal execution.
                self.check_inbox(program)?;
            },

            Instr::Arm(env, label) => {
                self.arm(env, label)?;
            },

            Instr::Disarm(label) => {
                self.traps.retain(|trap| trap.label != label);
            },

            Instr::Blocking(_) | Instr::Bye | Instr::Hcf => {
                return Err(RunErr::IllegalInstr(self.op))
            },
        }

        Ok(())
    }

    fn arm(&mut self, env: Reg, label: Label) -> Ret<()> {
        self.traps.retain(|trap| trap.label != label);
        let addr = self.stack.current().get(env)?.as_addr()?;
        self.traps.push(Trap {
            env: addr,
            label: label,
        });
        Ok(())
    }

    fn check_inbox(&mut self, program: &Program) -> Ret<()> {
        if self.stack.upper.is_some() {
            return Ok(());
        }

        if self.traps.is_empty() {
            self.inbox.clear();
            return Ok(());
        }

        if let Some(argv) = self.inbox.pop_front() {
            let cc = Continuation {
                return_addr: self.pc,
                queue: self.traps.clone(),
                frame: StackFrame::default(),
                argv: argv,
            };

            self.call(cc, program)?;
        }

        Ok(())
    }

    fn call(&mut self, mut cc: Continuation, program: &Program) -> Ret<()> {
        let trap = match cc.queue.pop() {
            Some(trap) => trap,
            None => return Ok(()),
        };

        cc.frame.set(Reg::env(), trap.env.into())?;
        cc.frame.set(Reg::arg(), cc.argv.into())?;

        self.stack.push(cc)?;
        self.pc = *program.jump_table.get(trap.label)?;

        Ok(())
    }

    pub fn fetch(&mut self, program: &Program) -> Ret<()> {
        self.op = *program.code.get(self.pc)?;
        self.pc.0 += 1;
        Ok(())
    }

    fn run_state(&self) -> Ret<RunState> {
        Ok(match self.op {
            Instr::Bye => RunState::Exiting,
            Instr::Blocking(io) => RunState::Blocked(io),
            Instr::Hcf => return Err(RunErr::IllegalInstr(self.op)),
            _ => RunState::Running,
        })
    }

    fn start(&mut self, argv: LocalValue, env: LocalValue, label: Label, program: &Program) -> Ret<()> {
        let argv = self.heap.localize(argv)?;
        self.stack.lower.set(Reg::arg(), argv)?;

        let env = self.heap.localize(env)?;
        self.stack.lower.set(Reg::env(), env)?;

        self.pc = *program.jump_table.get(label)?;
        self.fetch(program)?;

        Ok(())
    }

    fn run(&mut self, program: &Program) -> Ret<RunState> {
        const SOME_SMALL_NUMBER: usize = 100;

        // FIXME: When do we check the inbox?

        for _ in 0 .. SOME_SMALL_NUMBER {
            match self.run_state()? {
                RunState::Running => (),
                other => return Ok(other),
            };

            self.exec(program)?;

            self.fetch(program)?;
        }

        self.run_state()
    }
}

impl Program {
    pub fn init(self) -> Ret<Scheduler> {
        let mut scheduler = Scheduler {
            program: self,
            workspace: VecDeque::with_capacity(32),
            queue: RunQueue {
                running: HashMap::new(),
                sleeping: HashMap::new(),
                dead: VecDeque::with_capacity(32),
            },
            global_heap: Heap::default(),
            env_table: VecMap::with_capacity(32),
            outbuf: VecDeque::with_capacity(32),
            next_event: 0,
            next_pid: 0,
        };

        scheduler.build_env()?;

        Ok(scheduler)
    }
}

impl Scheduler {
    pub fn spawn(&mut self, name: &str, args: Vec<RawValue>) -> Ret<ActorId> {
        let SceneDef { label, argc } = self.program.scene_table.get(name)
            .cloned().ok_or(RunErr::UnrecognizedSceneName)?;

        let argc = argc as usize;

        if argc != args.len() {
            return Err(RunErr::ArgCountMismatch {
                expected: argc,
                found: args.len(),
            });
        }

        let mut task = self.create();

        // FIXME: Can't use Process::start() here

        let args = RawValue::List(args);
        let argv = self.unmarshal(args, &mut task.process.heap)?;
        task.process.stack.lower.set(Reg::arg(), argv)?;

        let env = task.process.heap.localize({
            let &env_id = self.program.env_table.get(&label)
                .ok_or(RunErr::NoSuchLabel(label))?;

            self.env_table.get(env_id)?.in_heap(&self.global_heap)
        })?;
        task.process.stack.lower.set(Reg::env(), env)?;

        task.process.pc = *self.program.jump_table.get(label)?;
        task.process.fetch(&self.program)?;

        let id = task.id;
        self.queue.running.insert(task.id, task.process);

        Ok(id)
    }

    pub fn write(&mut self, signal: InSignal) {
        match signal {
            InSignal::EndSay(SayReplyToken(ticket)) => {
                if let Some((id, mut process)) = self.wakeup(ticket) {
                    // FIXME: Reconsider fetch schedule so we don't have to
                    // ignore this error
                    let _ = process.fetch(&self.program);
                    self.queue.running.insert(id, process);
                }
            },

            InSignal::EndAsk(AskReplyToken(ticket, index, dst)) => {
                if let Some((id, mut process)) = self.wakeup(ticket) {
                    let _ = process.fetch(&self.program);
                    process.stack.current().set(dst, Value::Int(index))
                        .expect("Ask reply wrote to a bad register");
                    self.queue.running.insert(id, process);
                }
            },

            _ => unimplemented!(),
        }
    }

    pub fn read(&mut self) -> Option<OutSignal> {
        self.outbuf.pop_front()
    }

    pub fn dispatch(&mut self) {
        // FIXME: This isn't a very good scheduler.

        self.workspace.extend(self.queue.running.drain().map(|(id, p)| {
            Task {
                id: id,
                process: p,
                status: Ok(RunState::Running),
            }
        }));

        for task in self.workspace.iter_mut() {
            task.status = task.process.run(&self.program);
        }

        while let Some(mut task) = self.workspace.pop_front() {
            if let Ok(RunState::Exiting) = task.status {
                self.outbuf.push_back(OutSignal::Exit(task.id));
                self.queue.dead.push_back(task.process);
                continue;
            }

            match self.unblock(&mut task) {
                Ok(Some(tag)) => {
                    self.queue.sleeping.insert(task.id, (tag, task.process));
                },

                Ok(None) => {
                    self.queue.running.insert(task.id, task.process);
                },

                Err(err) => {
                    self.outbuf.push_back(OutSignal::Hcf(task.id, err));
                    self.queue.dead.push_back(task.process);
                },
            }
        }
    }

    fn build_env(&mut self) -> Ret<()> {
        let mut init = Box::new(Process::default());

        loop {
            let io = match init.run(&self.program)? {
                RunState::Exiting => break,
                RunState::Running => continue,
                RunState::Blocked(io) => io,
            };

            match io {
                Io::Export(reg, env_id) => {
                    let env = self.global_heap.localize({
                        init.stack.current().get(reg)?
                            .in_heap(&init.heap)
                    })?;

                    let id = self.env_table.push(env)?;

                    if id != env_id {
                        return Err(RunErr::EnvExportMismatch {
                            expected: id,
                            found: env_id,
                        });
                    }
                },

                _ => {
                    return Err(RunErr::IllegalInstr(init.op));
                },
            }

            init.fetch(&self.program)?;
        }

        self.queue.dead.push_back(init);

        Ok(())
    }

    fn unblock(&mut self, task: &mut Task) -> Ret<Option<Tag>> {
        let &mut Task { id, ref mut process, status } = task;

        let io = match status? {
            RunState::Blocked(io) => io,
            RunState::Running => return Ok(None),
            _ => return Err(RunErr::IllegalInstr(process.op)),
        };

        match io {
            Io::Export(_, _) => {
                return Err(RunErr::IllegalInstr(process.op))
            },

            Io::GetPid(dst) => {
                let pid = Value::ActorId(id);
                process.stack.current().set(dst, pid)?;
                process.fetch(&self.program)?;
                Ok(None)
            },

            Io::Say(msg) => {
                let value = process.stack.current().get(msg)?;
                let content = self.marshal(value.in_heap(&process.heap))?;
                let tag = self.tag(id);
                let token = SayToken(tag.private_clone(), content);
                self.outbuf.push_back(token.into());
                Ok(Some(tag))
            },

            Io::Ask(src, dst) => {
                let value = process.stack.current().get(src)?;
                let choices = self.get_menu(value.in_heap(&process.heap))?;
                let tag = self.tag(id);

                let token = AskToken(tag.private_clone(), choices, dst);
                self.outbuf.push_back(token.into());
                Ok(Some(tag))
            },

            Io::ArmAtomic(env, label) => {
                process.arm(env, label)?;
                let tag = self.tag(id);
                Ok(Some(tag))
            },

            Io::Native(_, _, _) => {
                unimplemented!()
            },

            Io::Roll(_, _) => {
                unimplemented!()
            },

            Io::SendMsg(_, _) => {
                unimplemented!()
            },

            Io::Sleep(_time) => {
                unimplemented!()
            },

            Io::Spawn(argv, label, dst) => {
                let mut new = self.create();

                {
                    let argv = process.stack.current().get(argv)?
                        .in_heap(&process.heap);
                    let &env_id = self.program.env_table.get(&label)
                        .ok_or(RunErr::NoSuchScene(label))?;
                    let env = self.env_table.get(env_id)?
                        .in_heap(&self.global_heap);

                    new.process.start(argv, env, label, &self.program)?;
                }

                self.queue.running.insert(new.id, new.process);

                process.stack.current().set(dst, new.id.into())?;
                process.fetch(&self.program)?;

                Ok(None)
            },

            Io::Recur(argv, label) => {
                // Same as Spawn, but we replace the current process
                let mut new = self.create();

                {
                    let argv = process.stack.current().get(argv)?
                        .in_heap(&process.heap);
                    let &env_id = self.program.env_table.get(&label)
                        .ok_or(RunErr::NoSuchScene(label))?;
                    let env = self.env_table.get(env_id)?
                        .in_heap(&self.global_heap);

                    new.process.start(argv, env, label, &self.program)?;
                }

                ::std::mem::swap(process, &mut new.process);

                self.queue.dead.push_back(new.process);

                Ok(None)
            },

            Io::Trace(reg) => {
                let value = process.stack.current().get(reg)?;
                let message = self.marshal(value.in_heap(&process.heap))?;
                self.outbuf.push_back(OutSignal::Trace(id, message));
                process.fetch(&self.program)?;
                Ok(None)
            },
        }
    }

    fn create(&mut self) -> Task {
        let mut process = self.queue.fetch();
        let new_id = ActorId(self.next_pid);
        self.next_pid += 1;
        process.stack = Stack::default();
        process.heap.clear();
        process.traps.clear();

        Task {
            id: new_id,
            process: process,
            status: Ok(RunState::Running),
        }
    }

    fn tag(&mut self, id: ActorId) -> Tag {
        let tag = Tag(id, self.next_event);
        self.next_event += 1;
        tag
    }

    fn wakeup(&mut self, ticket: Tag) -> Option<(ActorId, Box<Process>)> {
        let id = ticket.0;

        match self.queue.sleeping.remove(&id) {
            Some((waiting_for, process)) => {
                if ticket == waiting_for {
                    Some((id, process))
                } else {
                    self.queue.sleeping.insert(id, (waiting_for, process));
                    None
                }
            },

            None => None,
        }
    }

    fn marshal(&self, item: LocalValue) -> Ret<RawValue> {
        match item.value {
            Value::Int(i) => Ok(RawValue::Int(i)),
            Value::ActorId(id) => Ok(RawValue::ActorId(id)),

            Value::Atom(id) => {
                match self.program.atom_table.resolve(id) {
                    Some(s) => Ok(RawValue::Atom(s.to_owned())),
                    None => Err(RunErr::NoSuchAtom(id)),
                }
            },

            Value::StrAddr(addr) => {
                match item.heap.strings.get(addr as usize) {
                    Some(s) => Ok(RawValue::Str(s.clone())),
                    None => Err(RunErr::UnallocatedAccess(addr as usize)),
                }
            },

            Value::StrConst(id) => {
                match self.program.str_table.resolve(id) {
                    Some(s) => Ok(RawValue::Str(s.to_owned())),
                    None => Err(RunErr::NoSuchValue(item.value)),
                }
            },

            Value::ListAddr(addr) => {
                let len = item.heap.size_of(addr)?;
                let mut list = Vec::with_capacity(len as usize);
                for i in 0 .. len {
                    let value = item.heap.get(addr, i)?;
                    list.push(self.marshal(value.in_heap(item.heap))?);
                }
                Ok(RawValue::List(list))
            },

            Value::Capacity(_) => Err(RunErr::HeapCorrupted(item.value)),
            Value::Undefined => Err(RunErr::Uninitialized),
        }
    }

    fn unmarshal(&self, item: RawValue, heap: &mut Heap) -> Ret<Value> {
        match item {
            RawValue::ActorId(a) => Ok(Value::ActorId(a)),

            RawValue::Int(i) => Ok(Value::Int(i)),

            RawValue::Atom(name) => {
                if let Some(id) = self.program.atom_table.get(name) {
                    Ok(Value::Atom(id))
                } else {
                    Err(RunErr::UnrecognizedAtom)
                }
            },

            RawValue::Str(s) => {
                if let Some(id) = self.program.str_table.get(&s) {
                    Ok(Value::StrConst(id))
                } else {
                    unimplemented!()
                }
            },

            RawValue::List(items) => {
                let addr = heap.alloc(ListLen(items.len() as u32))?;
                for (i, item) in items.into_iter().enumerate() {
                    let value = self.unmarshal(item, heap)?;
                    heap.set(addr, i as u32, value)?;
                }
                Ok(Value::ListAddr(addr))
            },
        }
    }

    fn get_menu(&self, item: LocalValue) -> Ret<Vec<(i32, RawValue)>> {
        let addr = item.value.as_addr()?;
        let len = item.heap.size_of(addr)?;
        let mut menu = Vec::with_capacity(len as usize);
        for i in 0 .. len {
            let choice_addr = item.heap.get(addr, i)?.as_addr()?;
            let test = item.heap.get(choice_addr, 0)?.as_bool()?;
            if test {
                let tag = item.heap.get(choice_addr, 1)?.as_int()?;
                let title = self.marshal({
                    item.heap.get(choice_addr, 2)?.in_heap(item.heap)
                })?;
                menu.push((tag, title));
            }
        }
        Ok(menu)
    }
}

impl RunQueue {
    fn fetch(&mut self) -> Box<Process> {
        if let Some(old) = self.dead.pop_front() {
            old
        } else {
            Box::new(Process::default())
        }
    }
}

impl SayToken {
    pub fn content(&self) -> String {
        String::from(self.1.clone())
    }

    pub fn reply(self) -> SayReplyToken {
        SayReplyToken(self.0)
    }
}

impl AskToken {
    pub fn content(&self) -> &[(i32, RawValue)] {
        &self.1
    }

    pub fn reply(self, i: i32) -> AskReplyToken {
        AskReplyToken(self.0, i, self.2)
    }
}

impl Reg {
    pub fn env() -> Self {
        Reg(0)
    }

    pub fn arg() -> Self {
        Reg(1)
    }
}

impl Value {
    pub fn tag(&self) -> Ret<TypeTag> {
        Ok(match self {
            &Value::Int(_) => TypeTag::Int,
            &Value::Atom(_) => TypeTag::Atom,
            &Value::ActorId(_) => TypeTag::Actor,
            &Value::StrConst(_) | &Value::StrAddr(_) => TypeTag::Str,
            &Value::ListAddr(_) | &Value::Capacity(_) => TypeTag::List,
            &Value::Undefined => return Err(RunErr::Uninitialized),
        })
    }

    pub fn as_int(self) -> Ret<i32> {
        match self {
            Value::Int(i) => Ok(i),
            _ => Err(RunErr::TypeMismatch(self, TypeTag::Int)),
        }
    }

    pub fn as_bool(self) -> Ret<bool> {
        Ok(self.as_int()? != 0)
    }

    fn as_addr(self) -> Ret<HeapAddr> {
        match self {
            Value::ListAddr(addr) => Ok(addr),
            _ => Err(RunErr::TypeMismatch(self, TypeTag::List)),
        }
    }

    fn in_heap<'a>(self, heap: &'a Heap) -> LocalValue<'a> {
        LocalValue {
            value: self,
            heap: heap,
        }
    }
}

impl Tag {
    fn private_clone(&self) -> Self {
        Tag(self.0, self.1)
    }
}

impl From<i32> for Value {
    fn from(i: i32) -> Self {
        Value::Int(i)
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Int(if b { 1 } else { 0 })
    }
}

impl From<ListLen> for Value {
    fn from(len: ListLen) -> Self {
        Value::Capacity(len.0)
    }
}

impl From<HeapAddr> for Value {
    fn from(addr: HeapAddr) -> Self {
        Value::ListAddr(addr)
    }
}

impl From<ActorId> for Value {
    fn from(id: ActorId) -> Self {
        Value::ActorId(id)
    }
}

impl From<IndexErr<Label>> for RunErr {
    fn from(err: IndexErr<Label>) -> Self {
        match err {
            IndexErr::OutOfBounds(k) => RunErr::NoSuchLabel(k),
            IndexErr::ReprOverflow(u) => RunErr::Unrepresentable(u),
        }
    }
}

impl From<IndexErr<InstrAddr>> for RunErr {
    fn from(err: IndexErr<InstrAddr>) -> Self {
        match err {
            IndexErr::OutOfBounds(k) => RunErr::FetchOutOfBounds(k),
            IndexErr::ReprOverflow(u) => RunErr::Unrepresentable(u),
        }
    }
}

impl From<IndexErr<EnvId>> for RunErr {
    fn from(err: IndexErr<EnvId>) -> Self {
        match err {
            IndexErr::OutOfBounds(k) => RunErr::EnvNotInitialized(k),
            IndexErr::ReprOverflow(u) => RunErr::Unrepresentable(u),
        }
    }
}

impl From<SayReplyToken> for InSignal {
    fn from(token: SayReplyToken) -> Self {
        InSignal::EndSay(token)
    }
}

impl From<AskReplyToken> for InSignal {
    fn from(token: AskReplyToken) -> Self {
        InSignal::EndAsk(token)
    }
}

impl From<SayToken> for OutSignal {
    fn from(token: SayToken) -> Self {
        OutSignal::Say(token)
    }
}

impl From<AskToken> for OutSignal {
    fn from(token: AskToken) -> Self {
        OutSignal::Ask(token)
    }
}

impl Default for Stack {
    fn default() -> Self {
        Stack {
            lower: StackFrame::default(),
            upper: None,
        }
    }
}

impl Default for Process {
    fn default() -> Self {
        Process {
            stack: Stack::default(),
            heap: Heap::default(),
            traps: vec![],
            inbox: VecDeque::with_capacity(8),
            op: Instr::Nop,
            pc: InstrAddr(0),
        }
    }
}

// FIXME: Make this work again in one form or another
/*
#[test]
fn simplest_init_possible() {
    let mut code = VecMap::with_capacity(1);
    code.push(Instr::Bye).unwrap();

    let program = Program {
        code: code,
        jump_table: VecMap::with_capacity(0),
        atom_table: StringInterner::new(),
        str_table: StringInterner::new(),
        env_table: VecMap::with_capacity(0),
    };

    program.init().unwrap();
}
*/
