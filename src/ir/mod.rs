pub mod pass;
pub mod visit;
pub mod allocate;
pub mod translate;

use string_interner::StringInterner;

#[derive(Clone, Debug)]
pub struct Program {
    pub blocks: Vec<Block>,
    pub ep_table: EpTable,
    pub str_table: StringInterner<StrId>,
    pub atom_table: StringInterner<AtomId>,
}

#[derive(Clone, Debug)]
pub enum EntryPoint {
    Init,

    Scene {
        name: String,
        argc: u32,
        env: Env,
    },

    Lambda {
        name: String,
    },
}

pub type EpTable = Vec<(Label, EntryPoint)>;

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct Env(pub u32);

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct Var(pub u32);

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Flag(pub u32);

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct Label(pub u32);

//#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub type StrId = ::vm::StrId;

//#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub type AtomId = ::vm::AtomId;

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum ConstRef {
    Atom(AtomId),
    Str(StrId),
}

#[derive(Clone, Debug)]
pub struct Block {
    pub info: BlockInfo,
    pub ops: Vec<Op>,
    pub exit: Exit,
}

#[derive(Clone, Debug)]
pub struct BlockInfo {
    pub id: u32,
    pub flags_needed: u32,
}

#[derive(Clone, Debug)]
pub struct TrapRef {
    pub label: Label,
    pub env: Var,
}

#[derive(Clone, Debug)]
pub struct FnCall {
    pub label: Label,
    pub argv: Var,
}

#[derive(Clone, Debug)]
pub struct Ptr {
    pub start_addr: Var,
    pub offset: u32,
}

#[derive(Clone, Debug)]
pub enum Op {
    Arm(TrapRef),
    Disarm(Label),
    //Discard(Rvalue),
    Export(Env, Var),
    Let(Var, Rvalue),
    Listen(TrapRef),
    Say(Var),
    Store(Var, Ptr),
    SendMsg(Var, Var),
    Set(Flag, Tvalue),
    Trace(Var),
    Wait(Var),
}

#[derive(Clone, Debug)]
pub enum Exit {
    EndProcess,
    Goto(Label),
    IfThenElse(Flag, Label, Label),
    Recur(FnCall),
    Return(bool),
}

#[derive(Clone, Debug)]
pub enum Rvalue {
    Var(Var),
    Int(i32),
    Add(Var, Var),
    Sub(Var, Var),
    Div(Var, Var),
    Mul(Var, Var),
    Roll(Var, Var),
    Load(Ptr),
    LoadArg(u32),
    LoadEnv(u32),
    FromBool(Flag),
    Spawn(FnCall),
    Splice(Vec<Var>),
    Alloc(u32),
    Const(ConstRef),
    MenuChoice(Var),
    PidOfSelf,
}

#[derive(Clone, Debug)]
pub enum Tvalue {
    Flag(Flag),
    Eql(Var, Var),
    Gt(Var, Var),
    Lt(Var, Var),
    Gte(Var, Var),
    Lte(Var, Var),
    HasLen(Var, u32),
    Nonzero(Var),
    True,
    False,
    And(Vec<Flag>),
    Or(Vec<Flag>),
    Not(Flag),
}

impl Var {
    pub fn at_offset(self, offset: u32) -> Ptr {
        Ptr {
            start_addr: self,
            offset: offset,
        }
    }
}

impl Label {
    pub fn with_argv(self, argv: Var) -> FnCall {
        FnCall {
            label: self,
            argv: argv,
        }
    }
}
