use std::sync::Arc;
use std::time::Duration;

use thiserror::*;

/// Refcounted, dynamically typed Souvenir value. Safe to clone, reuse, mutate,
/// and send to other Actors, unlike in earlier implementations of Souvenir.
/// Designed to make reference cycles impossible.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i64),

    String(String),

    // TODO: Array(Arc<[Self]>),

    Time(Duration),

    Atom(Arc<str>),

    Entity(ActorRef),
}

#[derive(Error, Clone, Debug)]
pub enum ValueErr {
    #[error("Can't convert {value:?} to {typename}")]
    CantConvert {
        value: Value,
        typename: &'static str,
    },

    #[error("Can't borrow {value:?} as a &str")]
    CantBorrow {
        value: Value,
    },

    #[error("Can't format {value:?} as a string")]
    CantFormat {
        value: Value,
    },

    #[error("Can't convert {value:?} to actor reference")]
    NotActorRef {
        value: Value,
    },

    #[error("Expected a duration: {value:?}")]
    ExpectedDuration {
        value: Value,
    },
}

pub type Result<T, E=ValueErr> = std::result::Result<T, E>;

impl Value {
    pub const TRUE: Self = Value::Int(1);
    pub const FALSE: Self = Value::Int(0);

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Int(0) => false,
            _ => true,
        }
    }
}

/// Placeholder for Entity.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ActorRef(usize);

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        if value {
            Self::TRUE
        } else {
            Self::FALSE
        }
    }
}

impl TryFrom<Value> for usize {
    type Error = ValueErr;

    fn try_from(value: Value) -> Result<Self> {
        match value {
            Value::Int(n) if n >= 0 => Ok(n as usize),
            other => Err(ValueErr::CantConvert {
                value: other,
                typename: "usize",
            }),
        }
    }
}

impl<'a> TryFrom<&'a Value> for &'a str {
    type Error = ValueErr;

    fn try_from(value: &'a Value) -> Result<Self> {
        match value {
            Value::String(s) => Ok(s.as_str()),
            other => Err(ValueErr::CantBorrow { value: other.clone() }),
        }
    }
}

impl TryFrom<Value> for String {
    type Error = ValueErr;

    fn try_from(value: Value) -> Result<Self> {
        match value {
            Value::String(s) => Ok(s),
            Value::Int(n) => Ok(n.to_string()),
            Value::Atom(name) => Ok(format!("#{name}")),
            other => Err(ValueErr::CantFormat { value: other }),
        }
    }
}

impl TryFrom<Value> for ActorRef {
    type Error = ValueErr;

    fn try_from(value: Value) -> Result<Self> {
        match value {
            Value::Entity(aref) => Ok(aref),
            other => Err(ValueErr::NotActorRef { value: other, }),
        }
    }
}

impl TryFrom<Value> for Duration {
    type Error = ValueErr;

    fn try_from(value: Value) -> Result<Self> {
        match value {
            Value::Time(duration) => Ok(duration),
            other => Err(ValueErr::ExpectedDuration { value: other }),
        }
    }
}
