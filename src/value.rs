use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Result};

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
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self> {
        Ok(match value {
            Value::Int(n) if n >= 0 => n as usize,
            other => bail!("Can't convert {other:?} to usize"),
        })
    }
}

impl<'a> TryFrom<&'a Value> for &'a str {
    type Error = anyhow::Error;

    fn try_from(value: &'a Value) -> Result<Self> {
        Ok(match value {
            Value::String(s) => s.as_str(),
            other => bail!("Can't borrow {other:?} as a string"),
        })
    }
}

impl TryFrom<Value> for String {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self> {
        Ok(match value {
            Value::String(s) => s,
            Value::Int(n) => n.to_string(),
            Value::Atom(name) => format!("#{name}"),
            other => bail!("Can't format {other:?} as a string"),
        })
    }
}

impl TryFrom<Value> for ActorRef {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self> {
        Ok(match value {
            Value::Entity(aref) => aref,
            other => bail!("Cannot convert {other:?} to actor reference"),
        })
    }
}

impl TryFrom<Value> for Duration {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self> {
        match value {
            Value::Time(duration) => Ok(duration),
            other => bail!("Expected a duration: {other:?}"),
        }
    }
}

