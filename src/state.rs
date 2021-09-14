use std::fmt::{self, Debug};

use rpds::Vector;

use crate::value::{cons, Value};

/// The current variable state of the miniKanren interpreter.
///
/// We use a persistent vector both for performance reasons, and to satisfy
/// Rust's ownership rules for [`Value`],
#[derive(Clone, Default, PartialEq, Eq)]
pub struct State(Vector<Option<Value>>);

impl State {
    pub(crate) fn apply(
        &self,
        f: impl FnOnce(&Vector<Option<Value>>) -> Vector<Option<Value>>,
    ) -> State {
        State(f(&self.0))
    }

    pub(crate) fn walk(&self, u: &Value) -> Value {
        match u {
            Value::Variable(i) => match &self.0[*i] {
                Some(x) => self.walk(x),
                None => u.clone(),
            },
            _ => u.clone(),
        }
    }

    pub(crate) fn walk_full(&self, u: &Value) -> Value {
        match u {
            Value::Variable(i) => match &self.0[*i] {
                Some(x) => self.walk_full(x),
                None => u.clone(),
            },
            Value::Cons(u, v) => cons(&self.walk_full(u), &self.walk_full(v)),
            _ => u.clone(),
        }
    }

    pub(crate) fn extend(&self, x: usize, v: Value) -> State {
        self.apply(|s| s.set(x, Some(v)).expect("invalid index in extend_state"))
    }

    pub(crate) fn add_fresh(&self, n: usize) -> State {
        self.apply(|s| {
            let mut s = s.clone();
            s.extend(std::iter::repeat(None).take(n));
            s
        })
    }

    /// Lookup how many variables are in the state.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if this state is empty (the zero state).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Create a state from a collection of optional values.
    pub fn from_vec(v: impl IntoIterator<Item = Option<Value>>) -> State {
        Self(v.into_iter().collect())
    }

    /// Return the idempotent first `k` variables from the state.
    pub fn finish(&self, k: usize) -> State {
        State::from_vec((0..k).map(|i| self.0[i].as_ref().map(|v| self.walk_full(v))))
    }
}

impl Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "state![")?;
        let mut first = true;
        for value in self.0.iter() {
            if !first {
                write!(f, ", ")?;
            }
            first = false;
            match value {
                None => write!(f, "_"),
                Some(Value::Atom(ref v)) => write!(f, "{:?}", v),
                Some(Value::Cons(ref u, ref v)) => write!(f, "({:?}, {:?})", u, v),
                Some(Value::Variable(i)) => write!(f, "(@{})", i),
            }?
        }
        write!(f, "]")?;
        Ok(())
    }
}

/// Convenience macro for making state objects using the `ToValue` trait.
#[macro_export]
macro_rules! state {
    () => {
        $crate::State::default()
    };
    ($($args:tt),+ $(,)?) => {
        {
            use $crate::ToValue;
            $crate::State::from_vec(::std::vec![
                $($crate::state_inner!(@STATE; $args)),+
            ])
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! state_inner {
    (@STATE; _) => {
        None
    };
    (@STATE; (@ $x:expr)) => {
        Some(Value::Variable($x))
    };
    (@STATE; $x:expr) => {
        Some($x.to_value())
    };
}
