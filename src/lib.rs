//! An implementation of the microKanren relational programming system in Rust.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::any::Any;
use std::fmt::Debug;

use itertools::{Interleave, Itertools};
use rpds::vector::Vector;

/// An object in microKanren that can be unified.
#[derive(Debug)]
pub enum Value {
    /// A variable with a specific ID.
    Variable(usize),
    /// An atomic term, compared for basic equality.
    Atom(Box<dyn Atom>),
    /// A list containing multiple values.
    List(Vec<Value>),
}

impl Clone for Value {
    fn clone(&self) -> Self {
        match self {
            Self::Variable(n) => Self::Variable(*n),
            Self::Atom(x) => Self::Atom(x.box_clone()),
            Self::List(v) => Self::List(v.clone()),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Variable(x), Value::Variable(y)) => x == y,
            (Value::Atom(x), Value::Atom(y)) => x.eq(y.as_ref()),
            (Value::List(x), Value::List(y)) => x == y,
            _ => false,
        }
    }
}

impl Eq for Value {}

/// Trait representing an atomic type.
pub trait Atom: Debug {
    /// Compare two atomic type references for equality.
    fn eq(&self, other: &dyn Atom) -> bool;
    /// Convert this reference to an [`Any`] reference.
    fn as_any(&self) -> &dyn Any;
    /// Clone the current atom in boxed form.
    fn box_clone(&self) -> Box<dyn Atom>;
}

impl<T: 'static + Eq + Debug + Clone> Atom for T {
    fn eq(&self, other: &dyn Atom) -> bool {
        match other.as_any().downcast_ref() {
            Some(other) => self.eq(other),
            None => false,
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn box_clone(&self) -> Box<dyn Atom> {
        Box::new(self.clone())
    }
}

macro_rules! impl_atom {
    ($t:ty) => {
        impl From<$t> for Value {
            fn from(x: $t) -> Self {
                Value::Atom(Box::new(x))
            }
        }
    };

    ($t:ty, $($ts:ty),+) => {
        impl_atom!($t);
        impl_atom!($($ts),+);
    };
}

impl_atom!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, isize, usize);
impl_atom!(&'static str);
impl_atom!(String);

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self {
        Value::List(v.into_iter().map(Into::into).collect())
    }
}

/// The current variable state of the miniKanren interpreter.
///
/// We use a persistent vector both for performance reasons, and to satisfy
/// Rust's ownership rules for [`Value`],
type State = Vector<Option<Value>>;

fn walk(u: &Value, s: &State) -> Value {
    match u {
        Value::Variable(i) => match &s[*i] {
            Some(x) => walk(x, s),
            None => u.clone(),
        },
        _ => u.clone(),
    }
}

fn extend_state(x: usize, v: Value, s: &State) -> State {
    s.set(x, Some(v)).expect("invalid index in extend_state")
}

/// Goal for unifying two values.
pub fn eq(
    u: impl Into<Value>,
    v: impl Into<Value>,
) -> impl Goal<Iter = impl Iterator<Item = State>> {
    let u = u.into();
    let v = v.into();
    move |s: &State| unify(&u, &v, s).into_iter()
}

fn unify(u: &Value, v: &Value, s: &State) -> Option<State> {
    let u = walk(u, s);
    let v = walk(v, s);
    match (u, v) {
        (Value::Variable(u), Value::Variable(v)) if u == v => Some(s.clone()),
        (Value::Variable(u), v) => Some(extend_state(u, v, s)),
        (u, Value::Variable(v)) => Some(extend_state(v, u, s)),
        (Value::List(_u), Value::List(_v)) => {
            todo!()
        }
        (u @ Value::Atom(_), v @ Value::Atom(_)) if u == v => Some(s.clone()),
        _ => None,
    }
}

/// Goal that introduces a fresh relational variable.
pub fn call_fresh<G, I>(f: impl Fn(Value) -> G) -> impl Goal<Iter = I>
where
    G: Goal<Iter = I>,
    I: Iterator<Item = State>,
{
    move |s: &State| {
        let var = Value::Variable(s.len());
        f(var).apply(&s.push_back(None))
    }
}

/// A goal that can be executed by the relational system.
pub trait Goal {
    /// The state iterator returned by the goal.
    type Iter: Iterator<Item = State>;

    /// Apply this goal to an initial state, returning a stream of satisfying states.
    fn apply(self, s: &State) -> Self::Iter;

    /// Take the conjunction of this goal with another.
    fn and<G, I>(self, other: G) -> And<Self, G>
    where
        Self: Sized,
        G: Goal<Iter = I>,
        I: Iterator<Item = State>,
    {
        And(self, other)
    }

    /// Take the disjunction of this goal with another.
    fn or<G, I>(self, other: G) -> Or<Self, G>
    where
        Self: Sized,
        G: Goal<Iter = I>,
        I: Iterator<Item = State>,
    {
        Or(self, other)
    }
}

impl<G, I> Goal for G
where
    G: Fn(&State) -> I,
    I: Iterator<Item = State>,
{
    type Iter = I;

    fn apply(self, s: &State) -> I {
        self(s)
    }
}

/// A goal constructed from the conjunction of two goals.
#[derive(Clone, Copy)]
pub struct And<G1, G2>(G1, G2);

impl<G1, G2, I1, I2> Goal for And<G1, G2>
where
    G1: Goal<Iter = I1>,
    G2: Goal<Iter = I2> + Clone + 'static,
    I1: Iterator<Item = State>,
    I2: Iterator<Item = State>,
{
    // The boxing and 'static lifetime are necessary because Rust does not yet
    // have stable generic associated types (GAT) or higher-kinded types.
    type Iter = std::iter::FlatMap<I1, I2, Box<dyn Fn(State) -> I2>>;

    fn apply(self, s: &State) -> Self::Iter {
        let Self(g1, g2) = self;
        g1.apply(s)
            .flat_map(Box::new(move |s| g2.clone().apply(&s)))
    }
}

/// A goal constructed from the disjunction of two goals.
#[derive(Clone, Copy)]
pub struct Or<G1, G2>(G1, G2);

impl<G1, G2, I1, I2> Goal for Or<G1, G2>
where
    G1: Goal<Iter = I1>,
    G2: Goal<Iter = I2>,
    I1: Iterator<Item = State>,
    I2: Iterator<Item = State>,
{
    type Iter = Interleave<I1, I2>;

    fn apply(self, s: &State) -> Self::Iter {
        self.0.apply(s).interleave(self.1.apply(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atom_cmp() {
        assert_eq!(Value::from(2), Value::from(2));
        assert_eq!(Value::from(-42i8), Value::from(-42i8));
        assert_eq!(Value::from("hello"), Value::from("hello"));
        assert_ne!(Value::from(-42), Value::from(-42i8));
        assert_ne!(Value::from("hello"), Value::from(1));
    }

    #[test]
    fn list_cmp() {
        assert_eq!(Value::from(vec![2, 5, 6]), Value::from(vec![2, 5, 6]));
        assert_eq!(
            Value::from(vec![Value::from(2), Value::from("5")]),
            Value::from(vec![Value::from(2), Value::from("5")]),
        );
        assert_ne!(Value::from(vec![2]), Value::from(vec![4]));
    }
}
