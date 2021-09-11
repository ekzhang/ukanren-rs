//! An implementation of the _µKanren_ relational programming system in Rust.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::{
    any::Any,
    cmp::Ordering,
    fmt::{self, Debug},
    rc::Rc,
};

use itertools::{Interleave, Itertools};
use rpds::Vector;

/// An object in µKanren that can be unified.
#[derive(Debug, Clone)]
pub enum Value {
    /// A variable with a specific ID.
    Variable(usize),

    /// An atomic term, compared for basic equality.
    Atom(Rc<dyn Atom>),

    /// A cons cell containing a pair of values.
    Cons(Rc<Value>, Rc<Value>),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Variable(x), Value::Variable(y)) => x == y,
            (Value::Atom(x), Value::Atom(y)) => x.eq(y.as_ref()),
            (Value::Cons(x1, x2), Value::Cons(y1, y2)) => x1 == y1 && x2 == y2,
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
}

impl<T: 'static + Eq + Debug> Atom for T {
    fn eq(&self, other: &dyn Atom) -> bool {
        match other.as_any().downcast_ref() {
            Some(other) => self.eq(other),
            None => false,
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A type that can be converted to a value.
pub trait ToValue {
    /// Convert this type to a value.
    fn to_value(&self) -> Value;
}

impl ToValue for Value {
    fn to_value(&self) -> Value {
        self.clone()
    }
}

macro_rules! impl_atom_to_value {
    ($t:ty) => {
        impl ToValue for $t {
            fn to_value(&self) -> Value {
                Value::Atom(Rc::new(*self))
            }
        }
    };

    ($t:ty, $($ts:ty),+) => {
        impl_atom_to_value!($t);
        impl_atom_to_value!($($ts),+);
    };
}

impl_atom_to_value!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, isize, usize);
impl_atom_to_value!(bool, char, ());
impl_atom_to_value!(&'static str);

/// Construct a cons cell from two values.
pub fn cons(u: &impl ToValue, v: &impl ToValue) -> Value {
    Value::Cons(Rc::new(u.to_value()), Rc::new(v.to_value()))
}

/// Construct a list out of cons cells.
pub fn list<'a>(items: impl IntoIterator<Item = &'a (impl ToValue + 'a)>) -> Value {
    let mut it = items.into_iter();
    match it.next() {
        Some(v) => cons(v, &list(it)),
        None => ().to_value(),
    }
}

impl<T: ToValue, const N: usize> ToValue for [T; N] {
    fn to_value(&self) -> Value {
        list(self)
    }
}

impl<T: ToValue> ToValue for [T] {
    fn to_value(&self) -> Value {
        list(self)
    }
}

impl<T: ToValue> ToValue for Vec<T> {
    fn to_value(&self) -> Value {
        list(self)
    }
}

impl<T: ToValue, U: ToValue> ToValue for (T, U) {
    fn to_value(&self) -> Value {
        cons(&self.0, &self.1)
    }
}

/// The current variable state of the miniKanren interpreter.
///
/// We use a persistent vector both for performance reasons, and to satisfy
/// Rust's ownership rules for [`Value`],
#[derive(Clone, Default, PartialEq, Eq)]
pub struct State(Vector<Option<Value>>);

impl State {
    fn apply(&self, f: impl FnOnce(&Vector<Option<Value>>) -> Vector<Option<Value>>) -> State {
        State(f(&self.0))
    }

    fn walk(&self, u: &Value) -> Value {
        match u {
            Value::Variable(i) => match &self.0[*i] {
                Some(x) => self.walk(x),
                None => u.clone(),
            },
            _ => u.clone(),
        }
    }

    fn walk_full(&self, u: &Value) -> Value {
        match u {
            Value::Variable(i) => match &self.0[*i] {
                Some(x) => self.walk_full(x),
                None => u.clone(),
            },
            Value::Cons(u, v) => cons(&self.walk_full(u), &self.walk_full(v)),
            _ => u.clone(),
        }
    }

    fn extend(&self, x: usize, v: Value) -> State {
        self.apply(|s| s.set(x, Some(v)).expect("invalid index in extend_state"))
    }

    fn add_fresh(&self, n: usize) -> State {
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
                &None => write!(f, "_"),
                &Some(Value::Atom(ref v)) => write!(f, "{:?}", v),
                &Some(Value::Cons(ref u, ref v)) => write!(f, "({:?}, {:?})", u, v),
                &Some(Value::Variable(i)) => write!(f, "(@{})", i),
            }?
        }
        write!(f, "]")?;
        Ok(())
    }
}

/// Convenience macro for constructing new state objects. This requires the
/// [`ToValue`] trait to be in scope.
#[macro_export]
macro_rules! state {
    () => {
        $crate::State::default()
    };
    ($($args:tt),+ $(,)?) => {
        $crate::State::from_vec(::std::vec![
            $($crate::state_inner!(@STATE; $args)),+
        ])
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

/// Goal for unifying two values.
pub fn eq(
    u: &impl ToValue,
    v: &impl ToValue,
) -> impl Goal<Iter = std::option::IntoIter<State>> + Clone + 'static {
    let u = u.to_value();
    let v = v.to_value();
    move |s: &State| unify(&u, &v, s).into_iter()
}

fn unify(u: &Value, v: &Value, s: &State) -> Option<State> {
    let u = s.walk(u);
    let v = s.walk(v);
    match (u, v) {
        (Value::Variable(u), Value::Variable(v)) => match u.cmp(&v) {
            Ordering::Equal => Some(s.clone()),
            Ordering::Greater => Some(s.extend(u, Value::Variable(v))),
            Ordering::Less => Some(s.extend(v, Value::Variable(u))),
        },
        (Value::Variable(u), v) => Some(s.extend(u, v)),
        (u, Value::Variable(v)) => Some(s.extend(v, u)),
        (Value::Cons(u1, u2), Value::Cons(v1, v2)) => {
            let s = unify(&u1, &v1, s)?;
            unify(&u2, &v2, &s)
        }
        (u @ Value::Atom(_), v @ Value::Atom(_)) if u == v => Some(s.clone()),
        _ => None,
    }
}

/// Goal that introduces inverse-η delay to handle infinite streams.
pub fn delay<F, G, I>(f: F) -> BoxedGoal<LazyApplication<G, I>>
where
    F: Fn() -> G + Clone + 'static,
    G: Goal<Iter = I>,
    I: Iterator<Item = State>,
{
    (move |s: &State| LazyApplication::new(f(), s.clone())).boxed()
}

/// A lazy goal application that is not called until first polled for results.
#[derive(Clone)]
pub enum LazyApplication<G, I> {
    /// Lazy goal-state application that returns an iterator.
    Lazy(G, State),
    /// Realized iterator returned from the goal.
    Iterator(I),
}

impl<G, I> LazyApplication<G, I> {
    fn new(goal: G, state: State) -> Self {
        Self::Lazy(goal, state)
    }
}

impl<G, I> Iterator for LazyApplication<G, I>
where
    G: Goal<Iter = I>,
    I: Iterator<Item = State>,
{
    type Item = State;

    fn next(&mut self) -> Option<Self::Item> {
        if let LazyApplication::Lazy(_, _) = self {
            take_mut::take(self, |value| match value {
                LazyApplication::Lazy(goal, state) => LazyApplication::Iterator(goal.apply(&state)),
                LazyApplication::Iterator(_) => unreachable!(),
            });
        }
        match self {
            LazyApplication::Iterator(it) => it.next(),
            _ => unreachable!(),
        }
    }
}

/// Goal that introduces one or more fresh relational variables.
pub fn fresh<'a, F, I, T>(f: F) -> impl Goal<Iter = I> + Clone + 'a
where
    F: Fresh<T, Iter = I> + Clone + 'static,
    I: Iterator<Item = State>,
{
    move |s: &State| f.call_fresh(s)
}

/// Trait for closures that can take fresh variables.
///
/// This is automatically implemented for closures taking up to 8 values.
pub trait Fresh<T> {
    /// The iterator returned by the fresh closure.
    type Iter: Iterator<Item = State>;

    /// Call this closure on an initial state, adding fresh variables.
    fn call_fresh(&self, s: &State) -> Self::Iter;
}

macro_rules! impl_fresh {
    (@VALUE; $num:expr) => {
        Value
    };
    ($len:expr; $($nums:expr),+) => {
    impl<F, G, I> Fresh<($(impl_fresh!(@VALUE; $nums),)+)> for F
        where
            F: Fn($(impl_fresh!(@VALUE; $nums),)+) -> G,
            G: Goal<Iter = I>,
            I: Iterator<Item = State>,
        {
            type Iter = I;

            fn call_fresh(&self, s: &State) -> Self::Iter {
                let len = s.len();
                self($(Value::Variable(len + $nums)),+).apply(&s.add_fresh($len))
            }
        }
    };
}

impl_fresh!(1; 0);
impl_fresh!(2; 0, 1);
impl_fresh!(3; 0, 1, 2);
impl_fresh!(4; 0, 1, 2, 3);
impl_fresh!(5; 0, 1, 2, 3, 4);
impl_fresh!(6; 0, 1, 2, 3, 4, 5);
impl_fresh!(7; 0, 1, 2, 3, 4, 5, 6);
impl_fresh!(8; 0, 1, 2, 3, 4, 5, 6, 7);

/// A goal that can be executed by the relational system.
pub trait Goal {
    /// The state iterator returned by the goal.
    type Iter: Iterator<Item = State>;

    /// Apply this goal to an initial state, returning a stream of satisfying states.
    fn apply(&self, s: &State) -> Self::Iter;

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

    /// Box this goal, which simplifies types at the expense of performance.
    fn boxed(self) -> BoxedGoal<Self::Iter>
    where
        Self: Sized + 'static,
    {
        BoxedGoal {
            inner: Rc::new(self),
        }
    }

    /// Evaluate this goal on an empty state, returning a stream of results.
    ///
    /// These results contain normalized forms of the first `k` variables, to
    /// avoid including auxiliary data that is not relevant to us.
    fn run(&self, k: usize) -> RunStream<Self::Iter> {
        RunStream {
            inner: self.apply(&State::default()),
            k,
        }
    }
}

impl<G, I> Goal for G
where
    G: Fn(&State) -> I,
    I: Iterator<Item = State>,
{
    type Iter = I;

    fn apply(&self, s: &State) -> I {
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

    fn apply(&self, s: &State) -> Self::Iter {
        let Self(g1, g2) = self;
        let g2 = g2.clone();
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

    fn apply(&self, s: &State) -> Self::Iter {
        self.0.apply(s).interleave(self.1.apply(s))
    }
}

/// A boxed goal for type erasure, constructed from [`Goal::boxed`].
pub struct BoxedGoal<T> {
    inner: Rc<dyn Goal<Iter = T>>,
}

impl<T> Clone for BoxedGoal<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }
}

impl<T> Goal for BoxedGoal<T>
where
    T: Iterator<Item = State> + 'static,
{
    type Iter = Box<dyn Iterator<Item = State>>;

    fn apply(&self, s: &State) -> Self::Iter {
        Box::new(self.inner.apply(s))
    }
}

/// Iterator adapter created by [`Goal::run`].
pub struct RunStream<I> {
    inner: I,
    k: usize,
}

impl<I> Iterator for RunStream<I>
where
    I: Iterator<Item = State>,
{
    type Item = State;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|s| s.finish(self.k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(v: &impl ToValue) -> Value {
        v.to_value()
    }

    #[test]
    fn atom_cmp() {
        assert_eq!(value(&2), value(&2));
        assert_eq!(value(&-42i8), value(&-42i8));
        assert_eq!(value(&"hello"), value(&"hello"));
        assert_ne!(value(&-42), value(&-42i8));
        assert_ne!(value(&"hello"), value(&1));
    }

    #[test]
    fn list_cmp() {
        assert_eq!(value(&[2, 5, 6]), value(&[2, 5, 6]));
        assert_eq!(
            value(&[value(&2), value(&"5")]),
            value(&[value(&2), value(&"5")]),
        );
        assert_ne!(value(&[2]), value(&[4]));
    }
}
