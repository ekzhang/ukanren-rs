use std::{cmp::Ordering, rc::Rc};

use itertools::{Interleave, Itertools};

use crate::state::State;
use crate::value::{ToValue, Value};

/// A goal that can be executed by the relational system.
pub trait Goal {
    /// The state iterator returned by the goal.
    type Iter: Iterator<Item = State>;

    /// Apply this goal to an initial state, returning a stream of satisfying states.
    fn apply(&self, s: &State) -> Self::Iter;

    /// Take the conjunction of this goal with another.
    ///
    /// # Example
    ///
    /// ```
    /// use ukanren::{eq, fresh, Goal};
    ///
    /// // Goal where `x` is equal to `5` and `y` is equal to `6`.
    /// fresh(|x, y| eq(&x, &5).and(eq(&y, &6)));
    /// ```
    fn and<G, I>(self, other: G) -> And<Self, G>
    where
        Self: Sized,
        G: Goal<Iter = I>,
        I: Iterator<Item = State>,
    {
        And(self, other)
    }

    /// Take the disjunction of this goal with another.
    ///
    /// # Example
    ///
    /// ```
    /// use ukanren::{eq, fresh, Goal};
    ///
    /// // Goal where `x` is equal to `5` or `x` is equal to `6`.
    /// fresh(|x| eq(&x, &5).or(eq(&x, &6)));
    /// ```
    fn or<G, I>(self, other: G) -> Or<Self, G>
    where
        Self: Sized,
        G: Goal<Iter = I>,
        I: Iterator<Item = State>,
    {
        Or(self, other)
    }

    /// Box this goal into a trait object, making it easier to name the type.
    ///
    /// # Example
    ///
    /// ```
    /// use ukanren::{eq, BoxedGoal, Goal, Value, State};
    ///
    /// fn animalo(x: Value) -> BoxedGoal<impl Iterator<Item = State>> {
    ///     eq(&x, &"dog").or(eq(&x, &"cat")).boxed()
    /// }
    /// ```
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
    ///
    /// This is a low-level function. Instead of calling this function directly,
    /// you should probably use the top-level [`run`] function instead, which
    /// infers the value of `k` from your input.
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
#[doc(hidden)]
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
#[doc(hidden)]
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
#[doc(hidden)]
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

/// Goal for unifying two values.
///
/// # Example
///
/// ```
/// use ukanren::{eq, fresh};
///
/// fresh(|x| eq(&x, &42));
/// ```
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
///
/// # Example
///
/// ```
/// use ukanren::{delay, eq, BoxedGoal, Goal, State, Value};
///
/// // Outputs an infinite stream of states where `x = 5`.
/// fn fives(x: Value) -> BoxedGoal<impl Iterator<Item = State>> {
///     eq(&x, &5).or(delay(move || fives(x.clone()))).boxed()
/// }
/// ```
pub fn delay<F, G, I>(f: F) -> BoxedGoal<LazyApplication<G, I>>
where
    F: Fn() -> G + Clone + 'static,
    G: Goal<Iter = I>,
    I: Iterator<Item = State>,
{
    (move |s: &State| LazyApplication::new(f(), s.clone())).boxed()
}

/// A lazy goal application that is not called until first polled for results.
#[doc(hidden)]
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
pub fn fresh<F, I, const N: usize>(f: F) -> impl Goal<Iter = I> + Clone
where
    F: Fresh<N, Iter = I> + Clone,
    I: Iterator<Item = State>,
{
    move |s: &State| f.call_fresh(s)
}

/// Trait for closures that can take fresh variables.
///
/// This is automatically implemented for closures taking up to 12 values.
pub trait Fresh<const N: usize> {
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
        impl<F, G, I> Fresh<$len> for F
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
impl_fresh!(9; 0, 1, 2, 3, 4, 5, 6, 7, 8);
impl_fresh!(10; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
impl_fresh!(11; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
impl_fresh!(12; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

/// Top-level entry point for running a goal with fresh variables.
///
/// # Example
///
/// ```
/// use ukanren::{eq, run, state, Goal};
///
/// let mut iter = run(|x| eq(&x, &5).or(eq(&x, &6)));
///
/// assert_eq!(iter.next(), Some(state![5]));
/// assert_eq!(iter.next(), Some(state![6]));
/// assert_eq!(iter.next(), None);
/// ```
pub fn run<F, I, const N: usize>(f: F) -> RunStream<I>
where
    F: Fresh<N, Iter = I> + Clone,
    I: Iterator<Item = State>,
{
    (move |s: &State| f.call_fresh(s)).run(N)
}
