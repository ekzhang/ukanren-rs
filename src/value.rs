use std::{any::Any, fmt::Debug, rc::Rc};

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
