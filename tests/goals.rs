use ukanren::*;

#[test]
fn void_goal() {
    let mut iter = run(|x| eq(&x, &x));
    assert_eq!(iter.next(), Some(state![_]));
    assert_eq!(iter.next(), None);
}

#[test]
fn chained_disjunction() {
    let numbers = |x| eq(&x, &1).or(eq(&x, &2)).or(eq(&x, &3)).boxed();

    let mut iter = run(numbers);
    assert_eq!(iter.next(), Some(state![1]));
    assert_eq!(iter.next(), Some(state![3]));
    assert_eq!(iter.next(), Some(state![2]));
    assert_eq!(iter.next(), None);
}

#[test]
fn interleaving() {
    fn inf(x: Value, n: i32) -> BoxedGoal<impl Iterator<Item = State>> {
        eq(&x, &n).or(delay(move || inf(x.clone(), n))).boxed()
    }

    let mut iter = run(|x: Value| inf(x.clone(), 5).or(inf(x, 6)));
    for _ in 0..10 {
        assert_eq!(iter.next(), Some(state![5]));
        assert_eq!(iter.next(), Some(state![6]));
    }
}

#[test]
fn two_equal() {
    let mut iter = run(|x, y| eq(&x, &y));
    assert_eq!(iter.next(), Some(state![_, (@0)]));
    assert_eq!(iter.next(), None);
}

#[test]
fn multi_equal() {
    let mut iter = run(|x, y, z, w| eq(&x, &y).and(eq(&z, &w)).or(eq(&x, &z).and(eq(&y, &w))));
    assert_eq!(iter.next(), Some(state![_, (@0), _, (@2)]));
    assert_eq!(iter.next(), Some(state![_, _, (@0), (@1)]));
    assert_eq!(iter.next(), None);
}

#[test]
fn list_equal() {
    let mut iter = run(|x, y| eq(&[x, y], &["hello", "world"]));
    assert_eq!(iter.next(), Some(state!["hello", "world"]));
    assert_eq!(iter.next(), None);
}
