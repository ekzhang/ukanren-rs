use ukanren::*;

#[test]
fn void_goal() {
    let mut iter = fresh(|x| eq(&x, &x)).run(1);
    assert_eq!(iter.next(), Some(state![_]));
    assert_eq!(iter.next(), None);
}

#[test]
fn chained_disjunction() {
    let numbers = |x| eq(&x, &1).or(eq(&x, &2)).or(eq(&x, &3)).boxed();

    let mut iter = fresh(numbers).run(1);
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

    let mut iter = fresh(|x: Value| inf(x.clone(), 5).or(inf(x, 6))).run(1);
    for _ in 0..10 {
        assert_eq!(iter.next(), Some(state![5]));
        assert_eq!(iter.next(), Some(state![6]));
    }
}

#[test]
fn two_equal() {
    let mut iter = fresh(|x, y| eq(&x, &y)).run(2);
    assert_eq!(iter.next(), Some(state![_, (@0)]));
    assert_eq!(iter.next(), None);
}

#[test]
fn multi_equal() {
    let mut iter =
        fresh(|x, y, z, w| eq(&x, &y).and(eq(&z, &w)).or(eq(&x, &z).and(eq(&y, &w)))).run(4);
    assert_eq!(iter.next(), Some(state![_, (@0), _, (@2)]));
    assert_eq!(iter.next(), Some(state![_, _, (@0), (@1)]));
    assert_eq!(iter.next(), None);
}

#[test]
fn list_equal() {
    let mut iter = fresh(|x, y| eq(&[x, y], &["hello", "world"])).run(2);
    assert_eq!(iter.next(), Some(state!["hello", "world"]));
    assert_eq!(iter.next(), None);
}
