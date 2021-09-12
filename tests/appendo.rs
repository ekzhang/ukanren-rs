use ukanren::*;

fn appendo(first: Value, second: Value, out: Value) -> BoxedGoal<impl Iterator<Item = State>> {
    eq(&first, &())
        .and(eq(&second, &out))
        .or(fresh(move |a: Value, d: Value, res: Value| {
            eq(&(a.clone(), d.clone()), &first)
                .and(eq(&(a, res.clone()), &out))
                .and(appendo(d, second.clone(), res))
        }))
        .boxed()
}

#[test]
fn forward_append() {
    let mut iter = run(|x| appendo([1, 2].to_value(), [3, 4].to_value(), x));
    assert_eq!(iter.next(), Some(state![[1, 2, 3, 4]]));
    assert_eq!(iter.next(), None);
}

#[test]
fn guess_hole() {
    let mut iter = run(|x| {
        appendo(
            [1, 2].to_value(),
            [x, 4.to_value()].to_value(),
            [1, 2, 3, 4].to_value(),
        )
    });
    assert_eq!(iter.next(), Some(state![3]));
    assert_eq!(iter.next(), None);
}

#[test]
fn inverse_append() {
    let iter = run(|x, y| appendo(x, y, [1, 2, 3, 4, 5].to_value()));
    assert_eq!(
        iter.collect::<Vec<_>>(),
        vec![
            state![(), [1, 2, 3, 4, 5]],
            state![[1], [2, 3, 4, 5]],
            state![[1, 2], [3, 4, 5]],
            state![[1, 2, 3], [4, 5]],
            state![[1, 2, 3, 4], [5]],
            state![[1, 2, 3, 4, 5], ()],
        ]
    );
}

fn reverseo(first: Value, second: Value) -> BoxedGoal<impl Iterator<Item = State>> {
    eq(&first, &())
        .and(eq(&second, &()))
        .or(fresh(move |a: Value, d: Value, rd: Value| {
            eq(&(a.clone(), d.clone()), &first)
                .and(appendo(rd.clone(), cons(&a, &()), second.clone()))
                .and(reverseo(d, rd))
        }))
        .boxed()
}

#[test]
fn reverse_basic() {
    let iter = run(|x| reverseo(x, [1, 2, 3, 4, 5].to_value()));
    assert_eq!(iter.collect::<Vec<_>>(), vec![state![[5, 4, 3, 2, 1]]]);
}

#[test]
fn palindrome() {
    let iter = run(|x: Value| reverseo(x.clone(), x));
    assert!(iter.take(10).count() == 10);
}
