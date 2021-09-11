use ukanren::*;

fn appendo(first: Value, second: Value, out: Value) -> BoxedGoal<impl Iterator<Item = State>> {
    eq(&first, &())
        .and(eq(&second, &out))
        .or(fresh(move |a: Value, d: Value| {
            let out = out.clone();
            let second = second.clone();
            eq(&(a.clone(), d.clone()), &first).and(fresh(move |res: Value| {
                eq(&(a.clone(), res.clone()), &out).and(appendo(d.clone(), second.clone(), res))
            }))
        }))
        .boxed()
}

#[test]
fn forward_append() {
    let mut iter = fresh(|x| appendo([1, 2].to_value(), [3, 4].to_value(), x)).run(1);
    assert_eq!(iter.next(), Some(state![[1, 2, 3, 4]]));
    assert_eq!(iter.next(), None);
}

#[test]
fn guess_hole() {
    let mut iter = fresh(|x| {
        appendo(
            [1, 2].to_value(),
            [x, 4.to_value()].to_value(),
            [1, 2, 3, 4].to_value(),
        )
    })
    .run(1);
    assert_eq!(iter.next(), Some(state![3]));
    assert_eq!(iter.next(), None);
}

#[test]
fn inverse_append() {
    let goal = fresh(|x, y| appendo(x, y, [1, 2, 3, 4, 5].to_value()));
    assert_eq!(
        goal.run(2).collect::<Vec<_>>(),
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
