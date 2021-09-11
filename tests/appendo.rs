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

fn reverseo(first: Value, second: Value) -> BoxedGoal<impl Iterator<Item = State>> {
    eq(&first, &())
        .and(eq(&second, &()))
        .or(fresh(move |a: Value, d: Value| {
            let second = second.clone();
            eq(&(a.clone(), d.clone()), &first).and(fresh(move |rd: Value| {
                appendo(rd.clone(), (a.clone(), ()).to_value(), second.clone())
                    .and(reverseo(d.clone(), rd.clone()))
            }))
        }))
        .boxed()
}

#[test]
fn reverse_basic() {
    let goal = fresh(|x| reverseo(x, [1, 2, 3, 4, 5].to_value()));
    assert_eq!(
        goal.run(1).collect::<Vec<_>>(),
        vec![state![[5, 4, 3, 2, 1]]],
    );
}

#[test]
fn palindrome() {
    let goal = fresh(|x: Value| reverseo(x.clone(), x));
    assert!(goal.run(1).take(10).collect::<Vec<_>>().len() == 10);
}
