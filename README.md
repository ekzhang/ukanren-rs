# µKanren-rs

[<img alt="github" src="https://img.shields.io/badge/github-ekzhang/ukanren--rs-8da0cb?style=for-the-badge&labelColor=555555&logo=github" height="20">](https://github.com/ekzhang/ukanren-rs)
[<img alt="crates.io" src="https://img.shields.io/crates/v/ukanren.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/ukanren)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-ukanren-66c2a5?style=for-the-badge&labelColor=555555&logoColor=white&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDUxMiA1MTIiPjxwYXRoIGZpbGw9IiNmNWY1ZjUiIGQ9Ik00ODguNiAyNTAuMkwzOTIgMjE0VjEwNS41YzAtMTUtOS4zLTI4LjQtMjMuNC0zMy43bC0xMDAtMzcuNWMtOC4xLTMuMS0xNy4xLTMuMS0yNS4zIDBsLTEwMCAzNy41Yy0xNC4xIDUuMy0yMy40IDE4LjctMjMuNCAzMy43VjIxNGwtOTYuNiAzNi4yQzkuMyAyNTUuNSAwIDI2OC45IDAgMjgzLjlWMzk0YzAgMTMuNiA3LjcgMjYuMSAxOS45IDMyLjJsMTAwIDUwYzEwLjEgNS4xIDIyLjEgNS4xIDMyLjIgMGwxMDMuOS01MiAxMDMuOSA1MmMxMC4xIDUuMSAyMi4xIDUuMSAzMi4yIDBsMTAwLTUwYzEyLjItNi4xIDE5LjktMTguNiAxOS45LTMyLjJWMjgzLjljMC0xNS05LjMtMjguNC0yMy40LTMzLjd6TTM1OCAyMTQuOGwtODUgMzEuOXYtNjguMmw4NS0zN3Y3My4zek0xNTQgMTA0LjFsMTAyLTM4LjIgMTAyIDM4LjJ2LjZsLTEwMiA0MS40LTEwMi00MS40di0uNnptODQgMjkxLjFsLTg1IDQyLjV2LTc5LjFsODUtMzguOHY3NS40em0wLTExMmwtMTAyIDQxLjQtMTAyLTQxLjR2LS42bDEwMi0zOC4yIDEwMiAzOC4ydi42em0yNDAgMTEybC04NSA0Mi41di03OS4xbDg1LTM4Ljh2NzUuNHptMC0xMTJsLTEwMiA0MS40LTEwMi00MS40di0uNmwxMDItMzguMiAxMDIgMzguMnYuNnoiPjwvcGF0aD48L3N2Zz4K" height="20">](https://docs.rs/ukanren)
[<img alt="build status" src="https://img.shields.io/github/workflow/status/ekzhang/ukanren-rs/CI/main?style=for-the-badge" height="20">](https://github.com/ekzhang/ukanren-rs/actions?query=branch%3Amain)

This is a Rust implementation of µKanren, a featherweight relational programming
language. See the original Scheme implementation
[here](https://github.com/jasonhemann/microKanren) for reference.

## Features

- Structural unification of Scheme-like cons cells.
- Streams implemented with the `Iterator` trait.
- State representation using a persistent vector with triangular substitutions.
- Conjunction, disjunction, and `fresh` based on traits (macro-free API).
- Lazy goal evaluation using inverse-η delay.
- Integer, `bool`, `char`, `&str`, and unit type atoms.
- Explicit `ToValue` trait that converts vectors and arrays into cons-lists.
- Convenience macro `state!` to inspect and specify state.

## Usage

Here's a simple example, which defines and uses the `appendo` predicate.

```rust
use ukanren::*;

fn appendo(first: Value, second: Value, out: Value) -> BoxedGoal<impl Iterator<Item = State>> {
    eq(&first, &())
        .and(eq(&second, &out))
        .or(fresh(move |a: Value, d: Value, res: Value| {
            eq(&(a.clone(), d.clone()), &first)
                .and(eq(&(a.clone(), res.clone()), &out))
                .and(appendo(d.clone(), second.clone(), res))
        }))
        .boxed()
}

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
    ],
);
```

More examples can be found in the `tests/` folder and the API documentation.

<br>

<sup>
Made by Eric Zhang for
<a href="https://pl-design-seminar.seas.harvard.edu/">CS 252r</a>. All code is
licensed under the <a href="LICENSE">MIT License</a>.
</sup>
