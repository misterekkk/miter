# E8 amendment 1: `min_sum_hessian_in_leaf` in symmetric trees

Written before any run of the arms below.

## Finding

During development the arm `MSH1_CLF` (`min_sum_hessian_in_leaf=1.0`) tied `D0` on every dataset.
Cause: in symmetric trees the bound is applied only when `lambda_l2 == 0`
(`eval_multi_leaf_gain`: `if (lam == 0.0 && (hl < min_h || hr < min_h))`), so with the default
`lambda_l2=3` it is silently ignored. Leaf-wise trees always apply it. This dates from the
introduction of symmetric trees (84a89c8). `MSH1_CLF` therefore measures nothing and is reported as
void.

## Change (branch `exp/e8`)

New switch `enforce_min_hessian` (default false, models unchanged: 54/54 hashes identical to `main`;
212 Python and 89 C++ tests pass). When true, symmetric split searches apply
`min_sum_hessian_in_leaf` whatever `lambda_l2` is (`sym_min_hessian()`), as leaf-wise trees do.

## Added arms (H3, development set `dev`, then the same procedure and rules)

- `MSHFIX`: `enforce_min_hessian=True` at the default bound 1e-3, all tasks (the fix alone).
- `MSHFIX01_CLF`: classification only, `enforce_min_hessian=True, min_sum_hessian_in_leaf=0.1`.
- `MSHFIX1_CLF`: classification only, `enforce_min_hessian=True, min_sum_hessian_in_leaf=1.0`.

They compete with the other H3 arms for H3's candidate. If the fix is adopted, the switch is removed
and symmetric trees always apply the bound.
