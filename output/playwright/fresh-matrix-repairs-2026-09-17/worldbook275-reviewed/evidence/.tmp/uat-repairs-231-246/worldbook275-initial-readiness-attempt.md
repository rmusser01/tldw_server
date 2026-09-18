# Initial UAT275 readiness attempt — not causal evidence

The first real-QueryClient test run exited 1 before any mutation because the assertion used `findByRole` and observed the already-mounted parent output while its first query still displayed `loading`. Its reported failure expected `0` and received `loading`.

The test was corrected to wait for the first parent query to render `0`. The next retained run, `worldbook275-causal-red.log`, then failed after a successful add with the parent count still at `0` when `1` was expected. Only that later failure is used as causal evidence.

The initial raw log was overwritten by the corrected causal run before this distinction was recorded. This note preserves the limitation rather than treating the readiness failure as a product failure.
