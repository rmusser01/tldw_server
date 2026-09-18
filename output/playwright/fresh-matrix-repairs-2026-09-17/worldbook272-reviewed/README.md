# Reviewed WorldBook attachment display repair (UAT272)

Both character clients prefer the canonical collection route. Attachment management requests disabled associations explicitly, defers hydration until needed, surfaces transport failures, and offers a retry action. Default association reads, ownership rules, and rate-limit policy remain unchanged.

Independent verification:63 maintained frontend tests and30 actual SQLite/PostgreSQL read-contract tests pass without skips. Baseline-only overlays retain six expected causal failures across the review rounds. Eight-file lint comparison has no new diagnostics; existing lint/compiler debt and TypeScript security-scanner limits remain explicit. Original PostgreSQL WorldBook1/Character6 display and disable/reload/re-enable acceptance are pending. No full-matrix acceptance is claimed.

Evidence is exact bytes or lossless gzip with credential scanning; any excluded inputs/links are identified in the manifest.
