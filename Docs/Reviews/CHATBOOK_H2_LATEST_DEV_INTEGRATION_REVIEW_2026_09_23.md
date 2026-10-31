# H2 Task 1.1 latest-dev integration review

Tracking: TASK-13261.5. Independent backend review of the H2 projection after merging H1 PR #2968 at `c13bad37fefd85ad15e3d0e9145db308a2a4723e` and server `dev` at `91e8bbf84c25d3afbba2bb53ed06280d44c35307`.

| Priority | Finding | Resolution |
|---|---|---|
| P2 | Valid H1 saved-turn metadata (`client_message_id`, `image_details`, `content_placeholder_reason=image_attachment`) made native fork capture reject a source chat. | Exclude the source retry ID from child content and digest; validate and retain ordered image detail; remove only the exact unedited version-1 generated image placeholder while preserving edited literal text. The native source read supplies the physical version internally, without changing the public selected-history wire. Real SQLite capture and projection regressions cover these cases. |
| P2 | Malformed nested `image_details` values such as `[{}]` and `[[]]` raised raw `TypeError` during validation. | Check that every detail is a string before value lookup. The malformed values now raise the typed unsupported-history error. |

Independent re-review found no remaining P1/P2 in the H2 Task 1.1 integration scope, including digest/identity, schema, RLS and public wire. The affected projection and H1 selected-history suites passed 128 tests with four inherited warnings. Ruff, compile, diff checks and touched Python Bandit passed; Bandit reported zero issues. This qualifies the integrated projection, not the still-unimplemented H2 durable store or client runtime.
