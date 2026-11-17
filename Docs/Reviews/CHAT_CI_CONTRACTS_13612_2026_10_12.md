# PR3229 backend CI contract repairs — TASK13612

Actual Actions failures identified an unrecorded AuthNZ accounting isolation boundary, a truthy audit user filter with an obsolete ratchet location, an unassigned roleplay test, a stale invalid-content assertion and three trailing-whitespace sites.

The native reservation ledger is an AuthNZ accounting authority whose organization/team and admin/Billing queries must read multiple users. AuthNZ connections do not set app.current_user_id. Its explicit reviewed exemption records exact-key authenticated mutation/recovery checks and aggregate authority; it is not added to the uncovered baseline. Native chat quota intents retain owner RLS. Proposed ADR070 is clarified, not promoted; accepted ADR050 is unchanged.

An explicitly provided audit user filter now always invokes scope authorization and adds a parameterized predicate, including zero. Roleplay resume tests enter all five matching character-unit matrices. The integration assertion now requires invalid_content while retaining exact failed text and client correlation, preserving selected-history rejection. Only the reported trailing whitespace changes in the two UI files.

Causal RLS/predicate run:3 failed/29 passed,12 warnings11.14s. Explicit zero filter controls:2 failed/9 deselected,5 warnings2.23s. Green ratchets/admin/roleplay:188 passed/105 skipped,13 warnings63.44s. Targeted placeholder:1 passed/67 deselected,10 warnings7.43s. Shard guard:844 patterns/5090 test files/no newly uncovered files. Static scope: actual production Bandit0/HEAD0/errors0; four-file Ruff4/HEAD4 exact inherited findings, no new warnings; scoped new-test120-column AST Black/compile/diff checks pass.

Fresh bounded reviewer confirms no Critical/Important issue in this delta; its documentation correction on unlocked resolution and frozen aggregate attribution is incorporated. This is not final whole-PR approval. CI must rerun on the pushed head, and frontend/remaining H2/UX gates remain open. Private evidence /private/tmp/chat-ci-contracts-13612-* and original Actions logs /private/tmp/chat-pr3229-ci-*; raw startup/config output is not published.

Normal exact-owned13-file hooks pass without bypass; house filters and inherited findings are distinguished from the manual scoped checks. All bounded stages complete; only the owned plan retires to /private/tmp/IMPLEMENTATION_PLAN_chat_ci_contracts_13612_completed.md. Exact commit/push follows; final whole-PR review/rebase/merge remains pending completion of remaining work.
