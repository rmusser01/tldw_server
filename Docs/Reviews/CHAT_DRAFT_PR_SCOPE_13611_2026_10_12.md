# Chat remediation draft PR — TASK13611

The user authorized a PR targeting remote dev while remediation continues. The isolated branch is codex/chat-ux-remediation-20261004. Before this publication record it contains197 commits and1,095 changed files relative to dev2a8ffa78a839a53171a27bc1c5f3ce811bb892b3. This is an accumulated draft, not a claim that all ten NN/G heuristics or H2 acceptance gates are complete.

Scope inventory: shared UI744 files, canonical API package5, WebUI27, extension2, backend production64, backend tests61, Backlog145, documentation39, and small build/helper changes. Database-path changes are the two published schema source SQL files; no runtime database, .env, node_modules or .next paths appear in the tracked diff. Primary checkout/untracked preview artifacts remain outside this PR. All143 changed active task records (including this publication task) pass the canonical backlog-py check.

Review the layers separately:
1. Conversation setup, async model/owner state, draft/retry recovery, selection and roleplay usability; shared UI and both shells.
2. History ownership, retained-copy operation lifecycle, migrations and canonical API/storage authority.
3. Private native bytes, quota, exact references, source incarnation and preparation. Public H2 capabilities remain disabled pending their qualified callers.
4. Task records, ADR assessments and reproducible evidence. The closure record is Docs/Reviews/CHAT_NNG_REMEDIATION_CLOSURE_2026_10_09.md; latest preparation evidence is Docs/Reviews/CHAT_RETENTION_UPLOAD_13610_2026_10_12.md.

Recent scoped validation: TASK13610 final producer12 passed, original quota recovery5 passed, physical IO58 passed/59 skipped. Canonical2129 paths/3301 schemas fingerprint/path checks pass; actual touched-production Bandit and scoped Ruff/compile/hooks pass; two independent source reviewers clear the bounded changes. Earlier UI type/build/live-runtime evidence is in its associated task records. No single full-branch test run on the publication head is claimed.

Open gates: atomic retention publication and mapping; active Sync admission; native-source bridge and original-hash restore; rich retained fork/character snapshot independence; full cold/public HTTP/client journeys; install-level extension, physical assistive technology and scale qualification; physical PostgreSQL and full CI. Risk is high because this draft includes broad client-state changes and durable backend migrations. Do not roll back deployed databases by removing migration code; back up and use an explicit data recovery plan if migrations have run.

ADR assessment: no new decision in this publication unit. Existing049/050 and proposed069/070/071 remain relevant; proposed records are not promoted to Accepted. Repository policy Docs/superpowers/AI_GENERATED_PR_CHANGE_SUMMARY_POLICY_2026_04_17.md requires the human requester to write a Change summary before merge. AI-authored PR prose does not satisfy that gate. No merge or ready-for-review action is authorized here.

Remote publication/head read-back and draft URL will be recorded after creation. Worktree remains active for continued remediation.
