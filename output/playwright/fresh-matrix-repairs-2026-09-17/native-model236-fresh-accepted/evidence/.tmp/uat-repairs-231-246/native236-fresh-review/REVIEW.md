# UAT236 fresh-profile AC3 review

**CLEAR for the previously missing native AC3: fresh setup to TestBot library Chat, intended completion, without model reselection.** Independent audit passes **21 checks over43 hashed inputs**.

This is new evidence. The original picker-only review and `AC3-RETENTION-CORRECTION.md` remain unchanged; they correctly did not establish AC3 at the time.

## One preserved fresh application profile

The native sequence uses `mcp251-fresh-targeted-20260917`, PostgreSQL single-user API18704/WebUI18784, source **`a7d3155a567afb25982eb360ea24b973cc3249c9`**. The earlier independently reviewed MCP251 wizard configured and validated the real local provider, saved it, and reached First chat. That38-check review supplies the fresh initialization and restricted-role fixture provenance.

The wizard is resumed normally after a pause. **Send test chat** at **2026-09-18 01:55:37.229 UTC** returns200 at **01:55:38.866**, status`ready`, provider`llamacpp`, no failure category. Normal visible API-key entry then grants WebUI access. The private credential helper uses form filling/clicks; it does not mutate model selection or browser storage. Credential values are not retained in this review.

## Actual library entry and first completion

The normal Characters → New character form creates **E2E-TestBot**, ID3, with exact instructions:

> You are E2E-TestBot. Always respond with exactly: BEEP BOOP.

Creation returns201 at **01:59:23.119**. The same helper clicks the actual library **Chat as E2E-TestBot** button. The normal composer sends **Hello, who are you?**.

At **02:00:40.587**, the server creates initial conversation `a1979872-5e20-4288-8d06-062545bfeca2` for that Character, with no parent branch. The completion request reaches200. It sends provider`llama.cpp` and model`llama.cpp:` plus the exact model suffix returned by setup's ready response. This preserves the qualified identifier that previously failed availability checks.

At **02:01:12.143**, real persistence returns200/saved for assistant `pa_ff38-deb2-150-069a`. The new helpers/captured action code contain no model picker, Model Settings reselection, selected-model setter or storage mutation. Together with root's action record and the unchanged model suffix, this supports the no-reselection requirement. A friendly model label alone is not used as proof of raw stored identity.

## Canonical reload

Normal reload completes at **02:04:22** and retains the same saved Character conversation and healthy qualified model UI. The actual canonical GET200 contains exactly two rows:

- User `08ee15e2-76d7-4341-b2f6-2780dbf1c7c9`: exact original question.
- Assistant `pa_ff38-deb2-150-069a`:8,697 characters, SHA-256 `b74bf9b6966ee16b5191b1654888f20b107d74a71000cfd86a62e441d9a00381`. Removing think blocks yields exactly **`BEEP BOOP.`**,10 characters.

The independent extraction agrees with the root's safe canonical projection. Saved state and the final answer are visible after reload. Raw provider reasoning stays local; the audit retains only record identity, length, hash and exact-final checks.

## Provenance and limits

The fresh profile, initialization, official fixture holder and original API74305/Next74329 launch receipts still match their earlier independently reviewed bytes. Their intervals cover the resumed first-chat step through canonical reload. Source manifest remainsa7; provider normalization, availability and Models-owner source files equal both the prepared manifest and independently reviewed implementation snapshots. This reviewer reused those original process/fixture inspections and checked immutable hashes, without new live-process or DB probes.

This is a fresh application profile with reused isolated dependencies, resumed from its prior wizard checkpoint. It is not a clean OS installation or an uninterrupted same-minute setup. Native scope is this PostgreSQL single-user sequence; existing controlled tests supply alias and wrong-provider negative coverage. No full48, native multi-user/SQLite or general model-reliability claim is made.

This exact workflow-named E2E-TestBot differs from the earlier tagged UAT261 character. Its successful initial canonical pair does not retroactively erase those failures or alter their classification. Earlier partial236 evidence remains preserved; this later sequence supplies the missing criterion.

Verification: `node .tmp/uat-repairs-231-246/native236-fresh-review/audit.mjs` completed **21 passed / zero failed**. No product/test/task/Git/browser/runtime/DB/model mutation or new inference was performed by the reviewer. Only this safe review packet was written.
