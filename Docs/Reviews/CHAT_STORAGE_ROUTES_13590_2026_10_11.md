# Production storage reach and scope boundaries — TASK-13590

The normal full app omitted the existing storage router while minimal tests registered it. The full content catalog now includes it once under the original `/api/v1/storage` prefix and the existing storage route policy. No handler, dependency library, option or persistence layer is replaced.

Independent review found two security consequences of activating these existing routes: read-only keys could delete an owned file, and write-only keys could read private metadata. All seven user mutations now use the shared write-scope dependency; all eight user GETs use the shared read-scope dependency. File ownership, admin quota authorization and documented JWT/single-user/admin bypass semantics remain the house rules.

Causal regressions prove the original full-app authenticated404 and missing catalog spec, read-only deletion200 with a deleted row and L1→0, and write-only metadata200. The final real normal multi-user SQLite app uses actual API keys and canonical RBAC membership, rather than overridden authentication or a scalar admin label. It proves L1MB/N2MB/total3MB/available7MB, matching breakdown and admin components; anonymous401, foreign-file403 and nonadmin quota403; all eight write-only GET403 and all seven read-only mutation403; successful read-key metadata/usage and write-key deletion leaving N2MB charged. Actual private config disabling and idempotent registration remain verified.

The fixture closes real audit services through the documented lifecycle helper. Initial fixture TypeError and unclosed-audit-worker timeout are excluded diagnostics, not acceptance evidence. Supported Python3.12 final seven-file connected run:322 passed,192 warnings, no skips,64.03s. Two independent final source reviews are bounded CLEAR.

Canonical generation and full-app fingerprint/path checks exit0. The shared contract adds15 paths and25 schemas:2129 paths/3301 schemas, SHA256 `3b8ce49f0c3f97a0fc22ebe66df5599f528d431148096c500ae36a64ae4d2691`. Both normal WebUI and extension TypeScript configurations exit0. Scoped Ruff baseline1/current1 introduces zero findings; the inherited folder exception-chain warning is unchanged. Six-production-file Bandit has zero findings/errors. Supported Python compilation passes. Black formatting of the new regression preserves its exact AST.

The existing normal full private API on19767 is refreshed on supported3.12 with the same profile/configuration and no auth reinitialization. Usage and breakdown return200 with explicit legacy/native components combined once. Original9 conversations/28 messages remain deep-equal before/after against the immutable baseline. No browser is opened, and ambient19764 is untouched.

ADR required: no. Existing grouped registration policy and [ADR-069](../ADR/069-frontend-api-contract-package.md) govern the repair and canonical artifacts. TASK-13589 owns retained accounting; this closes its remaining public route/contract gate after normal hooks and the related commit.

Evidence: `/private/tmp/chat-storage-route-13590-*` causal/connected/API/type/static logs and safe runtime proof. Runtime credentials, startup diagnostics and canonical chat content are not copied into this record. Current fetched origin/dev `19d213e936da37a908b30e79d1e4ff7a4c3839d1` remains an ancestor of this branch.

Public H2 remains disabled. MIME/representation admission, adoption, typed reader/send, frozen character behavior, original D7/Sync recovery, complete persona/surface/theme/scale/package coverage, physical PostgreSQL/AT and participant studies remain separate open gates. No physical download or installed-extension acceptance is inferred. Normal exact-owned hooks and bounded completion recording follow.

Bounded completion: all 18 exact-owned normal repository hooks passed without bypass. TASK-13589 and TASK-13590 acceptance criteria and definitions of done are complete. Only their two owned completed plans retire to private evidence; the tracked accounting-plan removal is part of this unit. Final metadata hooks and the incremental local commit follow. Master criteria and the explicitly listed broader acceptance gates remain open.
