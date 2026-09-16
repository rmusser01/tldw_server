# UAT157 / TASK13260.95 implementation handoff

Base: `20942cd30fb19a3d55fc8ee542847ba9d5627567` (reviewed UAT103). The root explicitly leased the two shared paths and approved independent per-row ACK guards. Implementation is ready for review; no staging, commit, tracker update, runtime restart, browser action or model call by this agent.

## Result

Ordinary successful Chat fallback now skips a user POST when the current payload contains a non-empty canonical user ID, and independently skips an assistant POST when its canonical assistant ID exists. A user-only receipt therefore still allows an unsaved assistant to be persisted. An assistant-only receipt still allows an unsaved user. Existing capability, conversation/scope, regeneration/Continue, image-generation-event, and exact local RAG diagnostic guards are unchanged.

This fixes the demonstrated duplicate writes when Quickstart's fallback capability discovery says `hasChatSaveToDb:false` despite a completion having saved both rows and returned their IDs. The native cached capability receipt corroborates that deployed capability path. This change does not establish what ACKs the original native stream delivered, nor does it resolve persistence ambiguity if a server saves but its ACK is genuinely lost.

## Owned change

- `apps/packages/ui/src/hooks/chat/useChatActions.ts`:6 insertions/2 deletions, two independent trimmed-ID conditions plus one comment.
- `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx`:56 insertions/2 deletions. Seven actual-boundary cases are added to the existing fixture, and its capability mock resets to true for all earlier controls. Latest UAT103 cases are retained.

The tests use the real action hook, pipeline, human formatter, model factory/ChatTldw, success helper and saveMessage. Only downstream transport/server storage and Dexie I/O are substituted. They check current-stream ACKs reach local persistence before inspecting fallback writes; no loader supplies those IDs later. A valid small PNG exercises acknowledged image rows. Partial-ACK fixtures save only the acknowledged role, proving the other role is genuinely unsaved rather than merely missing an ACK.

## RED → GREEN

1. Against unchanged production at20942, the seven permanent cases produced **4 behavioral failures and3 passing controls**. Both ACKs with capability false/error duplicated both rows; each partial ACK duplicated its already-saved role. No-ACK ordinary fallback and capability=true controls passed. Receipt: `permanent-red.txt`.
2. Added only the two per-row guard conditions. Full saved-normal suite: **98 passed**, including all91 previous cases. Receipt: `green-saved-normal.txt`.
3. Combined affected control set: **219 passed across13 suites**. Covers saved-normal, Character, Persona, overlay, image-event sync, mirror/error guards, mounted server loader, persistence/error variants, abort lifecycle, actual ChatTldw metadata/image handling and canonical chronology. Receipt: `green-combined.txt`.

The historical diagnosis's private4RED/5PASS evidence and native receipts remain unchanged outside this implementation directory.

## Compiler, lint and security

- Full frontend compiler before and after: **90 diagnostics**, identical multisets after stripping only line/column positions. No added/removed diagnostics; both commands exit2 because of that baseline. `compiler-comparison.json` and complete logs retained.
- Root-invoked frontend ESLint config, both touched files: **0 errors/17 warnings before and after**, identical rule/severity/message/file multisets. `eslint-comparison.json` and complete JSON retained. Existing root Pages-directory config notice is unchanged.
- Owned `git diff --check`: clean.
- Bandit: not applicable; the entire owned change is TypeScript/TSX, with no Python/security configuration touched. No security bypasses, schema capability inflation or policy weakening.

The optional pa-tesseract.js dependency already exists in the Bun cache but lacks a workspace link. Private `vitest.config.ts` merges the real frontend config and aliases that installed package only, matching the existing UAT103 harness. No dependency installation or tracked test configuration change. All tests use normal mock transport; no live application/runtime environment is modified.

## Commands

From repo root:

```text
node apps/tldw-frontend/node_modules/vitest/vitest.mjs run --config .tmp/uat157-completion-ownership-20260916/implementation/vitest.config.ts apps/packages/ui/src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx -t UAT157
node apps/tldw-frontend/node_modules/vitest/vitest.mjs run --config .tmp/uat157-completion-ownership-20260916/implementation/vitest.config.ts apps/packages/ui/src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx
node apps/tldw-frontend/node_modules/eslint/bin/eslint.js --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/hooks/chat/useChatActions.ts apps/packages/ui/src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx --format json
```

Combined suite arguments are retained verbatim in `verification-commands.json`. From `apps/tldw-frontend`, before and after compiler command:

```text
node --max-old-space-size=8192 node_modules/typescript/bin/tsc --noEmit --incremental false --pretty false
```

## Acceptance still required

Root owns independent review and a fresh successful native image turn with both `/chat/completions` and `/chats/{id}/messages` network capture, followed by canonical listing/reload. The original duplicate data is preserved. This work does not close UAT013 wrong-answer acceptance, UAT103 native diagnostic acceptance, or any genuinely lost-ACK case. Normal legacy no-ACK fallback behavior, including its existing image omission/ignored returned IDs, is unchanged.

`owned.patch`, exact two-file review snapshots, and `owned-manifest.json` make this handoff reviewable even after later work continues.

Root review follow-up: inserted the missing newline between the describe callback and it.each. This is formatting only; no broad formatter rewrite. Root owns the independent219-scope rerun after this final change. Final snapshots/patch/hashes refreshed.
