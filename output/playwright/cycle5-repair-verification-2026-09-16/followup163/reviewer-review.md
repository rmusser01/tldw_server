# UAT163 independent review

Task TASK-13260.100. **Clear bounded code review: no material correctness defect found in the frozen three-file production change.** Independent targeted tests and both causal baseline replays completed. Native cold reload/scoped-send acceptance remains parent-owned and pending as of this review.

## Frozen identity

Author manifest SHA256: `6dd6a048814ed7feb0037b6a3ae50d7a518ffc08b8d7dbddbc50b779fc8d9c2a`.

Independently verified all14 manifest entries and all five source/test snapshot copies. Source/test hashes were checked again after verification and remained unchanged. Baseline copies were independently compared with the corresponding files from commit `223591ac4fb2520290634c74310351ce7d8b18ae` and match byte-for-byte. Receipts: reviewer-source-verification.json and reviewer-baseline-verification.json.

| Owned file | SHA256 |
| --- | --- |
| store/playground-session.tsx | `64db2ac276ecb6358eaa8197858869f7bd99573eee1079e4c9e49f0843c5cfa4` |
| hooks/usePlaygroundSessionPersistence.tsx | `f098d478bf69271fa92230a7a89f632aa5d36d89e3552808e1f42e5cc4db0bd7` |
| components/Option/Playground/PlaygroundForm.tsx | `a656a860faa08d8591f89b05c0fb144d748d35a86ae3c8ca08d57d26a950410a` |
| components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx | `d259d75252b94caf1f0e7b918fb7137c8052a130b4117d0ff34f40fa6a7c74d3` |
| hooks/__tests__/usePlaygroundSessionPersistence.test.tsx | `441431f82a13f752d7bd35941f752138bc5e75e3f43e53bdff0c995af99d658d` |

Paths above are relative to apps/packages/ui/src/.

## Independent verification

- **GREEN: 57 tests / four suites passed**, exit0, no skips, 5.65s. Command below; receipt reviewer-green.log.
- **Cold baseline RED:** three production modules supplied from verified baseline copies through a private Vite load hook, leaving working source untouched. Filter `cold`: **two failed, four passed, 31 filtered**. Both enabled cold-rehydrate cases fail because the real Send does not issue RAG; explicit false, missing legacy flag, changed account and changed server controls pass. Receipt reviewer-red-cold.log.
- **Toggle baseline RED:** only the Form is supplied from its baseline copy, retaining the current store and persistence hook. Filter `newer explicit retrieval toggle`: **two failed, 35 filtered**. The before-restore and during-restore explicit-disable tests both wrongly issue RAG. Receipt reviewer-red-toggle.log. Reverting all three files for this second check would hide this newly reachable race behind the original missing-persistence bug; the partial replay isolates the Form guard's purpose.
- **Fresh scoped ESLint: exit0, zero errors, 91 warnings** across all five owned files. Every rule/message/severity matches the retained pre-change baseline exactly; reviewer-eslint-comparison.json. No new lint diagnostics. The initial repo-root bunx launch failed before lint with temp-directory access denial; invoking the already installed ESLint directly with node succeeded. Both attempt receipts are preserved.
- Full compiler was not independently rerun. Author retains90 prior/current diagnostics with no added signatures, not a clean typecheck. Author Bandit cannot parse the three TSX files and gives no TypeScript security assurance; no Python source changed.

Successful targeted command, from apps/tldw-frontend:

```sh
/Users/macbook-dev/.bun/bin/bunx vitest run --config ../../.tmp/uat163-repair-20260916/vitest.config.ts ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx ../packages/ui/src/hooks/__tests__/usePlaygroundSessionPersistence.test.tsx ../packages/ui/src/store/__tests__/playground-session-store.test.ts ../packages/ui/src/hooks/__tests__/playground-session-restore.test.ts
```

Baseline commands use the same executable and Form test with `--config ../../.tmp/uat163-repair-20260916/reviewer-baseline.config.ts -t cold`, or `UAT163_REVIEW_BASELINE=toggle` with `-t 'newer explicit retrieval toggle'`. The private loader changes only which source bytes Vitest compiles; no source/test replacement on disk is performed. Existing private config aliases the already installed pa-tesseract.js package and does not install dependencies.

## Contract review

1. **Persistence completeness:** the boolean is included in the existing type/default, partialization, message-store subscription, snapshot and dependency list. The real unmount flush serializes it, and actual localStorage rehydrate restores it before the next Form action. The test does not replay the Home handoff after rehydrate. Both the initial selected-source session and a session following a controlled retrieval failure reach the real action/RAG serializer with the original media IDs and returned Rowan facts.
2. **Legacy and explicit disable:** default false plus `sessionStore.fileRetrievalEnabled === true` means absence and explicit false do not enable retrieval. The change does not infer activation from rag mode or IDs. Strict equality also rejects truthy nonboolean values at replay.
3. **Owner/server scope:** the existing validity check runs before settings replay. The cold account/server controls reject old conversation/source activation. The new field creates no alternate restore path or scope bypass.
4. **Newer source intent:** replay of mode, activation and media IDs remains inside the existing sourceSelectionRevision guard. The Form's explicit Knowledge toggle advances that same revision before updating the option flag. Thus false→true→false remains a newer choice even though its final value equals the cold default. Existing accepted handoff, same-value handoff, ABA, delayed restore and later deliberately requested restore controls remain green. Raw setters and persistence replay do not manufacture user intent.
5. **Conversation switching:** useSelectServerChat continues to cancel pending restore and clear selectedKnowledge/ragMediaIds. The strengthened actual hook integration confirms the pending saved source does not overwrite a newly selected chat. useClearChat still clears activation and IDs. A normal switch need not reset the activation flag itself for safety: shouldUseRagForTurn requires both that flag and nonempty selected media IDs (or separately selectedKnowledge), and the deliberate switch clears the selected sources. The repair does not restore an older conversation's sources after the cancellation guard.
6. **Actual wrapper review:** the source-flow fixture mounts the real Form callback, session hook/store, action router and RAG request serializer. The full useMessageOption wrapper and full KnowledgePanel are not mounted; their current production forwarding paths were inspected. The wrapper's activation handler directly invokes the store setter, and KnowledgeSection forwards the same callback into KnowledgePanel. No additional wrapper behavior contradicts the tested boundary.

## Limits and disposition

DB, network/model transport and unrelated UI are mocked in the source-flow regression; it proves routing/state contracts rather than live provider output. The prior-retrieval-failure case exercises activation persistence after failure but does not replace UAT103's separate canonical/local-history tests. The run is not a real browser cold reload, and it does not establish a cause for the earlier UAT013 wrong answer despite supplied facts.

Independent code/test gate is clear. Root should retain actual Home source selection → cold reload without another handoff → scoped RAG request and answer before declaring TASK-13260.100 native acceptance complete. No production/test, browser, service, model, task-record, staging or commit mutation was performed by this review. Only reviewer-prefixed private evidence/config files were created under the assigned packet.
