# UAT236 — SQLite single Character readiness diagnosis

Task association supplied by root: TASK-13260.178. Frozen source revision `8f8774e6c868b304a96d95ab82e28389c129a78b`, run `fresh-final-20260917`.

## Disposition

A product-side model-identity mismatch is supported by the native observations and frozen data flow. No additional operator provider configuration is indicated. The ordinary provider works, Settings reports one usable model, and selecting the actual Chat catalog option still leads to the library blocker. Preserve the three unsuccessful attempts; no further retry was performed by this reviewer.

There are two concrete identity failure branches involving the same missing `llama` normalization. The third attempt must not be attributed solely to the original `llama.cpp` selection. A separate Settings selection-ownership concern is source-supported and historically known, but is not independently established as another native finding here.

## Native sequence

All times 2026-09-17 UTC:

- `pirate-chat-result.txt`: ordinary `/api/v1/chat/completions` request 13:28:53.957 uses `api_provider: llama` and the raw configured Gemma GGUF path; response 13:28:54.469 is HTTP 200 and the visible answer includes pirate speech. The active-context display initially shows `Custom / tldw:llama.cpp:<path>`.
- TestBot creation returns HTTP 201 at 13:32:06.858, character id 4/version 1. It uses the manually entered TestBot system prompt. The generator model label is not evidence of a separate generation request.
- First library Chat opens “Choose an available chat model…”; recovery capture 13:33:11.626 retains that dialog. `/settings/model` displays the qualified `tldw:llama.cpp:<path>` saved selection and **1 usable** model.
- Selecting the visible LLaMa.cpp Settings option does not remove the blocker; second library attempt is captured at 13:35:41.045.
- `testbot-chat-recovery-final.txt` at 13:37:53.707 shows the actual Chat picker recovery rendered **LLaMa.cpp / <raw path>**, followed by a third blocked library action. Final capture 13:38:32.897 retains the same dialog. Its observer contains no `/chats` creation after 13:32. There is no TestBot inference failure: the library stops before navigation/creation.

The model path is `../../../Working/Language_Models/gemma-4-26B-A4B/gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf`. It is a model identifier, not a file read by this audit.

## Exact source boundary

Source paths below are relative to `sources/sqlite-single/apps/packages/ui/src` unless otherwise stated.

1. **Library gate:** `components/Option/Characters/Manager.tsx:247–254,870–871` reads `selectedModel` through storage and supplies `fetchChatModels` results. `hooks/useCharacterCrud.tsx:526–548` calls `buildCharacterChatReadiness` and returns before clearing/navigating when the chat model is blocked.
2. **Catalog identity:** backend `llm_providers.py:1617–1623,2460–2477` uses provider key `llama`. `services/tldw/model-normalization.ts:54–70,91–94`, `TldwModels.ts:566–583`, and `tldw-server.ts:95–100` retain the raw model identifier and provider, producing descriptor `{id: <path>, model: tldw:<path>, provider: llama}`. The backend provider key also matches the observed successful ordinary request.
3. **Original setup-selected branch:** setup publishes `tldw:<qualified provider>:<model>` (`UnifiedSetupWizard.tsx:515–531`); the displayed value is `tldw:llama.cpp:<path>`. Readiness normalizes that selection to provider `llama.cpp`, but `utils/chat-model-availability.ts:216–237` does not map descriptor provider `llama` to it. The exact-provider comparison at 640–659 fails; the same-base/provider-present conflict at 662–667 rejects the descriptor rather than accepting it.
4. **Actual Chat-picker branch:** `hooks/playground/useModelSelector.tsx:463–466,566–569` saves `getCanonicalModelKey(model)`. `modelSelectorUtils.ts:40–45,95–124` preserves provider `llama` and constructs **`llama:<path>`**. The picker display resolves metadata and prints its nickname/raw path (`useModelSelector.tsx:87–109,168–199`); the displayed raw path therefore does not mean the saved key is unqualified. Both `resolve-api-provider.ts:23–73,105–155` and readiness's known-provider set omit `llama`. Consequently `llama:<path>` is treated as a whole base identifier, which cannot match descriptor base `<path>`. This accounts for the third attempt without requiring a storage race.
5. **Not path punctuation:** these parsing functions inspect the first colon, not `../`. A bare `<path>` or `tldw:<path>` can match the catalog descriptor; the unrecognized `llama:` prefix is the problem. An existing related normalizer already equates `llama.cpp`, `llama-cpp`, `llama_cpp`, and `llamacpp` with `llama` (`model-provider-availability.ts:45–48`), demonstrating inconsistent identity conventions between layers.

This is a static trace through frozen functions, not an executed private reproduction. The supplied observers do not contain the library's in-memory query DTO or stored key, so those runtime values are inferred from the actual picker/mapper code and rendered evidence; no cache/storage/credentials were read.

## Settings recovery concern and historical mapping

`components/Option/Models/index.tsx:108–110,326–329` changes only storage. `useSelectedModel.ts:68–94` prioritizes the Zustand owner and writes its existing value back when storage differs; Layout mounts that owner through `useMessageOption` (`Layouts/Layout.tsx:167`). The post-click Settings snapshot still displays the old qualified value. This is consistent with, but does not separately prove, the exact historical **UAT115 / TASK-13260.55** mechanism repaired for AnalysisModal. Keep it as a required recovery regression within UAT236 until causally verified; do not invent UAT237.

**UAT106 / TASK-13260.47** introduced the verified setup selection handoff. **TASK-12416** covers Character model-usability/send gating. These are relevant predecessors, not evidence that this exact two-branch `llama` mismatch was previously accepted or tracked. No exact historical matching UAT was found in the bounded inspected records.

## Required future controls / bounded recommendation

Repair provider identity consistently across qualified selection parsing and descriptor matching while preserving wrong-provider conflicts, unconfigured/catalog-only rejection and explicit selection. Do not bypass readiness or choose the first catalog entry.

Before editing, reproduce through actual mapper + readiness using (a) verified `tldw:llama.cpp:<path>`, (b) actual picker-produced `llama:<path>`, and (c) bare/tldw raw-path positive controls. Include colons inside local model tags and conflicting real provider/model pairs. Add the actual Settings component with a mounted consolidated selection owner, then navigate to the Character library to prove the advertised recovery persists. Existing alias tests cover `llama_cpp` versus `llama-cpp` but omit backend `llama`; selector tests and Manager tests do not connect these real boundaries. ModelsBody tests mock storage setters and cannot prove shared-owner persistence.

## Limits and hashes

Only this ignored report was written. No product/test/task/config edits, tests, browser actions, requests, inference, services or DB access. This is diagnosis, not Character acceptance or repair verification. All 23 source/history files below match the original archive manifest.

### Inputs

| Packet-relative input | SHA-256 |
| --- | --- |
| `copy-preparation/sqlite-single-archive-manifest.json` | `26255fe54e27f7e92d849bbf810a7224c655602e3e2f9514eb6cbcce3c96bba1` |
| `native/sqlite-single/testbot-create.txt` | `082282b096cb19f13931d45cb6c28416c2795a2dafd171982389aa4e68db5538` |
| `native/sqlite-single/testbot-library-chat.txt` | `32ac102da8ca3ad314baae6adcf599ba087822ff0e91056cb3a7187e1ec8a548` |
| `native/sqlite-single/testbot-availability-recovery.txt` | `fcd77089efaa800cea3d51afe4b71cac4a94043d5253d01fd38c682a8d26e3d3` |
| `native/sqlite-single/testbot-model-settings-snapshot.txt` | `6d4366778ca472b365e3bec20456c3846723bc9f9ed17b175e7b8ea5356274c3` |
| `native/sqlite-single/pirate-chat-result.txt` | `27b982a6baed0e5fce330534a7529d7b0c5cd71f7192a019558f06e9780c6549` |
| `native/sqlite-single/testbot-model-reselected.txt` | `b0dc413dcc29931eb55995a1e4bf0f272ec56b394b3f6168c96c4f7bcfe2a112` |
| `native/sqlite-single/testbot-model-return.txt` | `01d3ffd75cb9241cf77bcde3a6c05dc5f36e780f257197a8201666ba4afc0d60` |
| `native/sqlite-single/testbot-retry-chat.txt` | `0e3020de0d759e7ac56aed056b965c2cd261e875ff008b5b13b8032ab4dc99dc` |
| `native/sqlite-single/testbot-chat-recovery-final.txt` | `276be75cb2677a32cd2829ed0bd43096cb585dccf5230357d795437501ce44e8` |
| `native/sqlite-single/testbot-three-attempts-final.txt` | `4b1b2ef3f8637372ec74cb351d73c4420c32f7dabb5b26f4aad0cedd5470908c` |

### Frozen source / history

| Frozen-root-relative path | SHA-256 |
| --- | --- |
| `apps/packages/ui/src/components/Option/Characters/Manager.tsx` | `7ac955754373a77ef432f0f656c85f7dec7a757c060c726681e29ba6afd7e341` |
| `apps/packages/ui/src/components/Option/Characters/hooks/useCharacterCrud.tsx` | `fcf0c9e164620df8391fa051faedff1e5b6b8f152887322d1dac27924a2e12ad` |
| `apps/packages/ui/src/utils/chat-model-availability.ts` | `7b2f6c28f27fe893838be48fd27fb110f1c6d4dc9f439c09f2d2f45b4678ace9` |
| `apps/packages/ui/src/utils/resolve-api-provider.ts` | `7d37b45a69a35ec41dfeb27eb45efedb0a2089df4c451db73906c93468cbb0bb` |
| `apps/packages/ui/src/services/tldw-server.ts` | `db8bf53c0afe32236af84cbfec37cbe5effb8f97881d8a4bbfdc1b31b9bf0aa6` |
| `apps/packages/ui/src/services/tldw/model-normalization.ts` | `573219e3f80cf4895ec76deb35eb1ef8fd05d61f6f93949dc67ad4405c31aeb9` |
| `apps/packages/ui/src/services/tldw/TldwModels.ts` | `ab26465ab068b6f65e4f54aacb48adb80e840d336aebabb144817a974906a537` |
| `apps/packages/ui/src/services/tldw/model-provider-availability.ts` | `26874b3fc7d659aff79707b2ffa675018a2c758db121c9884cb65f4cc3126c44` |
| `apps/packages/ui/src/hooks/playground/useModelSelector.tsx` | `6d7319fac1580944333f32439862e37fe18e452f7e6ef60dc27bb4420b83840a` |
| `apps/packages/ui/src/hooks/playground/modelSelectorUtils.ts` | `3bce44d1048612e2132df2382c22806199d3b66fca856f43c8dbcb5a09e122bc` |
| `apps/packages/ui/src/hooks/chat/useSelectedModel.ts` | `e528bbb7d2c1375041dda6e2555663c54d6e73b08ed18e91f7c627ca044f1c60` |
| `apps/packages/ui/src/components/Option/Models/index.tsx` | `21142d835bbc9cd687ce7fe6b70cf68f3debf337581052866ef3d0013f36fcdc` |
| `apps/packages/ui/src/components/Layouts/Layout.tsx` | `4d840d418eaeff05d04a2e77850eb98419dd57314c7eaa2058dc71b8b6fe141c` |
| `apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx` | `5e506e67e9d94b778bcad11ba13cd45f6724e107bfd48e90d2a88c5cac2ffb5b` |
| `apps/packages/ui/src/utils/__tests__/chat-model-availability.test.ts` | `150b18c94f4d5bd48c7d5b7b93193443fa4a7e20ef6b8f387644cccf33bbd315` |
| `apps/packages/ui/src/utils/__tests__/resolve-api-provider.test.ts` | `a52f67ce2f2ce7c6ba955f0b9111210a36922726ba08e66a1579f2a45ef4ebe1` |
| `apps/packages/ui/src/components/Option/Characters/__tests__/Manager.first-use.test.tsx` | `4efec5073920fde19b9156252d725d3d4f45dbe878cea3777345a950a32f7354` |
| `apps/packages/ui/src/hooks/playground/__tests__/useModelSelector.capabilities.test.tsx` | `144f0ee6fc3a6b80f66cf26bb626923d50b6c50ae99287a812b687b855fe837f` |
| `apps/packages/ui/src/components/Option/Models/__tests__/ModelsBody.test.tsx` | `50a5b1ce4c213ddb79c43b1a1c58101bd640fedc97c4d189c3fedd63a40b4789` |
| `tldw_Server_API/app/api/v1/endpoints/llm_providers.py` | `716c9177f435e925abff06e07dc2da4e3d89b2987ccc0a21259ce9885c80da46` |
| `backlog/tasks/task-13260.55 - Preserve-the-model-explicitly-selected-for-Media-analysis.md` | `4ef534fce9e6538805108956b05ec473c7431c542cf9d98978f53fc625ac321d` |
| `backlog/tasks/task-13260.47 - Carry-the-validated-setup-model-into-fresh-regular-Chat.md` | `74c53e3e7eaa6d5515e7639a0efb5715ed29db73ef0b9c0fb121db3d1f7f0cd5` |
| `backlog/tasks/task-12416 - Implement-Character-Chat-Phase-7-model-usability-and-send-gating.md` | `89c901ecdc68a8636eb8f6ec82ae810d3011daa90791e6639df6a34b8d6f504c` |
