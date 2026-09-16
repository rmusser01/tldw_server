# UAT130 / TASK13260.70 — Analyze initial model/catalog identity mismatch

## Confirmed cause

`AnalysisModal` compares two representations of the same model as literal strings. Setup persists `tldw:custom-openai-api:<exact Gemma path>`. The live catalog carries the bare model path and a separate provider field. The modal discards that field, makes its option `tldw:<path>`, fails the equality check, and silently selects the first catalog entry (`tldw:gemma3:1b`). The configured choice remains unchanged in shared state/storage; the modal's derived selection and outbound request diverge from it.

Relevant source, relative to `/Users/macbook-dev/Documents/GitHub/tldw_server2`:

- `apps/packages/ui/src/components/Media/AnalysisModal.tsx:147–158`: preserves qualified selection while loading, then exact `model.id === selectedModelKey` or `models[0].id`.
- Same file `:264–268`: maps catalog descriptors to only `{id,name}`, losing provider identity.
- Same file `:324–343`: generation uses that derived key and resolves its provider, so this affects dispatch, not only the displayed label.
- `apps/packages/ui/src/services/tldw/TldwModels.ts:425–431,580–607`: returns filtered ModelInfo while retaining bare `id`, separate `provider`, and optional `chatProvider`. The service does not promise provider-qualified IDs.
- `apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx:512–531`: intentionally publishes the exact verified provider/model as a qualified selection.

## Native observations

- `/private/tmp/cycle5-single-native-131-analysis-dialog.txt`: initial dialog shows qualified configured Gemma.
- `...-132-analysis-settings.txt` and `...-134-analysis-model-list.txt`: settled selection is gemma3:1b; catalog contains bare-ID gemma3:1b and exact Gemma path options.
- `...-135-analysis-model-chosen.txt`: user manually chooses Gemma.
- `...-138-media-requests.txt`: subsequent completion request2580 HTTP200 and version2583 HTTP201.
- `...-139-analysis-request.json`: manual-choice request uses exact bare Gemma path and `api_provider: custom-openai-api`.
- `...-140-analysis-version-request.json` and `...-141-analysis-saved-media.json`: grounded analysis and original source retained.
- `...-151-reanalysis-settings.txt`: root's subsequent reopen control retains the manual choice.

The native evidence establishes silent initial fallback and successful manual recovery. It does **not** establish a native wrong-provider dispatch from the initial fallback; that consequence is independently reproduced below with controlled transport.

## Private actual-component probe

No repository files were changed. A temporary Vite transform appends two cases to the existing mounted shared-owner test. It mounts the real AnalysisModal and useSelectedModel with real WebUI storage, delays catalog resolution, and keeps the actual provider resolver. Existing presentation mocks and controlled stream/version requests remain; no server or model is invoked.

Artifacts:

- `/private/tmp/cycle5-uat130-catalog-identity-probe.config.ts`
- `/private/tmp/cycle5-uat130-catalog-identity-probe.log`
- `/private/tmp/cycle5-uat130-source-hashes.txt`

From `apps/packages/ui`:

```sh
bun run test --config /private/tmp/cycle5-uat130-catalog-identity-probe.config.ts
```

**1 control passed, 1 regression failed, 1 existing case deselected**, 1.70s. The regression has three soft assertion failures (picker, provider, model), not three separate failing cases.

| Catalog representation; identical provider metadata | Hydrated picker | Controlled dispatch | Shared/persisted selection |
| --- | --- | --- | --- |
| Qualified IDs | configured Gemma | custom-openai-api / configured Gemma | unchanged Gemma |
| Bare IDs, as native | gemma3:1b | ollama / gemma3:1b | unchanged Gemma |

Before catalog resolution both cases display the configured qualified Gemma selection. This isolates identity projection from asynchronous storage ownership.

## Minimal post-freeze design

Expected owner: **AnalysisModal.tsx plus its model-owner/stage3 tests and task70**. Preserve TldwModels and the shared explicit selection owner unless a permanent regression proves another change necessary.

Retain provider metadata locally and resolve model identity as an exact model ID plus recognized provider, using the existing `parseProviderQualifiedModelSelection` normalization rules. Build a consistent option key/matched descriptor so a qualified saved choice matches an equivalent bare catalog ID with the same provider. Keep the raw model identifier intact for dispatch; never split arbitrary colon-delimited model names or filesystem paths as though every prefix were a provider. Continue writing explicit choices through `setSelectedModel`; catalog hydration must not publish an implicit selection write.

Existing useful patterns and limits:

1. `resolve-api-provider.ts:105–156` already recognizes known provider qualification and aliases, including a leading `tldw:`. Its selected-model provider resolver can preserve an explicit provider. Reuse it instead of a new parser.
2. `chat-model-availability.ts:609–675` already matches qualified selections against separate provider/base-ID descriptors and rejects a conflicting known provider. Its **unqualified branch chooses the first match**, so it is not by itself a safe ambiguity resolver for duplicate bare IDs.
3. `hooks/playground/modelSelectorUtils.ts:109–123` provides canonical keys, but blindly applying it to an already-qualified `id` can double-prefix the provider. Normalize before reuse; do not change that shared utility opportunistically.

Preserve the explicit compatibility test in `AnalysisModal.stage3.regression.test.tsx:259`: a genuinely removed **unqualified** persisted model falls back to an available catalog model. Alias-equivalent configured selections must be resolved before that fallback. A conflicting qualified provider or multiple provider candidates for an unqualified ID is different from a genuinely missing model; do not choose by array order or name. Keep the selection unresolved with visible reselection guidance/no dispatch in those cases. This adds no catalog bypass and does not invent provider identity from a model family or display name.

No general auth/catalog lifecycle rewrite is proposed. Preserve current close/cancel behavior, explicit-selection error feedback, stream fallback and version saving.

## Permanent regression plan

1. Convert the private delayed-catalog reproduction into a permanent real owner/storage component test: persisted qualified Gemma, Ollama-first bare catalog plus provider metadata, settled picker and exact outbound provider/model, unchanged stored selection. Keep already-qualified and native manual-choice controls.
2. Cover quoted persisted selections, recognized provider aliases, model IDs containing colons, and exact paths. Confirm no double prefix or inferred provider from an unknown prefix.
3. Same bare ID under two providers: explicit qualified selection resolves only its provider; unqualified ambiguity and explicit foreign-provider conflict do not dispatch via the first row. Unknown/providerless descriptors must not fabricate ownership.
4. Delayed catalog resolution after a newer explicit choice must retain the newer choice when present; failed persistence still uses existing visible feedback. Close/unmount must retain the existing cancelled-load behavior.
5. Preserve stage3 loading/no-selection/removed-model compatibility, actual explicit-owner115, cancellation, provider resolver and TldwModels suites. Use scoped lint/type baseline and independent review before native retry.
6. Native acceptance: first Analyze open immediately after verified setup, await catalog, verify unchanged configured identity without a manual repair, inspect exact completion payload and persisted analysis. Then explicit alternative selection/reopen remains stable. Existing manual success is a useful control, not acceptance of the first-open defect.

## Relation to UAT115 and evidence limits

Keep task70 distinct from completed UAT115/task55. That repair fixes direct storage writes being overwritten by the shared model owner. Its existing component test supplies provider-qualified catalog IDs, which avoid the present mismatch. This probe confirms the consolidated owner remains correct while the derived effective key changes.

This was read-only diagnosis plus an explicitly permitted private component probe; no production/test repository edits, live browser interaction, runtime writes or inference. No fix was applied or declared passing. All six retained SHA256 entries verified unchanged after the probe; AnalysisModal hash is `69512af1c5c3ad6ffd565e8c55ff185c342028bb6075cc1f4130d50cc71489b1` and full hashes are in the artifact above. Scope expansion beyond the local identity boundary should require a newly demonstrated regression.
