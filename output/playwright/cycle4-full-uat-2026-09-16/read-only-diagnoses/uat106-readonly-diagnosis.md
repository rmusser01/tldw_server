# UAT106 / TASK13260.47 — setup selection is not handed to regular Chat

## Finding

Confirmed missing frontend preference handoff. Unified setup verifies its own explicit provider/model pair, but its successful completion does not initialize regular Chat's separate selected-model preference. For a fresh browser, Chat then auto-selects the first catalog item. This reproduces with a catalog that already includes the validated custom model, so stale inventory UAT107 is not necessary for this failure.

No product/test/browser/runtime changes or provider requests were made. Only private probes/reports and official task notes were written. All 15 inspected production/test files match frozen product `7c9409`; `/private/tmp/uat106-frozen-source-comparison.json` records exact hashes.

## Supplied native evidence

- `/private/tmp/uat-cycle4-single-model-validated.txt` and `provider-saved.txt` (same prefix): normal Validate and Save providers actions. These two files are action records with snapshot links, not standalone API validation-response captures.
- `/private/tmp/uat-cycle4-single-first-chat-response.txt`: first-run verification actually returned 200/ready for `custom_openai`, the configured Gemma GGUF model, and `Hello!`.
- `/private/tmp/uat-cycle4-single-normal-chat-start.txt`: first ordinary Chat displays `Ollama / gemma3:1b` and `Healthy`.
- `/private/tmp/uat-cycle4-single-default-chat-result.txt`: failed turn identifies `tldw:gemma3:1b` and retains Ollama selection.
- `/private/tmp/uat-cycle4-single-default-chat-502.txt`: sanitized failure detail is `provider_unavailable` / `The chat service provider is currently unavailable.` The filename and parent native report supply HTTP502; this body-only file does not independently encode the HTTP status.

## Causal chain

All paths below are relative to `/Users/macbook-dev/Documents/GitHub/tldw_server2`.

1. `apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx:71–90,139–142` reads provider/default_model from first-run state into a wizard-local pair. `:569–576` passes that pair to FirstChatStep but only calls optional `onComplete()` when finished.
2. `components/Option/Onboarding/steps/FirstChatStep.tsx:184–244` sends the explicit provider/model to verification; a ready response marks a milestone, invokes setup completion and onComplete. It never writes regular Chat selection.
3. `hooks/useSetupOnboarding.ts:386–408` performs first-chat/complete API calls plus first-run state refresh. It also has no model preference publication.
4. The setup persistence is not a browser model preference: `tldw_Server_API/app/api/v1/endpoints/setup.py:872–892` writes provider configuration and `[API].default_api`. That does not populate frontend selectedModel. `routes/option-index.tsx:266–270` supplies only onStateChange; the explicit setup route `routes/option-setup.tsx:252–257` supplies a Home navigation callback, neither a model handoff.
5. `hooks/chat/useSelectedModel.ts:25–40` derives the regular Chat selection from the Zustand selectedModel or browser `selectedModel`; `store/option/slices/core-slice.ts:120–121` starts null. The setter updates both store and storage; the sync effect gives an existing nonempty store value precedence.
6. `components/Option/Playground/PlaygroundForm.tsx:1532–1552` asks `utils/model-startup-selection.ts:18–62` to choose a model after storage/favorite hydration. The resolver keeps an existing model, then tries a favorite, then returns the first nonblank model in catalog order. It takes no setup selection/default parameter. `services/tldw-server.ts:95–100,378–395` maps the aggregated model list into these Chat descriptors without choosing the setup default.

The production fallback therefore does not mean "first model successfully verified". It means "first catalog item". This accounts for Ollama being selected despite the custom model being visible in the picker.

## Why the badge said Healthy

`PlaygroundForm.tsx:1791–1802` derives Healthy from overall API connection readiness, not a successful inference for the selected provider/model. `ChatModelSelectorDropdown.tsx:55–63` uses that connection label unless a model-usability override is supplied. Thus the label is not evidence that Ollama's model endpoint was tested. Fixing model handoff addresses the wrong-model choice; if the label is changed, keep that a bounded truthful-status change rather than introducing automatic inference health calls.

## Private interaction evidence

Files:

- `/private/tmp/uat106-setup-model.config.ts`
- `/private/tmp/uat106-setup-model-cases.txt`
- `/private/tmp/uat106-setup-model-red.log`

Command from `apps/packages/ui`:

```sh
./node_modules/.bin/vitest run --config /private/tmp/uat106-setup-model.config.ts --testNamePattern UAT106 --maxWorkers=1 --no-file-parallelism
```

**1 RED, 1 positive control; 32 unrelated cases filtered.** Actual UnifiedSetupWizard/FirstChatStep handles the Send test chat action and successful completion, then mounts actual useSelectedModel with actual Next WebUI storage hooks. The real production startup resolver receives a catalog containing both first Ollama and second verified custom model.

- Fresh state: verified custom pair; selectedAfterWizard null; storedSelected undefined; startupFallback `tldw:gemma3:1b`. Expected validated custom target fails.
- Deliberate existing choice: selectedAfterWizard and storedSelected remain `tldw:deliberate-model`; startupFallback null. Positive control passes.

Limits: setup API/readiness and Home milestone identity are controlled mocks; the probe starts at first-chat with provider setup already persisted. It mounts the real shared selection consumer and calls the real resolver; it does not mount the entire PlaygroundForm, make a live completion call, or certify account transitions. No production source is transformed.

## Minimal repair contract

- Introduce one explicit successful-setup-to-Chat selection handoff. It must run only for a current successful verification/setup completion, before first-run parent publication can unmount the wizard. A post-unmount effect or Home onComplete callback is insufficient because Home currently provides no such callback and refreshParentState can publish completed setup first.
- Seed only a truly fresh/unselected Chat preference. Preserve an existing deliberate choice and a newer user selection made during verification/completion. Do not continuously force the server-wide setup default on every Chat/account mount.
- Publish through the consolidated Chat selection owner (store plus durable storage), not a bare storage-only write. The latter is the separately diagnosed UAT115 failure mode when a mounted nonempty store wins over storage.
- Resolve the verified provider/model to the existing catalog's canonical selectable identity, with existing provider normalization. Do not infer a provider from the model's human label or silently select a different provider when names collide. If the target cannot be resolved/loaded, surface selection required rather than silently fall back to an unrelated model. UAT107 remains a separate catalog-refresh defect.
- Capture current target/account and an operation/lifetime generation before awaits; only publish while that same authority and the same eligible selection are still current. Failed/skipped verification, replacement target, logout/unknown account, A→B→A, unmount, or a later deliberate selection must not publish an old setup choice into the new context. Reuse existing authority utilities; do not broaden auth semantics or read/print credentials.

Required regression coverage: actual setup completion → shared model owner → actual regular Chat startup; valid custom second in catalog; delayed preference hydration; existing and mid-flight deliberate selections; failure/skip; setup parent unmount ordering; same-model-name provider collision; absent catalog target; delayed completion after server/account replacement including A→B→A. After repair, native fresh setup → first ordinary Chat → reload should retain the validated target and succeed without manual model switching.

Implementation remains deferred under the frozen full-UAT matrix. No compiler/lint/Bandit or full acceptance claim is made for this read-only diagnosis.
