# UAT024 deterministic claim-verification negative control

Read-only preparation under parent UAT task. No live provider configuration, browser, server process or production file was changed. This fixture deliberately supplies a false generated claim; it measures rejection/error UX, not model quality.

## Prepared files

`provider-fixture/mock-config.json` is an ordinary configuration for the existing `mock_openai_server/run_server.py`. The model is explicitly named `uat024-claim-negative-control`, owned by `deterministic-uat-fixture`. Three ordinary OpenAI response files contain:

- Generator content: `[{"front":"What was the trial response rate?","back":"The trial response rate was 90%.","tags":["uat024-negative-control"]}]`.
- Verifier content: `{"label":"nei","confidence":0.1,"rationale":"Deterministic UAT negative control: source states 10%; generated card states 90%."}`.
- Unmatched content: a `fixture_error` object, making unexpected generation prompts unusable rather than returning a plausible default card.

Each file is a full chat.completion envelope whose assistant `content` is the JSON string above. No provider HTTP error is canned. Both intended provider replies are HTTP200.

Matching is constrained to the explicit model. The generator matches system substring `Generate ` and final user-message regex `^Generate flashcards from:[\s\S]*10%`. The higher-priority verifier matches system substring `You are a precise fact-checking judge. Output strict JSON only.` and final user regex `CLAIM:[\s\S]*90%[\s\S]*EVIDENCE:[\s\S]*10%`. The source text is `UAT024 deterministic fixture source: The trial response rate was 10%.`

## Validation already performed

`probe.py` mounted the **original generate_flashcards callable** on an isolated FastAPI route. The actual workflow adapter parses/normalizes the generated card; actual verification-unit construction, ClaimsEngine, numeric-precision check, result serialization and endpoint rejection run unchanged. Only the provider-call boundary uses the existing Mock OpenAI app's TestClient; usage persistence is disabled to avoid application database writes. `TEST_MODE` and `TLDW_TEST_MODE` are false.

`probe-result.json` records:

- Two provider calls matching `generator.json` then `verifier.json`.
- Actual handler HTTP422 with `detail.code = claim_verification_failed`.
- Artifact verdict `failed`, unit `flashcard:1:back`, status `numerical_error`.
- Generation/verification model and provider both explicitly identify the fixture.

The back is a complete factual sentence; the question front ends in `?`. The endpoint therefore creates one explicit semantic answer claim, with no separate extraction provider call. Low-confidence `nei` avoids a high-confidence LLM result overriding the backend's own numeric mismatch.

**Limits:** this component probe does not validate the native application, credential store, egress/socket transport, rate limiter or authentication middleware. An earlier attempt including full provider credential resolution stopped before dispatch with `credential_store_unavailable` in the intentionally database-free context. Matching the launcher's BYOK-disabled flag did not supply the absent credential store. The failed full-adapter receipt is retained separately; it is not acceptance evidence. Do not repeat that attempt or treat it as a live-profile defect. The current native profile has its own actual auth database and must validate the configured provider normally.

## Safe native setup for root to execute

1. Select the owned acceptance profile and preserve its exact private config/env bytes and SHA256 before any change. Also record the current Flashcards Provider/Model fields. Slot99 and port19124 are candidates; first confirm they are not already used. A read-only listener inventory found no listener on19124 at preparation time; it is not a reservation.
2. Launch only the deterministic fixture, with its own process receipt and private log:

   ```sh
   source .venv/bin/activate
   python mock_openai_server/run_server.py --config .tmp/uat024-native-20260916/provider-fixture/mock-config.json --host 127.0.0.1 --port 19124
   ```

   The runner prints a hardcoded localhost8080 introductory line; actual bind settings come from the explicit CLI and uvicorn. Verify `/health` and `/v1/models`; the latter must advertise the explicit fixture model. No real provider is called.
3. Configure the owned backend through its existing private profile `configPath`. The actual config loader reads numbered provider fields from **`[API]`**, not a new `[custom_openai_api_99]` section:

   ```ini
   [API]
   custom_openai99_api_ip = http://127.0.0.1:19124/v1
   custom_openai99_api_model = uat024-claim-negative-control
   custom_openai99_api_key =

   [Claims]
   CLAIMS_VERIFICATION_PROVIDER = custom-openai-api-99
   CLAIMS_VERIFICATION_MODEL = uat024-claim-negative-control
   ```

   Merge these keys into existing sections while preserving unrelated values; do not replace the file with this fragment. An equivalent normal process-env setup is `CUSTOM_OPENAI_API_IP_99`, `CUSTOM_OPENAI_API_MODEL_99`, `CLAIMS_VERIFICATION_PROVIDER`, and `CLAIMS_VERIFICATION_MODEL`. The mock requires no auth and custom slots do not require an API key.

   The existing `recovery-launcher.mjs` reconstructs runtime env on each launch, so editing `backend-env.private.json` alone does **not** change a restart. Prefer the actual private profile config path. Do not rerun `prepare`, which regenerates profile configuration. Root owns any needed stop/restart and must verify the old owned PID exited.
4. The visible GeneratePanel has ordinary free-text **Provider (optional)** and **Model (optional)** inputs. Use `custom-openai-api-99` and `uat024-claim-negative-control`; paste the exact fixture source above, request one basic card, and click Generate. The frontend does not expose/send claims-verifier overrides, so the private server `[Claims]` settings above are necessary to keep verification on the fixture even if a previous profile had a separate real verifier configured.
5. Capture the real `/api/v1/flashcards/generate` body/response, original source retention, loading settlement and visible error wording. Expect real backend422 with `claim_verification_failed`, `failed`, and `numerical_error`; native metadata should say verifier source `config`, whereas the prepared direct `request.json` uses explicit verifier overrides and the component probe therefore says `request`. Do not mock/fulfill the browser route, alter document APIs, or enable TEST_MODE.
6. Capture UX behavior before deciding acceptance. This is a diagnostic negative control, so raw JSON/error-code exposure may remain an actual UAT024 failure. It does not certify general generation quality. The current fixture logs no request bodies; root may retain the provider access log's two200 completions plus backend structured response. If provider payload proof is needed, use a dedicated request-only observer that excludes headers/credentials and records only this synthetic source.

## Restore

After retaining evidence, root should restore the exact original profile config/env bytes and hash, restart only that owned backend if it was restarted for the control, verify API/UI health and the original configured provider catalog without inference, restore the visible Provider/Model input values, and stop only the recorded fixture process. If a separate isolated profile was used, close it through the normal owned-profile lifecycle. Keep the fixture explicitly named in all evidence and exclude private logs, auth config, browser profiles and failed full-adapter logs from public retention.

## Source anchors

- `mock_openai_server/mock_openai/config.py`: ordinary model/system/content pattern matching and relative response directory resolution.
- `mock_openai_server/mock_openai/server.py`: config CLI; actual response endpoint.
- `tldw_Server_API/app/core/Workflows/adapters/content/generation.py:175`: generator prompt and JSON-array parsing.
- `tldw_Server_API/app/api/v1/endpoints/flashcards.py:175`: explicit answer verification units; `:2562` real generation endpoint and422 gate.
- `tldw_Server_API/app/core/Claims_Extraction/claims_engine.py:1123`: verifier prompt; `:1299` numeric check; `:479` numeric mismatch decision.
- `tldw_Server_API/app/core/config.py:4895`: `[API]` numbered provider config; `:1734` verifier configuration.
- `tldw_Server_API/app/core/custom_openai_providers.py`: actual provider99 names/env aliases, supported slots1–99.
- `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx:781`: actual visible provider/model fields.
