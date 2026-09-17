# Independent native audit — UAT193 and UAT212

## Disposition

- **UAT212 / TASK13260.150: native acceptance passes.** Recommend closure together with the source review and automated validation already recorded in the task. The actual one-result live region says **“1 character found.”** This audit does not claim native zero- or multiple-result exercises.
- **UAT193 / TASK13260.131: fresh chat creation and controlled failure → Retry → reload pass; task remains pending.** Literal AC3 also requires greeting save actions. The supplied fresh Alice chat evidence contains no greeting Save Note or Save Flashcard requests. Earlier Bob saves belong to a separate conversation and do not satisfy that remaining step.

Only the 18 named native inputs and two official CLI task snapshots in `input-manifest.json` were used. `audit_inputs.py` verifies the principal IDs, statuses, result count, exact announcement, and six reload histories; it completed successfully. No product, browser, runtime, task or git mutations were performed.

## UAT193 evidence

All times below are UTC on 2026-09-17. Safe identity events establish Alice, user 2. The visible character query returned one Helpful AI Assistant, ID 4. Public character responses omit `client_id`; this audit does not infer that field from the response.

| Boundary | Retained observation |
| --- | --- |
| Fresh conversation | POST `/chats/` with character `4`, state `in-progress`, source `webui-character-chat`, scope `global` returned **201 at 06:47:02.505**. Conversation: `471f52fa-22c5-4eba-a3d6-ee5e6982884f`. |
| Original greeting | POST messages returned **201 at 06:47:02.649**; greeting ID `1e23f18b-4842-4799-931f-df2b2ea75448`. |
| User turn | POST messages returned **201 at 06:47:02.710**; user ID `cb4a633c-3ef2-48bb-8658-d48702fa1ed6`. |
| Controlled negative | The installed one-shot route continued the real request after changing only provider/model to `ollama` / `uat031-unavailable`. The real backend returned **400 at 06:47:02.856**, `model_not_available`, naming those overridden values. |
| Visible recovery | The error snapshot exposes **Retry same model**. `alice-controlled-provider-retry.txt` records the actual button click. |
| Retry completion | Same conversation, original configured `llama`/Gemma model, completion **200 at 06:47:54.834**; response body reading ended at 06:48:03.521. |
| Persistence | Real `/completions/persist` returned **200 at 06:48:03.586**, `saved: true`, assistant ID `pa_5455-6f36-098-dd2b`. Settled UI contains “ALICE CHARACTER RETRY VERIFIED.” |
| Full reload | `alice-controlled-retry-reload.txt` records `page.reload()`. Six history responses from **07:18:10.220 through 07:18:10.748**, all 200, contain the same ordered three canonical IDs exactly once. Settled UI still displays the user prompt and answer. |

The six reload responses preserve these canonical rows:

| ID | Sender | Timestamp |
| --- | --- | --- |
| `1e23f18b-4842-4799-931f-df2b2ea75448` | assistant | 06:47:02.629 |
| `cb4a633c-3ef2-48bb-8658-d48702fa1ed6` | user | 06:47:02.701 |
| `pa_5455-6f36-098-dd2b` | Helpful AI Assistant | 06:48:03.573 |

The cumulative request observer sees the original pre-route provider/model. The separate one-shot fault receipt and real 400 response establish the effective override; this audit does not label that observer body as the altered wire body. This is an actual backend model-availability **400**, not an upstream 502 and not proof of a dispatched Ollama generation. The successful streaming response has no separately captured raw SSE body; persistence and canonical history provide the completion evidence. No extra chat or user message was created by Retry in these receipts.

### Remaining UAT193 step

AC3 says: “Native fresh character Chat creates its conversation, then controlled failed provider Retry and greeting save actions complete successfully.” Complete the ordinary greeting Save Note and Save Flashcard actions for greeting `1e23f18b-4842-4799-931f-df2b2ea75448` in this conversation and retain their successful persistence evidence. The existing automated ownership/migration review recorded under AC1/AC2 is outside this native-only audit; it was not rerun.

## UAT212 evidence

- Earlier `alice-reviewed-characters.txt` retains the causal native wording **“1 characters found.”**
- In `character212-events.txt`, the actual character query returned **200 at 07:19:31.039**, `total: 1`, sole character ID `4`. Alice identity 2 is captured again.
- `character212-settled.txt` contains the live **status** text **“1 character found.”** The corresponding retained event capture is **07:20:05.983** on `/characters`.
- This satisfies the remaining native one-result clause of AC3. The task records the prior independent 18-test review and the ICU resource/fallback implementation; this audit does not replace those source and zero/multiple-result checks.

## Attribution and limits

Parent handoff attributes this run to backend `47e23`, PID 56113, and frontend repairs `4c233fcaa7` (UAT211) / `938aab72d6` (UAT212). These 18 receipts do not independently establish a startup source manifest; that attribution is supplied by the runtime owner. No stale a130 manifest is used. This is a bounded Alice native check, not the complete tenant/auth/provider matrix or a clean-console certification. Other separately tracked Chat issues are neither erased nor certified by this result.

Original input bytes are unchanged. The manifest records original SHA-256 values without normalization or copying native material. Reduced facts exclude model reasoning text and authorization/session data. No credential or private helper files were read.
