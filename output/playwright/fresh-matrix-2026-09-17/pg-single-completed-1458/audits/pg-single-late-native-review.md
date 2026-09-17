# Independent PG-single late-native evidence audit

Reviewed 2026-09-17T15:05:01.040617+00:00. Scope: matrix rows 2, 7, 8 and 9 only; 49 hash-bound inputs below. Frozen product revision is reported as `8f8774e6c868b304a96d95ab82e28389c129a78b` by the matrix/progress records. This audit did not re-hash the source archive, use the browser, run tests/inference, access credentials/private helpers/logs, or change runtime/product/tracker state.

## Disposition

| Row | Independent disposition | Boundary |
|---|---|---|
| 2 | Functional recovery supported, with qualifications | Disconnected Media credential gate, normal visible key reconnection, real backend unavailability and native Retry recovery. API-key mode has no JWT-expiry acceptance. Browser continuity was interrupted. |
| 7 | Pass for the named Pirate Prompt journey | Saved/synced prompt, actual system instruction and model request, ARRR response, canonical three-row reload. |
| 8 | Fail / downstream blocked | Character 3 was created with the exact instruction, but Character entry stopped at model readiness; no Character completion or BEEP BOOP response/reload. Two world-book catalogue 500s are independently visible. |
| 9 | Partial; re-rate preview failure reproduced | Real Pirate-answer Note/card reuse, backlink, one-card scheduling/practice and reload work. Stale re-rate preview is demonstrated. Media-grounded reuse, five-card/mixed/early-End Study are not established. |

The current row-level claims are supported within these limits. No new product finding is assigned by this audit. The possible Hard/lapse analytics interpretation remains with the separate diagnosis; observed metrics are not treated as a new defect here.

## Row 7 — exact prompt, real completion and reload

- `pirate-save-new-chat.txt` records native title/system-prompt fills and Save, followed too quickly by Back. The unsaved-navigation confirmation and failed snapshot are preserved in `pirate-editor-snapshot.txt`; the subsequent dismissal/save-state/library receipts show the prompt retained as Synced server #1, local edit ID `pa_6fe8-75fc-17c-9031`. The failure is not evidence of lost saved content.
- The saved 87-character system prompt is `You are a pirate. Respond to everything in pirate speak. Always say ARRR at least once.` Native Use in chat displays Use as System Instruction.
- Actual POST `/api/v1/chat/completions` at **14:32:20.977Z** contains that system instruction followed by `Tell me about the weather today.`, `api_provider=llama`, the configured Gemma GGUF model, `save_to_db=true`, conversation `392c8053-b7bc-45b2-8ef8-693b690b7939`, and client message `pa_e46f-57e1-be7-4abf`. Response status is **200 at 14:32:21.371Z**, body-read completion 14:32:26.970Z. This receipt records response status and persisted output, not a separate raw SSE transcript.
- The real `page.reload()` receipt includes the latest messages GET200 at **14:32:45.194Z**: three unique version-1 rows, in system/user/assistant order: `1b84800a-8de1-43d1-8755-ef0906453184`, `e312073d-f897-489f-a9b9-ff790a40cce5`, `7a6f2b4c-9bd3-436d-a52d-2b1a57c77807`.
- Persisted assistant content contains ARRR and asks for the user's location. No live-weather factual lookup or accuracy claim is warranted.

## Row 8 — creation succeeds, Character entry fails

- Actual Character POST **14:34:38.506Z → 201 at 14:34:38.527Z** creates ID **3**, version **1**, named `E2E-TestBot — PostgreSQL Fresh 20260917`, with `You are E2E-TestBot. Always respond with exactly: BEEP BOOP.`
- `testbot-create-and-entry.txt`, `testbot-entry-result.txt` and the blocked snapshot show the retained Character plus the setup dialog requiring an available model. The recorded events contain **zero complete-v2 requests** in this boundary. The working Pirate request does not turn this failed Character path into a pass.
- GET `/api/v1/characters/world-books?include_disabled=true` returns **500** at **14:33:37.055Z** and **14:33:38.108Z**, both `Failed to list world books`. The matrix's UAT239 causal diagnosis is external to this audit; no private backend log or worldbook excerpt was read. These two observed responses support the catalogue-failure qualification alone.
- Only this cell's retained attempt is certified; no invented three-attempt PG recovery or Character response/reload pass.

## Row 9 — actual answer reuse and one-card Study

### Reuse and provenance

- After an earlier More actions locator timeout, the explicit article-hover/retry clicks succeed. Actual `/api/v1/chat/knowledge/save` **201 at 14:37:35.371Z** creates Note `8fd0cca2-b54c-4d93-aec1-b53c6bea8daa` from conversation `392c8053-b7bc-45b2-8ef8-693b690b7939`, assistant message `7a6f2b4c-9bd3-436d-a52d-2b1a57c77807`.
- The reviewed flashcard dialog has its question explicitly edited before Save. The second save **201 at 14:38:32.634Z** creates card `c780acb5-a76d-4d60-983a-d49d3d508161` and linked Note `16652599-f165-4581-b9ae-9baac1bea92c`, preserving those conversation/message IDs. Two distinct Notes are expected from the two distinct save actions; this is not duplicate-card evidence.
- The original Note GET200 at 14:39:45.216Z and the editor input exactly match the canonical assistant text. Native Open conversation returns to the same three-message Chat. The card's back also exactly matches the assistant text; its `source_ref_type=note`, `source_ref_id=16652599-f165-4581-b9ae-9baac1bea92c` and message ID are retained.
- Later native Note link opens `/notes?source_ref_id=16652599-f165-4581-b9ae-9baac1bea92c`; GET200 at **14:51:26.874Z** returns the matching original content/version **1**, with a saved linked Note in the editor. This is Pirate-answer reuse, not a grounded media answer or successful generated five-card deck.

### Ratings, practice, interruption and re-rate

| Action | Actual response / saved state |
|---|---|
| Initial Easy | POST14:40:59.211Z, rating5; 20014:40:59.238Z; card v2, repetitions1, gap4days; due2026-09-21T14:40:59.229Z; session1. |
| Update schedule OFF practice Good | `beforeReviewPosts=1`, `afterReviewPosts=1`; completed one-card Cram and `Practice saved. Scheduling unchanged.` No scheduling review POST in the observed practice window. |
| Scheduled Good after browser recovery | POST14:49:59.405Z, rating3; 20014:49:59.434Z; card v3, repetitions2, gap10days; due2026-09-27T14:49:59.423Z; session2. |
| Re-rate Hard | UI at14:50:40.111Z still says Hard6days, despite preceding successful response and list preview saying Hard14days. POST14:51:00.773Z, rating2; 20014:51:00.798Z; card v4, repetitions3, gap14days; due2026-10-01T14:51:00.789Z; session3. |

- Initial Study preview, saved schedule and toast agree on Easy4days. The later stale Hard preview directly supports UAT235. The successful Good request predates the retained second Show answer locator timeout; the next observed UI already exposes the answer/rating controls. That timeout is not a failed Good review or an additional rating. The receipt alone does not preserve the whole interrupted command, so the parent description of its prior Re-rate click is supported by the resulting UI/events rather than a complete command transcript.
- After real reload in `study-rerate-reloaded.txt`, the native UI displays Reviewed today3, three completed one-card sessions and October1 due date. Wire sessions are exactly IDs1/2/3, each completed with `cards_reviewed=1`; card remains v4/repetitions3/gap14days. No claim that Re-rate undid the preceding stored review is made.
- Browser closure is explicit: last retained practice14:41:55.607Z, closure observed14:46:51Z, old browserPID30028 absent and freshPID38579, while the receipt reports API30006/frontend30012 unchanged. `study-scheduled-start.txt` and `study-resume-snapshot.txt` preserve the unavailable session; `study-restored.txt` shows the same saved4day review after recovery. Cause is unknown, not established as product crash. No uninterrupted session-continuity pass.
- Hard's saved card has `lapses=0` while the analytics UI reports33.3% lapse rate. This audit neither equates those metrics nor assigns a new issue; semantic diagnosis is separately pending.

## Row 2 — credential and backend recovery

- `auth-disconnect-current.txt` visibly shows **Add your credentials to use Media**, with reachable server and Open Settings. `auth-disconnected-media.txt` instead waited for **Credentials required** and timed out: a harness wording mismatch, not evidence that the gate failed to render.
- Normal visible credential form fill, Save and Test Connection are retained in redacted form. At **14:53:38.124Z**, UI shows Core reachable/RAG healthy; returning to Study preserves three reviews/sessions and October1 due date. Credential values were not accessed from private sources or emitted.
- **Important UX qualification:** `auth-reconnect-form.txt` explicitly contains **three `Failed to search media` notices**. Their presence is observed; this packet does not establish their individual timing/lifetime, cause, or a clean transient-toast outcome. This is consistent with the matrix's stated no-brief-toast-certification limit, not proof that no notices occurred.
- The owned API stop receipt records SIGTERM PID30006 at **14:55:14.046Z** with checked cwd/port18602; port check **14:55:33.019Z** records ECONNREFUSED. Real reload shows Backend readiness check failed/Retry. Native Retry's ensuing requests begin **14:56:45.997Z** and succeed200, with recovered UI at14:56:46.506Z; no credential re-entry appears in that Retry command.
- The latest full card response after outage is exactly equal to the latest card response retained after the earlier Study reload (including v4, due timestamp, source reference and content). Sessions remain1/2/3, each completed once; all three review records are displayed. The selected outage observer contains zero `pageerror` records, while the CLI reports expected network-console errors. No clean-console claim.
- These authorized inputs prove stop, actual unavailability and restored app/data. The matrix's replacementPID40346/same-profile/no-initialization statement is not independently established by a launch receipt within this allowlist; the observed recovery itself is sufficient for the bounded row2 outcome. API-key mode makes JWT natural-expiry N/A, not tested/pass.

## Limits and record fidelity

No vision, truly hidden-tab behavior, upstream provider outage, media-grounded reuse, exact-five-card deck, mixed-card/early-End Study, successful Character completion, multi-user isolation or full-matrix acceptance is inferred. Source, ingestion and five-card failures are referenced as existing matrix limits rather than re-audited here. The input progress snapshot is timestamped `2026-09-17T15:00:50.020Z` and phase `SQLite multi-user startup; PG single evidence retention`; concurrent later progress is outside this snapshot.

This was evidence inspection and deterministic receipt comparison, not new product test execution. The report contains no model reasoning. No private targets referenced by snapshot/log links were opened. All input bytes were rechecked unchanged before report creation.

## Input SHA-256 manifest

Paths are repository-relative. Native files are wrapper-redacted; matrix/progress are the exact read snapshots.

| Input | Bytes | SHA-256 |
|---|---:|---|
| `.tmp/uat-next-matrix-20260916/matrix-progress.json` | 17395 | `7c911ceb292a7891c1dd4268ecadf4bedc2ef7e6df821838f3e6f54ee3bbb175` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/answer-card-draft.txt` | 2057 | `c602f8a2294afb9db7db0ae7bbf0c6fe622511c0a160f08359b674fb291cd3da` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/answer-card-save.txt` | 16843 | `833fc694bbf3f41a4b22eafc4ee0bc99a7e7a117bcb1c27ee17130c8159b2d16` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/answer-note-backlink.txt` | 31223 | `b7f1e326603b3f108420c2445399d001d2cf1c4aa2d8795d6fa03f389ce3e5e6` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/answer-note-open.txt` | 1731 | `8c593806b2537ab27030feab5ec78cb5f3c21022eda5d0368d35f6fd62740e35` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/answer-notes-snapshot.txt` | 14081 | `f0555bd1640d5ba91be856e38c3e866c54238e44a9d947ae1f9ce371fdd73953` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/answer-original-note-selected.txt` | 792 | `0be108ba27608977f7222d7d24089e597637d1461c010f44cdcc5680911780ab` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/answer-reuse-actions.txt` | 1738 | `abd24670b6ac00f978aa69530ee504e3eb2a1c16da7c2a620bd929b821953106` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/answer-reuse-snapshot.txt` | 11406 | `6d2af01402b820a4a0e7436d725987f653e454fbf5b6f0725466d4a508fec42a` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/answer-save-note.txt` | 13993 | `175dff09a24d57f3df29d0005353cd0d627cb4b619a93297d9425cb37b1ca84e` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/auth-disconnect-current.txt` | 704 | `0b4b446e34d3abd7b9e3cca5378539eab0809d709632aa051a693209c0bb3d60` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/auth-disconnected-media.txt` | 151 | `9f211d0e84951138b377a89a108246a156bf16828ed37f65cfd960d68dd389f8` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/auth-outage-port.json` | 83 | `3eda57d4a5758e2b76a2a307fbffd928bc3e6817d5aec66b28111b8eef76b5de` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/auth-outage-recovered.txt` | 12808 | `e35e66bcbd76c5f41cd081642f1ba7e8212c9ce96ec96fef228abcb4db642c00` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/auth-outage-stop.json` | 344 | `f1fee80b2eb198d0a37ca1e97ea79efd611794162c9e618548afcce5b4c96f01` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/auth-outage-ui.txt` | 916 | `1e546ec7545e5d1a157ac5a34b0ef3bcf093f90c2ad1f64cc55fc0441ccb82ad` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/auth-reconnect-form.txt` | 5929 | `4f8b3a51a51ad1b7df9962177fa8342ea77fef6246ce12d84203245c83ea7262` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/auth-reconnected-redacted.txt` | 2592 | `c8cd28954c61fdb5cbafe574398ce83814afb64f7cd8483d368a0b8086c48c17` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/auth-reconnected-study.txt` | 1486 | `11293a94c0f28ad8af2ac70d4eb4b9d1327c270ba0c0f6bc65908f0683206393` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/auth-settings.txt` | 6453 | `5adea302160c6d0c1b9aeebcdb4c2b47385e9834b07febc9981cd0407f249b80` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/browser-interruption.json` | 556 | `79f3c13bbb6deea3686a6b3f06e2d391eab21a53c13f77bb04821789f5135fbe` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/pirate-chat-send.txt` | 1775 | `9c785eea5c601831104830dc796760cb40e3795ab70164b4fb264b0c73b22238` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/pirate-create-entry.txt` | 2154 | `bcd445958da7f49cc3aa4f0932230e2d16ba6f222b502fbf05532ab9330886a3` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/pirate-editor-snapshot.txt` | 218 | `116d7f024c273bb696b43d8859cd793b46208a559cc143e14a51d84625ac4510` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/pirate-reloaded-evidence.txt` | 39701 | `1ad78f18e12a652ad841feafeafcc556548f33b95f6900e5e3b1474efcc92ba5` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/pirate-save-new-chat.txt` | 1026 | `b1ea1a76e2d85b4ff925344016fccb479fcad4d0527c40731a010f631d13383d` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/pirate-save-state.txt` | 1762 | `3b5e375c32bb4c4ed082343a896294019dbcba1018952ae3eebc344d3c8e83fa` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/pirate-saved-library-snapshot.txt` | 18320 | `e194af73c78c7beb4ee4ed6cbb9859265ffcf71a5372548b12dfd6935872ab24` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/pirate-unsaved-confirm-dismiss.txt` | 141 | `6d082f7750761d82115bed0b6d36f0e8fb71f1a978cc04b58445dd078db8095b` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/pirate-use-in-chat.txt` | 2100 | `0ff79c542edb96fa64912333c02bce2db9d998ad5b6170c4e6d54eb048a021b7` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/study-card-revealed.txt` | 55370 | `76d61e2e66731bed3648f727aa60b9440149e96ae0f9109e8dcfab275c590a9a` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/study-easy-practice-entry.txt` | 57099 | `ff39af40cb4ce48f0c752c6800b4699997b4159690478107a94c9efbbfe53c06` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/study-note-source.txt` | 12887 | `da8baba380a97113ee0e7fa73c853cbbe6202ce6e913f3a23dc3df24899b3d91` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/study-practice-complete.txt` | 59897 | `c8b32732395e2d14782ce9007763e5c0d19a488d953e17adafc455c51e758dda` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/study-practice-snapshot.txt` | 8900 | `67fd14c3b7b21343a9b793cf20a9c12c2b50a6615af3fe06d41ded9715924c3b` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/study-rerate-observed.txt` | 5179 | `ca0efa4972af91d679010197f062b682275c170b33d053509cf0aa8503d2eb70` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/study-rerate-preview.txt` | 159 | `3fa738318587fc2b95cf490c75656411cd5bb6aa545a60b181a4cd243cb85c4c` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/study-rerate-reloaded.txt` | 57385 | `23bc30d0a787872cb68d3aba8741669bf4fa44bd647aabf409ad935266a3289f` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/study-restored.txt` | 3098 | `a6c7344ee48d96757907ae5115fcff3e900a9eb43d5f43bd099b96b7dc8f0475` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/study-resume-snapshot.txt` | 147 | `08551b8c12a71533f27c98e2d2dca9ed919e84f31403f356b408d6e67d70c838` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/study-scheduled-ready.txt` | 2717 | `4171ea2ea3c49612c98fd019d0bb9c58818a0b9d13830c2908926f8372c0fd7b` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/study-scheduled-start.txt` | 772 | `e50cb814455ea91225601a2e230aec6cb5999006d4ab03f919505b7af167a05b` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/testbot-blocked-snapshot.txt` | 10775 | `c12ccf976ed480ac1f932271924b3bb6c38bc8e19019a068af749ccf26099ee0` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/testbot-create-and-entry.txt` | 13710 | `50ab0b6234a6039c75f5b488e62a3a0ec43acfbcdc129917235319c75fbcd79e` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/testbot-entry-result.txt` | 12217 | `9d3b84b0887a47fedb29c969e045c9ba4effe530602cba78de3a574bb30ef873` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/testbot-field-labels.txt` | 952 | `e54c18ad1939d2ad817ad75291713ab4b4cb606c9fe1bf3416bbbf4c9cc21f01` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/testbot-form-snapshot.txt` | 16283 | `f0d592800ec42d886f4bcc6140c4d6508886d780859584e5a9c000c100e5ecbe` |
| `.tmp/uat-next-matrix-20260916/native/pg-single/testbot-new.txt` | 2688 | `34e343964a69071a1507c8f665243b9bc96076ce3e6fdb3e0e8bef10acd2e2e6` |
| `Docs/Reviews/FRESH_INSTALL_UAT_MATRIX_2026_09_17.md` | 22877 | `ad2c1c7afe0bbd996c1adc79137657b991cb325a95153f547756e74adcc5b1bf` |
