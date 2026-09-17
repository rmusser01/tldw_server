# SQLite-multi middle-journey independent native audit

Reviewed 2026-09-17T16:09:21.879119+00:00, under parent TASK13260. Frozen source attribution: `8f8774e6c868b304a96d95ab82e28389c129a78b`. The companion JSON binds 54 exact inputs: 53 authorized native receipts/scripts and the current controller snapshot. Only this report and its manifest were written; no browser, runtime, private-file, source-code, inference, product-test, Backlog/tracker or git action occurred.

## Verdict

| Boundary | Bounded disposition |
|---|---|
| Row 4: Knowledge QA, citation and recovered inline-source Chat | Supported, with known handoff defect UAT241. Initial handoff is not a clean pass. |
| Row 7: Pirate Prompt | Pass: saved prompt, actual system instruction, real ARRR answer and reload. |
| Row 8: TestBot | Pass for this Alice/operator-default path: Character4, actual complete-v2, visible exact `BEEP BOOP.` and canonical reload. Raw persisted output is not an exact-only string. |
| Row 9: Note/card/backlink and one-card Study | Partial: reuse/persistence pass; UAT235 and240 reproduce. Separate Due-completion singular-label defect observed. No five-card/mixed/early-End acceptance. |
| Natural access-token expiry | Accepted for ordinary automatic renewal after natural expiry in the separately parked Alice context. |

Rows 5/10/11/12 and their ongoing results are outside this audit. UAT241's cause has a separate source audit; this report verifies the later successful path without erasing the initial failure.

## Source QA, citation and ordinary Chat

Knowledge POST `/api/v1/rag/search/stream` at **15:31:16.103Z** uses `sources=[media_db]`, standard strategy, **fts/chunk**, actual llama.cpp/Gemma generation and citations. Response is **200 at 15:31:21.578Z**, body-read complete at15:31:29.541Z. The retained response has five contexts (`late_chunk:1:1`,1:5,1:2,1:3,1:0) and a successful terminal record with upstream_dispatched/output_emitted=true and fallback=false. Raw streams/provider reasoning are not reproduced here.

The settled UI answers **Dr. Mira Vale / Cedar Ridge / Friday18:00** with one cited source. Expanded excerpt and native Source1 preview identify Media source1, chunk `late_chunk:1:1`, and contain those same fictional-source facts. Native Open in Media opens `/media?id=1`, displays the source and1914-character statistics, while preserving the separate expiry context. These are five chunks of one source, not five independent documents.

The initial source-send attempt stops with `Source not present in native handoff` before Send. After the documented recovery, native composer expansion contains **1971 characters**:55-character handoff title, two newlines and1914-character source. The successful native command appends the three-fact question/instruction and clicks Send. Canonical user text independently equals the composer value plus that exact127-character suffix, totaling **2098 characters**. Completion response is **200 at15:43:47.190Z**. Its visible/persisted answer is the correct115-character sentence:

> Dr. Mira Vale directs Rowan Observatory, which is located at Cedar Ridge. Public tours begin at 18:00 every Friday.

This is ordinary Chat with source text inline, distinct from Knowledge QA retrieval. It does not prove scoped-RAG Chat. The full outbound ordinary-completion request body falls outside these source captures' event window; the native Send, actual200 and exact source-bearing canonical user row establish the bounded result.

Conversation `af70ef91-059c-44bb-b106-8704f61e8820` is actually reloaded. The immediate reload-action receipt ends before the canonical response; later readbacks **15:55:32.884/.892Z**, retained in `pirate-reloaded.txt`, prove three unique version1 rows:

- System `53df80ee-57eb-4959-9415-70d7796321fe`.
- User `673079ed-35cb-4035-abe3-31847bc19d94`.
- Assistant `e4aab5d7-333e-4b44-953e-e1b738114bdb`.

Earlier two-row reads occurred during generation and are not a missing-answer defect. Ingestion job completion/UUID, analysis quality, chunk/vector completion and the original Media search submission are outside these selected inputs; this audit certifies actual retrieved source/citation, Media1 destination and recovered source Chat.

## Note/card creation, provenance and backlink

Save to Notes returns **201 at15:47:36.277Z**, creating `51470044-f5ea-4e6b-a5ef-3d49dfd224ab`. Reviewed card Save returns **201 at15:49:24.255Z**, creating card `56320d51-cc2f-4049-baed-65a888230112` and linked Note `6174102e-59d7-4c12-8eb2-0e408ef935aa`. Both responses preserve the source conversation and assistant ID above.

The canonical answer, card dialog's reviewed answer and original Note editor input are byte-equal115-character strings. The question is explicitly reviewed/edited before Save. This is one reviewed card, not a generated five-card deck. Actual Note GET200 from15:50:06.995Z retains owner2/version1, exact answer and conversation/message provenance. Later card readback retains source_ref_type=note, the second Note ID and that same assistant message. Two Notes follow two distinct Save actions; this is not duplicate-card evidence.

An earlier top-level Open conversation locator times out because the real action is a **menuitem under Note editor More actions**. `reuse-backlink-pirate-entry.txt` clicks that actual item and renders the saved source Chat; later canonical IDs match. `study-final-note-entry.txt` separately navigates directly to the original Note URL. It does not prove a click on the card's source link, so that additional action is not claimed.

## Study: persistence and retained failures

| Action | Actual saved result |
|---|---|
| Easy | POST15:52:05.232Z, rating5;20015:52:05.254Z; v2/repetitions1/lapses0,4days, due2026-09-21T15:52:05.251Z; session1. |
| Practice, schedule OFF | Explicit switch false; review POST count1 before/1 after; Cram completion and Scheduling unchanged. |
| Schedule ON, Good, Re-rate |20015:53:02.620Z; v3/repetitions2/lapses0,10days; response next Hard14days. Re-rate UI incorrectly keeps Hard6days. |
| Hard |20015:53:40.308Z; v4/repetitions3/lapses0,14days, due2026-10-01T15:53:40.305Z; session3. |

The actual Good→Re-rate command is retained. Stale Hard6days versus server14days reproduces **UAT235**; choosing Hard stores14days. After actual reload, the early heading-only snapshot is superseded by `study-final-note-entry.txt`'s settled capture. It shows Reviewed today3, retention66.7%/lapse33.3%, three completed one-card sessions and October1 due date. Actual sessionsGET200 at15:53:42.248Z has IDs1/2/3, all completed/cards_reviewed1/client2. Card readback at15:53:42.457Z retains v4, the answer and source reference.

Card lapses0 versus aggregate lapse33.3333% at15:53:42.445Z reproduces **UAT240**'s native behavior; the scheduler/analytics semantic diagnosis is separate. Three stored review events are not treated as an undo failure merely because the control is named Re-rate.

**Additional observed copy defect:** `study-easy-practice.txt`'s Due completion says **`1 cards reviewed this session`**. The off-practice Cram message is correctly singular. This exact line was reported to root for separate association (parent plans UAT242); no finding/task was edited here. Retained harness action/role timeouts are not substituted for the later successful outcomes. Five-generated-card, mixed-card and early-End controls remain uncertified.

## Pirate Prompt

Native Save creates `Pirate Prompt — Alice SQLite 20260917`, local edit `pa_aeeb-25d9-cad-4609`; settled library shows **Synced #1**. Native system-instruction insertion sends POST **15:56:53.019Z**, providerllama/realGemma, exact87-character prompt `You are a pirate. Respond to everything in pirate speak. Always say ARRR at least once.` plus the weather question. Response **20015:56:53.423Z**, body-read15:56:57.515Z. The answer containsARRR and requests location; no live-weather lookup is asserted.

Actual reload followed by canonicalGET200 **15:57:09.200Z** preserves three unique v1 rows in `870f8d17-44ff-488e-ae03-ee03f0e90f5d`: system `89b9b36b-d3b5-4560-96aa-61d0d2e7577c`, user `418f4da0-8485-446b-9793-199e1e66c639`, assistant `3e20bde6-26c6-4c35-991d-5ad0c5105255`.

## TestBot Character

CharacterPOST **20115:57:55.222Z** creates **4**,v1, `E2E-TestBot — Alice SQLite 20260917`, exact instruction `You are E2E-TestBot. Always respond with exactly: BEEP BOOP.` World-book catalogue returns200 on this SQLite path.

Actual Character Chat creates `ea6efc19-3395-4640-8757-88bd6f0be6ec`. `/complete-v2` POST **15:58:32.965Z** has include_character_context=true, providerllama/rawGemma and save_to_db=true. Response **20015:58:33.219Z**, body-read15:58:47.900Z. Persist endpoint **20015:58:47.945Z** confirms saved assistant `pa_b948-041e-646-1eea` for user `916ae4cd-7d19-4d91-9446-93b8d1188d95`, speakerCharacter4.

Real reload waits for exact visible **`BEEP BOOP.`**. CanonicalGET200 **15:58:51.628Z** has those two unique v1 rows; the assistant sender is the Character name. No additional system row is invented. Raw persisted assistant content is4337 characters including think markup; removing its marked reasoning section leaves `BEEP BOOP.`. Native UI shows reasoning collapsed and that exact answer. **Visible exact-output acceptance is supported; raw/provider/canonical exact-only output is not.** No reasoning text is included in this report.

This operator-default llama/rawmodel path succeeds. It does not fix or overturn UAT236's failed single-user qualified-model paths or certify every Character configuration. Earlier persona-discovery307/429 observations are not erased by the successful authenticated Character outcome.

## Natural expiry and ordinary renewal

The separate child uses `browser.newContext()` and ordinary visible login; no storage/token injection. The initial guide/login locator timing errors are retained, then Alice2 login succeeds **15:07:07.498Z** with1800-second advertised lifetime and authenticated Notes200. The page is closed at15:07:52.271Z with **0 pages/0 service workers** reported; the context remains parked. No artificial clock/token mutation appears in the scripts.

Return starts **15:39:42.644Z**, **1955.146 seconds** after issuance,155.146 seconds beyond lifetime. The script checks the natural deadline before opening Notes. Sorted event timestamps show **auth/me40115:39:42.891Z → refresh20015:39:42.920Z → Alice2 identity20015:39:42.936Z**, followed by more identity200s. Async callback array insertion order is not mistaken for actual chronological order.

Owned NotesGET200 **15:39:43.566Z** returns Biology Note `4b06d0b4-51a5-4857-8d83-4082df846767`, owner2/version1. Settled UI15:39:43.757Z shows that Note and active notifications, without a login form. The return command does not reenter a password. This supports natural access-token expiry followed by normal refresh/recovery, not refresh-token expiration/revocation, forced sign-out or all account-switch/logout boundaries. A later context-close action is not recorded in this allowlist.

## Limits and integrity

No full matrix, vision, true hidden-tab inference, upstream-generation-outage, cross-owner isolation or unrelated row10 exact-token pass is inferred. Source ingestion and five-card failure qualification rely on their separate matrix evidence rather than being re-audited here. The controller has chronological pending prose; settled evidence above controls these conclusions.

The first review-only canonical parser encountered the Character201 single-message response mixed with GET message-list events; it stopped before writing. Filtering actual list readbacks completed the checks without altering evidence or product. Every input byte was rechecked unchanged before writing the two authorized outputs. The companion JSON binds this report and all inputs without normalization; it contains no credentials, raw streams or model reasoning.
