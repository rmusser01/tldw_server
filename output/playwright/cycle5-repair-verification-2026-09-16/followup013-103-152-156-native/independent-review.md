# Independent native acceptance audit: UAT013 / 103 / 152 / 156

## Scope and disposition

Read-only audit of retained browser/network receipts and screenshots. Source recorded by the parent and retention manifest: `223591ac4fb2520290634c74310351ce7d8b18ae`. Conversation: `2e94e4d9-7d1d-4572-a271-7f2c5b805cde`. No browser, service, model, source, task, or git mutation was performed by this audit; no new test run is claimed.

| Check | Independent disposition |
| --- | --- |
| UAT013 | **PASS for the actual Home selected-source handoff into the existing saved Cedar chat and correct final grounded answer.** Original TASK-13260.3 AC2 says: “Source starter questions reach Chat with the selected source context and prompt using the existing handoff contract.” It does not require a newly created Home media row. This run deliberately exercised deduplication. The separate reload source-activation failure UAT163 remains open. |
| UAT103 | **PASS for the reopened local retrieval-diagnostic promotion case.** The failed pair remains visible/local exactly once, is excluded from both later completion histories and canonical history, and produces no additional duplicate pair after reload. This native run complements, rather than replaces, the earlier ordinary/character/ownership regression evidence in TASK-13260.44. |
| UAT152 | **PASS for the pending real-generation acceptance:** actual RAG124 completes HTTP200 after 14,588 ms, with retrieved evidence and generated answer, exceeding the old 10-second limit. Read together with retained actual Custom→Balanced Save/reload evidence under followup152-153-native, this satisfies the native portion of TASK-13260.91 AC3. |
| UAT156 | **PASS for targeted source send/reload chronology:** the preserved raw UI user is immediately before its correct assistant; canonical matched user timestamp precedes assistant timestamp; final seven canonical IDs are unique and UI nine rows are accounted for. This complements the already recorded independent 128-test controls; it does not independently retest all malformed timestamp cases. |

These are bounded native acceptance verdicts, not a claim that every possible answer or all source restoration flows are correct. Parent retains responsibility for aggregate task completion and the open UAT163 work.

## Evidence locations and integrity

Private originals: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.tmp/uat013-103-156-final-native-20260916/`.

Durable retained evidence: `/Users/macbook-dev/Documents/GitHub/tldw_server2/output/playwright/cycle5-repair-verification-2026-09-16/followup013-103-152-156-native/`.

Independently verified all **31** manifest entries: original SHA256, retained SHA256, and byte identity after only the documented text trailing-whitespace/EOF normalization; PNGs are byte-identical. No mismatches. Manifest SHA256: `a4e21805a5c012e6dd9fe0ca4e7a68d1d24e189ac2f75bde8036447d400e6360`. The manifest records the parent's zero-hit known-secret scan; this audit did not rerun that scan.

Key retained hashes:

| File | SHA256 |
| --- | --- |
| final-source-events.txt | `c4aa8c3ea3bc1a4d04358da665aee2cf5cdf14641a5b179047483f2e6d057fe0` |
| final-source-reloaded.txt | `9e90b6f54c85a13f1b19d1e722b672caa3c7c03fd8144737b3df64c704fa9dcf` |
| final-source-reloaded.png | `401818db9c80f8c897b16b5838091032d95c01593d5491076de996a4fb433f02` |
| real-source-answer.txt | `97f424dc9f859bcdb7ca3b3b25e24d91964a89743a073082d02cf4c66a67b963` |
| failed-retrieval-reloaded.txt | `2ad2951eee2dc3e43216566f5cad51df865ce178e2f3aa4c22a16b47ff7ac3e9` |
| genuine-reloaded-events.txt | `a261acfdde04f1ec9730633b8b8f22c506dc7e7095495791468a7fbafccb1dd6` |
| pre-source-events.txt | `007125a6f5ecdfd9edb02e5c816bcbb901bbbf2e9bdd53c7b7a5eb0e2ce6dcab` |
| ingest-result.txt | `b8ebe0099812bd82a257e0cd58912fb8f3e345c49bbd3b4c6beafa908b805aaf` |

Both fictional Rowan input files have SHA256 `accd97c16fc4b07dc6bba54970a1ec0f60cbf60eb1ef8efc1da177a838b48f67`. Their four facts are Mara Chen; 7 December 2026; east entrance; Friday 19:30. Cedar's deliberately distinct facts are Leo Park and west entrance.

## Actual ingestion and Home handoff

The final cumulative event receipt contains the original creation as well as the later Home flow; the retained pre-source-events.txt and ingest-result.txt also preserve the earlier stage.

- Request21 creates ingest job1 (batch `67683312-94bd-40c8-8228-e6ac701ad192`, job UUID `0e22654c-ae07-49f1-a1b4-4743c6c3c610`). Request29 returns completed Success: media1 / UUID `f0cbedb9-54cb-438b-b0c5-f6a685d543c7`, message “Media 'rowan-community-library' added.” Completed 21:22:07 UTC.
- Actual Home file action creates request35/job2 (batch `236e9be1-3dea-4fb1-9dd1-01e47fc7b16f`). Request38 returns the same media1 with “already exists. Overwrite not enabled.” home-final-result.txt accurately reports zero succeeded, one skipped. This is a successful handoff through the deduplicated existing source, not evidence of a second newly created media row.
- home-first-source-done.txt and home-starter-clicked.txt show the real Home “Summarize this source.” starter carrying `Chat with this media: rowan-community-library-home.md` plus the question into the existing saved Cedar conversation.
- After the deliberately failed request and subsequent UAT163 observation, reapply-home.txt and reapply-starter.txt show a real Home starter reapplication. The final successful run is this reapplied handoff; it is not misrepresented as uninterrupted source state surviving the intervening reload.

## UAT103: actual failed retrieval remains local

Reviewed failed-retrieval-control.js: it routes the actual RAG endpoint, records the actual request, aborts transport, and clicks the real Send message button. It does not fabricate model output or a response body. Request58 is a real scoped `/api/v1/rag/search` request at `1789594210705`; its transport fails at `1789594210710`. Exactly one routed abort is recorded. There is no completion request between this failure and the later new user send79.

The local diagnostic is: “I couldn't retrieve evidence from the selected sources, so I did not send this as general chat. Check that the sources are ready and indexed, then try again.”

- failed-retrieval-reloaded.txt: five UI articles, diagnostic once. Canonical GET70/72/73/74/75/76: three rows (system plus Cedar pair).
- genuine-reloaded.txt: seven UI articles, diagnostic still once. The new genuine completion79 is persisted, while the failed pair remains local.
- final-source-reloaded.txt: nine UI articles, diagnostic still once. Canonical GET140 and146 agree exactly on seven rows.
- Completion79 has three messages (Cedar pair plus new Rowan question), no diagnostic and no failed starter pair. Completion126 has five messages (the two genuine prior pairs plus the newly augmented source user), no diagnostic and no failed starter pair.
- The final raw starter appears twice in UI because there were two distinct attempts: the failed attempt and the later successful source send. These are intentional separate turns. Each diagnostic/success assistant appears once; no extra server copy of the failed pair appears.

This run tests a later genuine submission, not an actual Retry-button click. Retry/Continue and ownership contracts remain covered by the separate implementation/regression review.

## UAT013 / 152: real source evidence and answer

Request124: actual `POST /api/v1/rag/search`, `include_media_ids:[1]`, `sources:["media_db"]`, generation enabled, provider llama.cpp and model `llama.cpp/gemma-4-26B-A4B-it`.

- Request time `1789594466876` (21:34:26.876 UTC); HTTP200 and request finished `1789594481464` (21:34:41.464 UTC): **14,588 ms**.
- Actual body is available: three documents from media1, with the four Rowan facts in `late_chunk:1:3`; `errors:[]`; `generation_executed:true`; retrieval cache false; explicit source selection enabled and cache bypassed; nonempty selected scope. Server total is 14.552567 seconds.
- RAG generated answer correctly states all four Rowan facts and distinguishes Cedar.
- Request126 follows immediately and sends those actual Rowan documents in the augmented final user message, despite genuine prior Cedar/wrong-answer history. It has `stream:true`, `save_to_db:true`, the same conversation, and client message ID `pa_f273-1672-a2e-7850`.
- Canonical assistant and actual UI both say: “The Rowan Community Library, coordinated by Mara Chen, is scheduled to open on 7 December 2026. The book return box is located at the east entrance, and the weekly book club meets every Friday at 19:30.”

This is an actual successful generation request exceeding 10 seconds. The RAG payload's `timeout_seconds:45` is a server budget and must not be confused with the saved browser generation timeout of 120 seconds established by the separate Settings native receipts. The evidence shows no automatic blind retry to conceal a timeout. The earlier failed request was the explicitly controlled transport-abort test, followed by visible new user actions and the Home reapplication.

## UAT156: exact final canonical identities and visible ordering

GET140 and146 both return total7, has_more false, identical messages and seven unique IDs:

| Order | Role | Canonical ID | UTC timestamp |
| --- | --- | --- | --- |
| 1 | system | `a319ede9-36b1-4f5d-b0dc-fe835e33577e` | 2026-09-16T21:18:44.632000Z |
| 2 | Cedar user | `7802ea57-a896-4d8f-84db-f0a0e639055a` | 2026-09-16T21:18:44.865000Z |
| 3 | Cedar assistant | `bc174efd-14da-414d-8f66-643ecd54f4b5` | 2026-09-16T21:18:48.405000Z |
| 4 | ordinary Rowan question | `f2d70245-d9a7-4e50-b6a4-1a3c02a25c2c` | 2026-09-16T21:31:24.009000Z |
| 5 | ordinary genuine wrong answer | `bf94832a-d85b-4ae9-8711-b639d80013eb` | 2026-09-16T21:31:30.099000Z |
| 6 | grounded source user | `8a424389-3b3a-40c6-852d-df292bba6e39` | 2026-09-16T21:34:41.531000Z |
| 7 | correct source assistant | `44c8a365-b3bc-4034-8839-32ae42588e3c` | 2026-09-16T21:34:47.032000Z |

The final canonical user metadata links client ID `pa_f273-1672-a2e-7850`, matching request126. The server stores the augmented source user; the reloaded UI preserves the raw user wording as article8 immediately before the correct assistant article9. The screenshot final-source-reloaded.png visually confirms that ordering and the full correct answer. The seven canonical rows plus two local failure rows explain all nine UI articles.

No direct local database timestamp probe was performed in this audit; the verified claim is the actual reloaded order, preserved raw content, matched canonical user identity/timestamp, and absence of unintended duplicates.

## Separate UAT163 and evidence limits

1. **UAT163 remains a real separate failure.** After the first reload, request79 is an ordinary completion with the Cedar pair and a new question referring to Rowan. It contains the word Rowan in the question but no Rowan source facts or document context, and there is no corresponding new RAG search. The genuine wrong answer says source text is missing and repeats Cedar facts. Reapplying the actual Home starter restores retrieval for124. This does not establish the cause of the earlier lost UAT013 wrong answer despite supplied evidence, and that historical observation is not erased.
2. **Raw SSE frames/ACKs were not retained.** Completion5/79/126 HTTP200 responses are observed, followed by observer `ERR_ABORTED` and a CDP response-body-unavailable error. There is no separate raw SSE receipt. This audit therefore makes no claim to have inspected `finish_reason`, `[DONE]`, or streamed acknowledgment frames. Persisted canonical server rows and actual UI independently prove the bounded completed answer/save outcome; the observer termination reason is not established.
3. real-source-answer.png mostly shows the prior answer and new user, with the new assistant content below the viewport/composer. It is not visual proof of the full final answer. real-source-answer.txt contains that answer, and final-source-reloaded.png directly displays it.
4. This is one targeted local single-user saved conversation using the real configured llama.cpp model. It is not a repeated model-quality evaluation or a fresh native test of every account/mode, Retry, image, timestamp fallback, or ownership branch.

No additional material acceptance defect was found within these four bounded checks. Aggregate completion should retain the separate UAT163 limitation and the absent raw-SSE limitation verbatim in substance.
