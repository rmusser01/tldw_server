# Independent SQLite single-user source/reuse audit

## Disposition

**Row 4 core workflow is supported, with its analysis-warning/API233 qualification. Row 9 Note/backlink and reviewed-card creation are supported; the full row remains in progress because Study is outside this packet.** No additional functional failure was found in these bounded outcomes.

Read-only review of 26 explicitly allowlisted, wrapper-redacted native receipts and the current matrix. No browser, model, service, database, credential, task or source operation was performed. Frozen source `8f8774e6c868b304a96d95ab82e28389c129a78b` is the parent/matrix attribution; this audit does not independently rehash the running source archive. Times below are UTC on 2026-09-17.

## Row 4: source identity, retrieval and Chat

- The actual Start processing, Minimize to Background and Open ingest wizard receipts show the same named file and one tracked job. Reopening shows Processing, one job, elapsed 0:24. The final result shows one saved-with-warning item and 1:14 elapsed.
- `POST /media/ingest/jobs` at 12:51:19.995 returned 200 and job **1**, UUID `416260cb-5888-49a4-b950-772a093a1502`. The completed readback at 12:52:34.265 identifies media **1**, UUID `9023fb11-1882-49e6-b351-d7ee4df1bb32`, result Warning, no terminal error. The response contains the identical provider-analysis-truncated warning twice; the UI displays it once. This supports the separately tracked UAT233/TASK13260.175 observation, not a successful analysis claim.
- The real Media read returns 306 words/1,914 characters of source text; processing analysis and version-1 analysis content are null. UI says Chunking: Completed and Vector: Pending. Decoded source-text SHA256 is `a94b1e966d89b7b94e0cd69dafe9ab1c554dc81accf43e57957276b08294225c`. This binds the saved text to downstream reuse; no independent upload-file byte comparison was performed.
- Knowledge QA requests `/rag/search/stream` at 12:55:23.869 with `sources=["media_db"]`, `search_mode="fts"`, chunk-level search, `enable_reranking=false`, citations/generation enabled, token limit 2000 and configured llama.cpp/Gemma. The 200 response contains the source-1 context and a completed generation envelope. Its assembled answer correctly states Dr. Mira Vale, Cedar Ridge and 18:00 every Friday, with citation [1]. A generic plan mentions reranking; the explicit request disables it, so the plan is not treated as proof that a reranker executed.
- Actual citation actions expose Source ID **1**, chunk **late_chunk:1:1**, and the supporting paragraph with all three facts. Open in Media opens `/media?id=1` in tab 1. Thus the citation leads to the same saved source.
- The initial Media-to-Chat handoff used the existing image conversation, as the pre-send snapshot shows Uploaded Image. The accepted grounded answer is a later text-only conversation. The separate source-send harness timeout is retained as a failure before submission, not counted as an applied Send. Expanded-composer receipt then exposes the complete 1,971-character source handoff; its 57-character title prefix plus the entire saved 1,914-character source matches exactly. Handoff SHA256: `e0ee0a607f0627dcccd8946c3e0cd425d8bad732529ccc988cef0ece2a89b30e`.
- The final actual Send produces one recorded `/chat/completions` request at 13:02:17.232 for conversation `975ea2bb-1844-4bd8-b1d0-79e875f26268`, client message `pa_26f9-e74d-ee1-79a0`. Its sole user message is the exact handoff plus the 175-character question suffix, 2,146 characters total. The response is 200; body read finishes at 13:02:21.157. UI and persisted assistant text agree: “Dr. Mira Vale directs Rowan Observatory, which is located at Cedar Ridge. Public tours begin at 18:00 every Friday.”
- This Chat is **inline supplied-source grounding**, not a scoped-RAG Chat request. Knowledge QA is the separate FTS retrieval proof. The Chat stream body itself was not retained by this observer; answer content is supported by the visible UI and subsequent canonical readbacks, not an invented SSE transcript.

The actual `page.reload()` receipt and readbacks at 13:03:30.968 onward preserve exactly three unique canonical version-1 rows, in order:

| Role | ID |
|---|---|
| system | `930b1fcc-e609-4a40-9285-e526ae814a5f` |
| user | `fb123f28-bc99-48e1-8c3e-338c6672063f` |
| assistant | `29c95eaa-b8c9-4567-a4b6-0ed23f278bc8` |

The stored user content equals the actual request, and the assistant contains the three source-supported facts. The later Note backlink readback at 13:06:45.892 returns the same trio without duplicates.

## Row 9: reuse of that answer

1. Actual Save to Notes triggers `/chat/knowledge/save` at 13:04:30.029 with the exact assistant snippet, conversation/message IDs above and `make_flashcard=false`. Response **201** creates Note `c219a2d1-41c8-42d7-ad03-02803aa7c512`. Its subsequent 200 GET and selected Notes UI show the same clean 115-character answer, version 1, and Saved from Chat provenance.
2. The actual Open conversation menu action returns to the saved Chat. Its canonical GET and visible answer retain the same source conversation and trio; this establishes the backlink, not just copied prose.
3. Review flashcard visibly presents the same answer, initially with an empty Question and disabled Save. The final recorded save request at 13:08:13.557 supplies the three-fact question and exact answer with `make_flashcard=true`. Response **201** creates flashcard `2cdc609e-7d72-4f74-882b-c134e9be32f6` and linked Note `3b597cff-d69c-4e95-86b6-7e4b4895050b`, returning the same conversation/message provenance. This supports reviewed creation; these inputs do not separately prove the later card reload, Study queue, rating or schedule behavior.

## Limits

- Source ingestion succeeded with warnings; provider analysis did not complete. The later correct QA/Chat answers do not erase that earlier analysis failure.
- The new card was created from the grounded Chat answer; this is not acceptance of the separate five-card biology journey.
- No vision, hidden-tab behavior, upstream outage, complete SQLite cell, other database/auth cell or full matrix is certified.
- Snapshot/text evidence was inspected; no referenced private snapshot or console file was opened. Console cleanliness is not claimed. Earlier unrelated events inside cumulative receipts are not attributed to this source workflow.
- Matrix rows 4 and 9 match these bounded claims at the document hash below. The matrix is being updated independently by the parent; later row changes are not covered by this snapshot.

## Reviewed input hashes

SHA256 is over exact original receipt bytes; no normalization or copied inputs. Native paths below are relative to the repository. The audit is bound to these 27 inputs.

| Input | Bytes | SHA256 |
|---|---:|---|
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/ingest-start.txt` | 15151 | `70255501d03d39fb8ebbe9ae0e0f63e99b8e9829649b4c7369fdb4ff91aeee68` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/ingest-minimize.txt` | 12521 | `3b2da42d467f357167c1f31f5e6c776730db391b3d98bdc3293308314955b6cd` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/ingest-resume.txt` | 15591 | `49544464d21e0abf2f6847e28ea62d44aa4f1cbf3f53718a16585dc6819445b5` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/ingest-complete.txt` | 780 | `9a6ee5b14927e63a30b8fedb2b78c49b8ec4f474c04cad0d5969a4d5c85a96a7` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/ingest-terminal-evidence.txt` | 228742 | `884c795831a4840d9e1c71ca52257d03bd36558e09251ebb227329bccda1939a` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/source-open-actual.txt` | 16934 | `3592e4929313aefe046b7e4e01280df93afc86f7242383871cef77acac5828c5` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/knowledge-query-evidence.txt` | 270085 | `8e3b8e97f3e63fe54042eebf0fd62c9b9e400488309f9c32d13378f9681363e5` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/knowledge-answer-snapshot.txt` | 19082 | `90b5b76b992e6859ddd58739c299335329c2fd627b2f296e40ce123e8a0a31e4` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/knowledge-citation-excerpt.txt` | 1147 | `294cb42b7ce88c3d63f3be02840c355b9ecb00e5375ee2698f60b30b9ada6b62` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/knowledge-source-view-actual.txt` | 20796 | `f6d103d6560445e2f2c992ffdce361065b1164dd10717e01dba4a80538baa74d` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/citation-media-link.txt` | 20933 | `7e5c908e8e02b9db32e768230160113309493b9987d787772355e842535a410c` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/citation-media-tab.txt` | 228 | `b4e46019603b5b460e0430450224a65e8197d38b1fa7161d555b0a2da4b6b518` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/media-to-chat-handoff-actual.txt` | 4107 | `0b01b1858e4a9bdbd53af5b8644a030269958660610d3cc3505136077ebde7af` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/media-chat-pre-send.txt` | 12592 | `8aa0aee74043f96412ce7f4f78a7537f083b798e7b312b48c9c9f6a5e504d5d3` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/media-source-chat-send.txt` | 73 | `2a52b807eb5eb1d8cb79a21fdf6ddf6b6588b890cdefc0e02edae54946e9133d` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/media-handoff-expanded.txt` | 2320 | `e9b7b11d318a0590fd12dafead76222efb6ff39c1745805ee029fb9cc7044e4e` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/media-source-chat-send-final.txt` | 1046 | `92691566fc121436b3425208d7c1ba83fde181b90ccf0c26994e1518f85e61e4` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/media-source-chat-result.txt` | 63326 | `97a8037e372c3b39522709660bcf5aa81108e643a94bb74d1f0b41ed983e8aa8` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/media-chat-reload.txt` | 838 | `2fb034fc59770c246e4ae85aedbe51875d13b302d1370f4225148c4c2d5c57e0` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/source-chat-reloaded-snapshot.txt` | 13381 | `87436960e411684d490cf781997ae1f3a68e2f7eee5aec4377a5026e6746c580` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/source-answer-save-note.txt` | 14088 | `30ba377b2a239f4952a4d97d07923cbac5c2896a463f30719163f15832c3b27a` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/saved-note-selected-after-tour.txt` | 13747 | `c12df202d597d3a98324e6afbd33ac18f2ed63d0e54559425a77d0f1886f6856` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/note-open-conversation.txt` | 13739 | `294b94a1a82ce43f1f489a2d3e1196eb8f69a575bae3d91161bf5048fb6c170d` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/note-backlink-chat.txt` | 13404 | `7c47e2e64b749d6432697361949222e88e04be194a5a14b0ebe1387cb1c86c3b` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/chat-card-draft-review.txt` | 14382 | `34bbcfcd6024c2e8cac789faf5735359a927a99501a9d8bef192983d4ad2d46c` |
| `.tmp/uat-next-matrix-20260916/native/sqlite-single/answer-reuse-evidence.txt` | 139999 | `920db5098398e435b9d229e067647c5bf0abd091a8ab49ccd68bfb293bd40f1b` |
| `Docs/Reviews/FRESH_INSTALL_UAT_MATRIX_2026_09_17.md` | 8902 | `6d2e13abdcee5dc204fb8e011a8d8c5230758a2dfd7b282b5e000c67a31245db` |

Audit written 2026-09-17T13:17:24.982430+00:00.
