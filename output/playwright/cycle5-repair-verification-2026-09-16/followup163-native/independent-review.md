# UAT163 independent native acceptance

**PASS for the targeted cold reload → scoped retrieval → correct answer → second reload workflow.** This completes the pending native portion of TASK-13260.100 alongside the separately retained independent57/4 passing tests and causal baseline replays. No new defect found in this bounded workflow.

## Source provenance

The recorded source is UAT163 commit `ec28c34b7c` **plus three separately frozen UAT164 guidance files**, not a clean full-HEAD checkout. source-manifest.json records the three163 files and additional Models/index.tsx, Models/AvailableModelsList.tsx and CompanionHome/CompanionHomeShell.tsx hashes. Independently checked all six against current bytes; all match. The three163 hashes also match the prior independently reviewed freeze. This native run does not by itself accept UAT164.

## Verified sequence and outcome

1. home-start.txt/home-ready.txt show actual Home. starter-selected.txt records one real “Summarize this source.” click, reaching the existing saved Cedar conversation `2e94e4d9-7d1d-4572-a271-7f2c5b805cde`; its snapshot is dated21:59:02.525 UTC. This is the only recorded starter-click action in this packet.
2. session-before-reload.txt reads actual localStorage and records that conversation, `chatMode:"rag"`, `ragMediaIds:[1]`, **`fileRetrievalEnabled:true`**. The script only reads storage; it does not set the flag.
3. reload-start.txt records actual `page.reload()` with loading snapshot21:59:11.206 UTC. reloaded.txt returns to the same saved chat with nine existing UI articles. Subsequent recorded actions directly fill a new question and click Send; no second Home/starter action is recorded between reload and Send.
4. Request38 is actual `POST /api/v1/rag/search` at **21:59:41.212 UTC**, for the new question “After this reload, use only the selected Rowan source: name its coordinator, opening date, return-box location, and weekly book-club time.” It carries **`include_media_ids:[1]`, `sources:["media_db"]`**, generation enabled, provider llama.cpp and model `llama.cpp/gemma-4-26B-A4B-it`. HTTP200 at21:59:46.546: **5,334ms**. The body has three documents from media1, including all four Rowan facts; errors empty; generation executed; retrieval cache false; explicit selected scope enabled/nonempty. The generated RAG answer is correct.
5. Completion40 starts21:59:46.549 UTC and returns HTTP200 at21:59:47.067. Its seven input messages preserve the three genuine earlier pairs and add the newly augmented user with the actual retrieved Rowan documents. All four facts appear in this last user context; the earlier local retrieval diagnostic does not. It saves to the same conversation with client ID `pa_256b-c88c-68a-310d`.
6. answer.txt shows the real answer: “The coordinator is Mara Chen, the opening date is 7 December 2026, the return-box location is at the east entrance, and the weekly book-club time is every Friday at 19:30.” The answer matches the actual returned source, despite existing Cedar history.
7. answer-reload-start.txt records a second actual reload at22:00:13.317 UTC. answer-reloaded.txt and the independently inspected answer-reloaded.png show the raw new user as article10 before this correct answer as article11. The entire final answer is visibly readable in the screenshot.

## Canonical identity and duplicate control

Final canonical GET59 starts22:00:14.387 UTC, returns200, total9/has_more false, and **nine unique IDs**. Its first seven rows are identical to the pre-send canonical GET10. The new pair is:

| Role | Canonical ID | UTC timestamp |
| --- | --- | --- |
| user | `d9e23967-e209-4709-abc6-48e19ec0233f` | 2026-09-16T21:59:46.605000Z |
| assistant | `5bc644b7-efa0-4480-b9ec-d300b73ff894` | 2026-09-16T21:59:51.507000Z |

The new canonical user carries client ID `pa_256b-c88c-68a-310d`, matching completion40. Final UI11 equals canonical9 plus the existing local failed-retrieval pair. The diagnostic appears once in both pre/post reload UI, and zero times in canonical data or completion40 input. No new duplicate is present. The only generation requests in the captured event sequence are RAG38 and completion40.

## Evidence limits

- Read-only monitor.js records request/response metadata and response bodies; it does not route or fabricate responses. It excludes headers. final-events.txt contains244 events parsed from the CLI `### Result` JSON section.
- **No raw SSE/ACK claim:** completion40 records observer `net::ERR_ABORTED` at21:59:51.609 and CDP response-body-unavailable. No raw streamed frames, finish_reason, [DONE], or acknowledgment frames were available for independent inspection. Actual UI and canonical server rows establish the completed/persisted answer; the observer termination reason is not established.
- The retained action sequence has one starter before reload and no subsequent handoff. It is not an application-internal handoff-consumption counter or a continuous video.
- One actual same-owner local single-user conversation was exercised. Account/server rejection, legacy/explicit-disabled state, newer handoff/toggle races and deliberate chat switching remain covered by the independent automated review, not newly repeated natively here. This run is under10seconds and adds no new >10second timeout evidence; UAT152's separate14.588second receipt remains the relevant timeout proof.
- No product, browser, service, model, task-record or git action was performed by this audit. Only this requested private review file was written.

## Exact original receipt SHA256

| File | SHA256 |
| --- | --- |
| source-manifest.json | `c76eddf2c5e4d56487d48fd8fc561dd7a1fabcc7c434d8a08f799656f32b55a4` |
| starter-selected.txt | `8f51f897f63576021970f468d83446157b7be9b28e9a1b1650a89079467743d7` |
| session-before-reload.txt | `51ac1fd3b9d4ffacd5519c854090197c92f92beaa003396bac6fc9b018bc045a` |
| reload-start.txt | `2ca7f6e1100a8884a6a39dfb349d855ff151f5621f8a754ef982bbac44bfd25c` |
| question-sent.txt | `3da3f37ee04f50f7bee5805a6bb7825af464d73a17e319f709dd7a9305bbf703` |
| final-events.txt | `640bd90186366cb947b3d198d84ac03987669fe86ae6cd10e65bfeba901f7b5b` |
| answer-reloaded.txt | `27cdab34451ddef386b7a9493ba346f6dc4bc31ec1c1e8336d3188c74ad4befc` |
| answer-reloaded.png | `01f7c7430b7d89008f2824bc191e62cf802816ae5d684d8c26d938aa0787cd5b` |
