# Independent PostgreSQL single-user setup / Chat audit

## Disposition

**Bounded setup/first-chat, ordinary two-turn context and persistence, and backend availability rejection/Retry are supported.** The generic error guidance reproduces the already tracked UAT232 limitation. Actual image upload, capability guard, local-turn persistence and canonical-empty behavior are supported with the harness qualification below. No vision, true-hidden visibility, upstream generation outage, whole-cell or full-matrix acceptance is claimed.

Read-only audit of the 27 named nonprivate receipt files below. Ongoing ingest files, private helpers/logs/credentials, browser, runtime, database and product tests were not accessed or operated. The sole write is this report. All times are UTC on 2026-09-17.

## Setup and first real Chat

`startup-summary.json` attributes this cell to frozen revision `8f8774e6c868b304a96d95ab82e28389c129a78b`, single-user PostgreSQL, API18602/UI18682, initializer exit0, backend PID30006 and frontend PID30012. It records the official `pg_temp_db`/`pg_temp_db_session` fixtures and a login runtime role with superuser, BYPASSRLS, inheritance, createdb, createrole and replication false, memberships0. This audit binds that startup receipt; it does not independently query live role settings or rehash the runtime/archive. Dependencies are explicitly reused.

The fresh wizard is visible. Native controls select Solo/local and llama.cpp and enter `http://127.0.0.1:9099/v1`. The discovered-model snapshot has an empty Default model and the actual Gemma GGUF choice. The subsequent action selects that choice and invokes Validate; its immediate snapshot still says validating. The packet does not separately capture the final settled validation result or Save providers click. Later optional setup steps and the successful configured first-chat prove the working setup outcome without inventing that missing click receipt.

The observed real `POST /api/v1/setup/first-run/first-chat` sends “Say exactly: fresh PostgreSQL single-user ready.” at **14:10:55.018**. Response **200** at **14:10:56.461** reports status ready, provider `llamacpp`, the actual configured Gemma model and exact response “fresh PostgreSQL single-user ready.”

The UI then truthfully presents Restore media access: a working server key is still needed. `key-save-redacted.txt` records normal entry through Single-user API key and Save API key at **14:11:54.134**, with the value redacted. Ordinary Chat subsequently works. This supports the disclosed manual-key recovery path, not automatic WebUI credential provisioning. No secret was read for this audit.

## Two turns, failed turn and native Retry

Saved ordinary conversation: `2f4fa839-f278-4a7b-9397-e1b706a35c72`.

| Event | Request time | Result |
|---|---|---|
| First ordinary turn | 14:12:34.471 | 200; body read 14:12:36.173; answer ORBIT-742. |
| Second ordinary turn | 14:13:08.195 | 200; body read 14:13:09.895; answer ORBIT-742. |
| Controlled unavailable model | 14:15:02.899 | Actual backend400 at 14:15:02.967, `model_not_available`. |
| Visible Retry same model | 14:15:22.818 | 200; body read 14:15:25.695; answer ORBIT-742. |

The second request contains the exact first user turn and its ORBIT-742 assistant response before the second question. Actual `page.reload()` followed by the settled readback returns **five unique version-1 rows**: one system and two user/assistant pairs. Intermediate in-flight reads returning zero/one/two rows are not treated as final history.

The fault is a one-shot outgoing-request override: configured `llama`/Gemma becomes Ollama plus a deliberately unavailable model, using `route.continue` without response fulfillment. The actual backend error body corroborates that effective provider/model. The ordinary observer records the request before interception; it is not evidence that the original model produced the400. This is backend model-availability rejection, not an upstream generation outage.

The native Retry button restores the configured model and preserves client message **`pa_f490-b7e2-89c-ae73`**, conversation, all five context messages and generation options. A decoded deep comparison confirms its only difference from the original pre-override failed request is `metadata.tldw_retry_failed_turn=true`. It does not resend the local error as context. Normal reload returns **seven unique version-1 rows**, with the failed client identity stored on exactly one user row and one final assistant answer.

| Role | Canonical ID | Present after two-turn reload |
|---|---|---|
| system | `a7a073d1-9b68-41d4-8ac1-367ed4e22c28` | yes |
| user1 | `6709ef37-ab5f-433e-b6ff-386ebcc3dd81` | yes |
| assistant1 | `76a27b62-e6d8-4213-9fd3-ca9691a9647f` | yes |
| user2 | `d9c17a32-d7b5-43b4-a30d-2ac33d0b328d` | yes |
| assistant2 | `3e2f8bc9-dcbb-46d1-9cb5-d9fa06df53a0` | yes |
| user3 / retried client | `7987b238-1984-4fd5-b921-5aaf5b674720` | added by recovered turn |
| assistant3 | `eba30676-ad5c-463c-8bf7-1ebffcb47f88` | added by recovered turn |

All three canonical assistant bodies are exactly ORBIT-742. The settled two-turn read is at **14:13:37.078**; the retry-reload receipt includes the final seven at **14:15:46.461**. The actual first-chat wizard exchange is separate from this ordinary conversation.

The visible400 UI still says “Something went wrong while talking to your tldw server” and recommends trying again/Health & diagnostics. This supports generic guidance UAT232 remaining unresolved despite the precise technical error and functional Retry recovery. It does not justify a clean-error-UX claim.

## Image flow and harness attribution

- The original combined action starts a new saved Chat, opens the real file chooser and targets the public `apps/tldw-frontend/public/icon/128.png`. `chat-current-snapshot.txt` records the pending native chooser; `image-upload.txt` records actual `fileChooser.setFiles` for that path.
- The upload receipt's accessibility snapshot already shows **Uploaded Image**, the image question, one user article and an assistant capability explanation with **2 of 2**. Together with the retained combined action, this supports the CLI-resumed original Send and Retry after upload.
- A later harness action accidentally sent the identical prompt again before its strict Retry locator failed because two buttons existed. The final receipt explicitly preserves this qualification; the two local user prompt turns are **harness-authored, not a product duplication finding**. The earlier modal/strict-locator errors remain unsuccessful harness operations, not silent successful retries.
- Actual `page.reload()` in `image-final-evidence.txt` restores two local prompt turns and their image-support explanations. The first remains grouped 2 of 2. The explanation says image support is not confirmed and directs the user to an image-capable model or a new text-only conversation.
- The full retained observer still has exactly the four preceding ordinary completion requests; the last was Retry at **14:15:22.818**. No completion is dispatched for either later image prompt. The image conversation **`87527762-402d-41cd-a729-02210d7f1ac1`** repeatedly returns canonical messages **total0**, including after reload at **14:17:50.313**. These never-sent local guard turns are not claimed as server-persisted messages.

The actual uploaded image element is shown before the final reload. The final receipt uses body text rather than an image-element/byte inspection, so it proves local prompt/guard recovery, not post-reload image-byte equality or a separate post-reload image-element observation. No vision inference or independent provider-capability configuration test was performed here.

## Limits

No true-hidden-tab control, upstream outage, semantic image answer, multi-user isolation or full matrix outcome is established. Startup-role evidence is receipt-bound. Provider validation completion/Save click and post-reload image-byte preservation have the narrow evidence limits stated above; they do not invalidate the observed first-chat/text recovery outcomes. No clean-console claim is made. This audit does not evaluate the ongoing ingestion journey.

## Reviewed input hashes

Exact original bytes, without normalization. Paths below are relative to `.tmp/uat-next-matrix-20260916/native/pg-single/`.

| Input | Bytes | SHA256 |
|---|---:|---|
| `chat-controlled-failure.txt` | 4212 | `7364b897fa4e423844c0e718c13492a3e7f6f1268a61cc71550b1651fbca162f` |
| `chat-current-snapshot.txt` | 183 | `f23013a2042f02c5fed90cce282d0a0a78c657557450a518e108d7c4bde0f12d` |
| `chat-entry.txt` | 1657 | `eacf0f0ac812a537e4587678e75ec3bcabe5c36c7e9685e79530af1fa9848758` |
| `chat-retry-reloaded.txt` | 73398 | `85b03a6da27cd0b931ca735d55d747d9ec28592c316a0c26990c551ec73baa5d` |
| `chat-retry.txt` | 53052 | `29a06433128625daf3540985f1754a1823cfbe50f8c7c3f852763acfc123797b` |
| `chat-turn1-result.txt` | 16113 | `ea5f45d4355a45cca4d6d80e0584248e959fa7f0356632e19f464cac5914c953` |
| `chat-turn1-send.txt` | 1702 | `41c1fdfdee24aa69cf0eec694f20e5da6d59f56f31a39357296c7a9e037a0893` |
| `chat-turn2-send.txt` | 2039 | `5e16af8ef7de2518ce4b0aa6bee3d77ccb13a7fbfc7f88e8d9f57db156599004` |
| `chat-two-turn-reload.txt` | 19160 | `217544794e4190f26574550ae2856c64799e46179966511b936bc03fa1222ff6` |
| `chat-two-turn-settled.txt` | 48603 | `8f5c85ebab421ba2aa08f486deafb20d087bf558d3c8e343cd97c4a31ed971da` |
| `first-chat-finish-snapshot.txt` | 1070 | `95ceca3eddf1cad979cbcfeeabd7f4522d917993b43b0a413ebe369e90f0e1e8` |
| `first-chat-observer.txt` | 1030 | `2ec1f6dbb0b4e27b434ba0b6b023525b646a54ac05a7e615413ada2671271125` |
| `first-chat-result.txt` | 1261 | `be29dfe51d3d2cbcc311a880ff4a128d580f9b25e41f1ee52a434117672256dd` |
| `first-chat-send.txt` | 1383 | `bb50ae27aac54929afa48cbb8582869cdb79e5289a4b83a180f6c946441808eb` |
| `image-attach-guard-retry.txt` | 1257 | `ddbdd7dd957a305dc2e247c255927a1609b90c1f4bfacec1b08c787a21faec87` |
| `image-final-evidence.txt` | 117754 | `29050c586db9544d7ebdb634e4f653a6aa86294f46b519bf81b6a844e4f24ed8` |
| `image-guard-retry-reloaded.txt` | 843 | `2d97d2e571dd63e13257d7117275902c98cf343d2d98f51828f76318557b91d9` |
| `image-upload.txt` | 13544 | `8144f4df13c13930e45b6818d549a2944e251153198125fc053a06490fc10869` |
| `key-save-redacted.txt` | 979 | `73c156b01a290b3c1859b85bc0d0a78a3814e8689f0497f5468080eb6aadbb27` |
| `provider-discovered-snapshot.txt` | 6918 | `500c8d25e05975fb66b50a2a87f0e09653c3cd9280b54b9954fd3708f7892563` |
| `provider-discovery-start.txt` | 1491 | `e571a1ecd7b1b811f8b96423896146b3d12935be0348fd3feb4f7556c9f9627e` |
| `provider-entry.txt` | 1908 | `3ae0445ec7a0469bfa579032b2633752a5d9172f0040a4b1c1b2043c5cc2cb34` |
| `provider-selected-validation.txt` | 1641 | `bae8fad3f66cf2b127e5d8d5f3101b793baac7a0ca6c5d91c56e2b137031605c` |
| `setup-mcp-step.txt` | 925 | `d206eaeafbe3db08d92f8a50561db9ce63c0129dd8acd869cb50f28270e3121a` |
| `setup-optional-steps.txt` | 2656 | `6644f7714da3d4885bef031ee0397a147b29f21cfbac599df2dd23624f7fac7d` |
| `setup-snapshot.txt` | 3179 | `f9c9839fafe98c16b25246adde572c8f59c8714a5740ab8b5aff51a59b3731d9` |
| `startup-summary.json` | 1023 | `d07c453a47c5512fed514665138d91b86a81f081b83ae7d4cd113ab937ee9093` |

Audit completed 2026-09-17T14:22:44.133920+00:00.
