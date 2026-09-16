# UAT122 native attachment guard — PASS with separate124/125 findings

Committed frontend53ef4bf4d5 (includes1239cabac49f8), preserved single API18402/backend e7bf2184d3. Next18482 restarted without changing API/config/data. Existing local Gemma and unavailable Ollama were used only for capability-refusal controls; no inference request was dispatched.

## Actual user actions and observations

A real386byte PNG was chosen via Attach image and native file chooser. Composer displayed Attachments1 and a vision recommendation. Input: Keep the original blue image. WILLOW SAFE 0724. Actual Send to Ollama/gemma3:1b displayed an explicit image-support-not-confirmed alert with guidance. Original user pa_6783-8ca1-144-53ea and exactPNG retained. Retry same model kept the same user and bytes, with one assistant variant group before reload. Switch provider opened the actual model picker. Choosing configured Custom OpenAI Gemma then Retry produced the same truthful local refusal.

Normal reload retained the original user/image. Chat96409d59-1f81-44ce-ad7c-51d4142c65c8 canonical messages GET200 remained empty. Separate new chatf30a0b77-8eb5-43ae-a265-7cc0b62f2235 with image only also refused; local userpa_7188-7333-3a2-1432 retained the exact PNG and canonical GET200 total0. Full captured request inventory has zero Chat completionPOSTs. Conversation creation201 is expected; it does not claim user content was persisted remotely.

ImageSHA256: 314f71d711b39db674e4679f96307ef75f4da2ab93c3c8947d6e5c3ec1d0f1f7. Exact identity/network audit: cycle4-native122-final-integrity.json. Retry screenshot visually inspected. Final console error query returns zeroerrors; initial disconnected-browser warnings are not treated as a clean-start console claim.

## Newly recorded independent issues

UAT124 / TASK13260.64: before reload errors were one assistant with variants. After reload they become three separate error rows (originalpa_d6af-2303-5c9-dffa, retrypa_20e2-a070-ae0-c143, model-changepa_6fd7-026e-185-1b21). No duplicated user or server turn. Source investigation identifies a missing parent ID in general failure persistence. Original record remains evidence, not repaired by guessing.

UAT125 / TASK13260.65: immediate image-only error article contains a transient screen-reader Response complete announcement along with the explicit failure alert. Message.tsx needs to suppress success announcements for decoded errors. Snapshot/identity capture preserves this transient observation.

## Harness and scope limits

The original named browser unexpectedly closed immediately after a successful fill/request-observer command, before any test send. CLIinventory showed it absent; no product failure is inferred. Reopened only that session, reconnected via Settings/private key/Save/Test Connection (Core reachable), and began new chats. API/data remained unchanged, but the new browser lost the prior in-memory session. Saving connection replaced Settings refs; two stale-ref commands were rejected, then current snapshot controls succeeded. The installedCLI uses requests, while the skill reference's network command was unsupported; corrected to requests before send. None of these harness events count as application acceptance failures.

No provider claims confirmed vision, so successful real image inference and full118 canonical-image recovery remain unverified. Automated supported-image controls are separate. Historical unproven OCR images now refuse as explicitly documented by122 design; no nativeOCR-history pass claimed. No full freshcycle5 profile started. Both Next frontends paused after these observations for124/125 repairs.
