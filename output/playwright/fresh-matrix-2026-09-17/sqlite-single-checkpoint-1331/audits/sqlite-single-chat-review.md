# SQLite single-user — independent Chat evidence audit

**Bounded acceptance:** row1's provider selection/first-chat/manual-key recovery and row3's text context, persistence and backend rejection/Retry claims are supported. The matrix's explicit image/visibility/upstream limitations are necessary and accurate. This is not a whole-row vision or upstream-outage pass, nor completion of the cell or matrix. Frozen revision `8f8774e6c868b304a96d95ab82e28389c129a78b` is the parent-declared matrix source; this evidence-only review does not independently rehash the archive/runtime.

## Row1

The ordinary Validate llama.cpp snapshot shows base URL `http://127.0.0.1:9099/v1`, an empty Default model field and the discovered Gemma GGUF ID. The later validation shows that exact selected ID; Save providers yields Saved and enables Continue. The actual first-chat POST at12:36:18.407Z returns200 at12:36:25.086Z with provider `llamacpp`, the same GGUF model and “It is fresh SQLite single-user ready.” These are UI discovery/validation observations plus the real first-chat response, not an independently captured `/models` exchange.

The first-chat result truthfully presents Restore media access because a working WebUI server key is still needed. `key-save-redacted.txt` records the disclosed key field and normal Save API key click with its value redacted. Ordinary Chat subsequently succeeds. This supports the manual recovery path, not automatic credential provisioning. UX231 remains separately recorded; the selected evidence does not certify a clean console or a clean-machine dependency installation.

## Row3 text/context/Retry

Conversation `45351b4c-b213-4827-b855-712ad2c75579` has successful ordinary completion responses at12:42:59.339Z and12:43:33.358Z. The second request includes the exact prior user message, assistant `ORBIT-742`, and the new question. UI and canonical readbacks show the expected answer; completion response records contain statuses, not retained SSE chunks, so no token-event claim is made.

The one-shot fault receipt changes only the outgoing provider/model to `ollama` / `uat-deliberately-unavailable-20260917`, continuing a real backend request without response fulfillment. The general request observer records the original pre-override body; the fault receipt and actual400 body establish the effective override. At12:45:03.543Z the backend returns `model_not_available`. This is a server model-availability rejection, **not an upstream generation outage**.

The visible Retry same model click sends the configured llama/Gemma request at12:45:31.240Z and receives200. Original messages, model, conversation and client-message identity `pa_408b-d409-713-5449` are retained; Retry intentionally adds `metadata.tldw_retry_failed_turn=true`. The whole request is therefore not byte-identical. No diagnostic error text enters its five-message context.

Explicit normal reload commands are supplied for both the two-turn chat and the recovered Retry. The first settled UI shows five canonical messages including the system prompt. The final canonical GET at12:45:54.368Z has **seven distinct IDs: one system, three users, three assistants**, with each assistant equal to `ORBIT-742`. The third user/answer pair appears once in canonical history. The local UI retains grouped variants (`2 of2`); that is not a duplicate canonical message.

| Canonical role | ID |
| --- | --- |
| system | `56b26624-dece-4e49-b9ab-3674d1e053d7` |
| user | `e8002565-3ccf-4fc6-834c-247838fc17c1` |
| assistant | `c8f1d076-5301-4a37-8647-3ce66e88e265` |
| user | `e9d40e27-f514-4e4c-8336-55775202bcf2` |
| assistant | `007be632-a085-4e7b-a1fd-5e15c642bc06` |
| user | `2fc86f2c-ef63-4051-b2e1-79cd68375c48` |
| assistant | `8190fd90-7190-4e2a-a887-86a02007a78f` |

## Image and visibility limits

The actual file chooser receives the archived public `icon/128.png`; the settled snapshot has an Uploaded Image element. The capability guard says image support is not confirmed for this model. No image completion request is present in the captured interval; the only completion requests are the four text requests described above. An empty server conversation was created for the image attempt, so “never sent” applies to the completion turn, not to all backend activity.

`image-reload.txt` records normal page.reload. The later approved `quick-ingest-snapshot.txt`, still in the original image chat, shows one user, Uploaded Image, and one displayed grouped assistant diagnostic at2of2. The canonical messages GET at12:47:46.080Z returns total0. This supports local attachment-element/error persistence after reload. Image-byte identity, successful image decoding/pixel appearance and vision inference are not certified by this text-only audit. No byte hash is required to support the narrower image-element claim. True-hidden visibility recovery and upstream inference failure remain untested here. UX232's redundant recovery choices remain separately recorded.

The additional approved provider/reload/snapshot receipts close the initial nine-file subset's evidence gaps. No further gap was found in the matrix's **bounded** row1/row3 paragraphs. Keep the backend400 qualification near any “provider failure” label; a full upstream-outage or vision claim would require separate evidence.

## Reviewed inputs and SHA256

Only these wrapper-redacted files and the matrix document were read. No credential/private file, raw browser output, browser, runtime, database or inference action was used. The sole reviewer write is this report. Input hashes bind the reviewed revision of the evolving matrix document; later row updates are outside this snapshot.

| Input | SHA256 |
| --- | --- |
| first-chat-result.txt | `8d2c5f95bdafd7ea70f82842c524f8b2dff6cafa471f66b295c7bcd162834724` |
| key-save-redacted.txt | `13d84b05d61f2567a89b191510acb4a01dce28869ff2b8ad54673ffb5d5492a2` |
| chat-turn1-result.txt | `00534a3c0ba9b542aa47d8a8b74a9a3a8484101c29ededa3fa944cc9c8997df9` |
| chat-turn2-send.txt | `4ee2a48733da6365802a122c4a3e54d52b2465400afb618b17831b53fc273fbc` |
| chat-retry.txt | `fdbf65c962233d6b40c3632897298031d9b925b63c6a8eeb5435b67a7ee96831` |
| chat-retry-final-evidence.txt | `0f4e45858cf77af78ed433a60f68c6d4e7accc800639136c3968a49fc807666c` |
| image-upload.txt | `507f8f3cfb28e54a9683b2c1ae6d0544d4aec4945521a287d7a8a04003fd12c5` |
| image-guard-settled.txt | `7b3dbefb7a629e77478db3d93b501ce0c0e63dd457e5c7f3379339891c50fe72` |
| image-final-evidence.txt | `00062434a63351dede73f4cbc1374ccfb4426f6d88f3a3dada6e90fa095f08a4` |
| provider-validate.txt | `223cd098e9cc9c00afe9db188329101817f1efc8c443583a3df9401337e9f307` |
| provider-selected-validate.txt | `8a454f974f919bf03a70afd96b0dae885f384a24444267bd73c7c6e49f5e4775` |
| provider-save.txt | `07ea3e748f69a5b69f801a82ddebd3b5716aa18c16d5d2d9edf27b46f19d60b9` |
| chat-two-turn-reload.txt | `c5d6e3392b096a64373922e866ad2ab804663762b5d9e6283ed72c4c58a03775` |
| chat-reload-settled.txt | `0a80f250dcf016eb43d5c92060b8dd9ae9686bd67a99e271eb7aaa1170b49a87` |
| chat-retry-reload.txt | `36052733e77987d677557d6912651061219a6678aee823cd7d4c8467f010bc65` |
| chat-retry-reload-snapshot.txt | `3968d46dcf9a7fa83b9e5681ea4b44cd89be06a7171881790a8678acd86f9317` |
| image-reload.txt | `cd515ec62a91e8208e03509e8520efc31824d4f472000d085c930f7ef69846dc` |
| quick-ingest-snapshot.txt | `b42932e6d884619b1bcf2b1d58bca8b8e45509c17bccb9f447306681c7ceba77` |
| Docs/Reviews/FRESH_INSTALL_UAT_MATRIX_2026_09_17.md | `c38ed27907e575a7b9cbd36bffe121ca037fea2799d3e356c1f0be956ea40531` |

Audited at 2026-09-17T12:57:37.778297+00:00.
