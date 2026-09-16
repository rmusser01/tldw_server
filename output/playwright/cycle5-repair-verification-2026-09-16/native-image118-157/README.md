# Actual image attachment and failed-turn Retry

2026-09-16, approximately20:00–20:04UTC. Targeted SQLite single-user replacement profile, existing dependencies. This is not another full UAT. Source is82d61258f3 plus the held UAT103 production hashes in `source103-during-native.json`; those in-progress changes still require final independent review.

## Real browser evidence

1. Completed visible solo-local Setup, validated/saved the actual19099 provider and completed its first Chat. Entered only the generated isolated single-user key.
2. Saved a normal enabled vision profile with the actual discovered GGUF/projector assets and provider alias `llama.cpp/gemma-4-26B-A4B-it`. Autostart remained false; the app did not start/adopt the external process. Actual catalog reports vision=true with no warnings.
3. Used Chat's Attach image/file chooser to select `apps/tldw-frontend/public/icon.png`. Sent “Describe the attached image in one short sentence.” The real provider returned200 and “A white speech bubble containing three black dots.”
4. Stopped only root's test-owned19099 process. Attached the same PNG to “Count the black dots in this image. Answer with the number only.” Real completion returned502 `provider_unavailable`.
5. Reloaded the browser, restored the exact provider process, and clicked the actual Retry chat control. Request reused client ID `pa_bd73-5eec-382-7228`, set `metadata.tldw_retry_failed_turn=true`, and retained `data:image/png` with SHA256 `1792198785947731fc31e4c6184adeb00b988e703fc548893e3c1049a9b45453`. Real completion returned200 and “3”.
6. Another real reload retained the image and answer. The canonical listing contains exactly one user for that failed/retried question, ID `7e6b415e-cc06-474a-b3bc-95a97b30ac34`, with identical PNG bytes, and answer `8580bd3b-2f37-4bd9-8f21-d52516c51e4b`.

Saved conversation: `9127b6f8-7ed8-49c5-9258-7dcb00110138`. Screenshots were inspected. Raw request bodies, real response statuses, canonical opt-in image listing, catalog and source hashes are retained. SSE response bodies could not be recovered through Playwright's response-body API; this is explicitly a capture gap, not evidence that server acknowledgements were missing. The listener matched `/chat/`, so it did not capture separate `/chats/` fallback mutations. No response bodies were fabricated or substituted.

## New UAT157: successful image pair saved twice

The settled reload shows seven messages: system, two copies of the first successful user/assistant exchange, and the second user/Retry answer. The canonical listing proves this is stored duplication, not just a local rendering duplicate:

- Original image user `6cfcf0da-4d16-4728-be15-fb897b5a7c3d` at20:00:15.700; answer `23742a74-62e6-4722-94ba-ccd2e40c50ef` at20:00:21.412.
- Extra text-only user `56a5dbe0-e761-4f9f-84d6-13329fe2da3b` at20:00:21.480; identical answer `2d426942-16eb-4659-a252-4883448fb45f` at20:00:21.497.
- Second Send starts20:01:13.556. Thus the extra pair was created immediately after the first completion. The initial attribution to the later Send was incorrect and is superseded by these canonical timestamps.

Track separately underTASK13260.95. Diagnosis is pending; do not expand103's local retrieval-notice classification or deduplicate by equal text. Preserve these records for investigation.

## Limits and next gate

118's previously blocked real vision path now has a successful native text+image failure/reload/Retry control. Final reviewed-source acceptance and the required full single/multi SQLite/PostgreSQL matrix remain pending. This run does not claim an image-only native control or managed-runtime ownership. Existing9099 was untouched. The normal-runtime profile shares the repository's inactive system-ops maintenance baseline; broader isolated-install coverage requires that limitation to be resolved separately. Optional audio, RAG and MCP setup were deferred only for this targeted image check.
