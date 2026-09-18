# Captured optional failures: classification supplement

Read-only follow-up; no browser, runtime or model calls. Original acceptance artifacts remain unchanged.

## World Books character catalogue429

At five ordinary World Books mounts/reloads/navigation events, the application automatically requested `GET http://127.0.0.1:18783/api/v1/characters`, received307, and followed to `http://127.0.0.1:18703/api/v1/characters/`, which returned429. Each group has one retry about a second later: ten429 responses in the review window. Captured bodies state `error=rate_limited`, `policy_id=character_chat.default`, `retry_after=1`.

This corrects the original report's loose “polling” description: it is bounded mount/reload query plus retry, not observed continuous polling. Helpers performed ordinary UI navigation/reload, not direct catalogue calls. The groups start02:15:19,02:17:40,02:18:06,02:19:26 and02:30:39 UTC. Earlier matching failures at01:49:20/21 are historical evidence only.

The failure is user-facing: at the final WorldBook1 Attachments view, the character catalogue cannot hydrate and the UI shows zero despite Character6's independently read-back association. Prepared source `WorldBooks/Manager.tsx` loads `listCharactersForWB` first and enables per-character association reads only when that list exists and is nonempty. This makes the catalogue failure relevant to **UAT272 / TASK13260.213**; root still owns causal diagnosis. No authorization headers were captured, so redirected credential loss is not established here. Earlier tasks12918 and13124.24 concern different chat/picker request storms; they are related history, not proof this exact path was fixed.

There is no native request to a `/world-books/1/characters` endpoint. The helper waited for an assumed endpoint that is not this implementation's contract; that timeout is a harness mistake. The actual false-zero UI and prerequisite catalogue failure are the product observations.

## PostgreSQL visual authoring501

Expanding Metadata for Character4 and reopening Metadata for Character6 automatically mounted Expression packs and dispatched four parallel reads. At02:21:21.735 and02:29:58.454, `GET http://127.0.0.1:18783/api/v1/visual-identities/packs?status=active` returned501. Sibling `/capabilities`, `/expression-slots`, and `/bindings/resolve?actor_kind=character&actor_id=4|6&expression_key=neutral` returned200.

No helper clicked Refresh, Import ZIP or Generate. The optional panel visibly displayed `visual_identity_metadata_backend_unsupported`; its pack-loading action failed. World Book edit and Character6 creation/attachment were still successful. No optional visual authoring was attempted or accepted.

This backend limitation is explicitly documented in **UAT208 / TASK13260.146** (Done): ordinary PostgreSQL resolution returns a supported placeholder200, while authoring/explicit override remains unsupported501. The current source dependency deliberately rejects PostgreSQL metadata service access with501. The panel loads capabilities and packs concurrently, so it does not use the returned `metadata_supported` flag to avoid that unsupported pack read. Classify this as the known unsupported-authoring boundary plus an exposed UI capability/error-presentation candidate; it is not evidence that ordinary-chat resolver208 regressed. No exact tracked panel-specific follow-up was found in this bounded search.

The companion JSON retains safe request/status facts, source and task hashes. No credentials, headers, private logs or provider content are serialized.
