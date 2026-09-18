# Native acceptance — World Books, Characters, Metadata, Sources

**CLEAR for bounded UAT264 / 267 / 271 / 272 / 273. UAT268 / 269 remain partial and blocked by UAT274. UAT275 remains open.**

Independent browser UAT used the original PostgreSQL multi-user profile, Alice2, Web18783/API18703, reviewed source `270682e98f2dd35c753fdf18cfb440ee80cb0ff1`. The audit passes **38 checks, 76 hashed inputs**. No inference was requested. This is not fresh full-matrix acceptance.

## Accepted observations

| Finding | Native evidence |
|---|---|
| 271, original loaded-version save | Original Character4 Save at03:33:27.939UTC sent `PUT /characters/4?expected_version=1`;200 at.973 returned version2. Name/instructions stayed unchanged. Original book1 attachment POST200 at03:33:28.037; normal reload/reopened Metadata at03:33:57 retained canonical enabled association and selected book. Captures04–07. |
| 272, reciprocal and disabled associations | Original book1 showed Attached Characters(2), original4 and6. Same-origin `/characters/` and association reads200; `enabled_only=false`. Character6 link update POST200 at03:35:46.568 disabled it; normal reload03:36:42 retained both reciprocal characters and a false settings switch. Reenable POST20003:37:05.389 and normal reload03:37:27 retained true. Captures08–16. |
| 273, unsupported metadata authoring | Original Character4 Metadata capabilities200 at03:32:49.176 returned `metadata_supported:false`; slots/resolver200. Clear authoring-unavailable guidance appeared and both ZIP import buttons were disabled. Zero pack requests and zero501 in full observed lease. Capture05 and final observer. |
| 264 / 267, authenticated Sources listing | Normal `/sources` at03:38:02 and normal reload03:38:29 requested same-origin Web18783 `/api/v1/ingestion-sources/`, authorization present,200 empty arrays. “No sources yet” and New source displayed. No Sources redirect/401/500, no persistent sign-in gate. Captures17–18. This does not qualify source creation or queued worker execution. |

## Partial lifecycle and new findings

Dedicated public book3 was created201 at03:39:18.086; public entry1 was created201 at03:39:54.435. The settled entry displayed the expected blue-lantern text. These are real normal-UI/backend results, not mocked responses.

**UAT274 — entry edit fails:** a single normal Edit Entry → Save Changes at03:41:10.973 sent `PUT /api/v1/characters/world-books/entries/undefined`, returning422 at.001 of the next second. Validation is `int_parsing`, location `path.entry_id`, input `undefined`, message “Input should be a valid integer, unable to parse string as an integer.” The editor stayed open. Canonical list rows expose `id:1`, not `entry_id`. After closing the unsaved form and normal reload, entry1 still contained the original blue text and unchanged modification timestamp. Captures21–28 preserve the successful create, failed edit and readback; there was no second save.

**UAT275 — stale parent count:** after creation, the parent catalogue/detail still showed zero entries while the settled Entries panel showed “Showing 1 of 1 entries” and18 tokens. This persisted into the failed-edit snapshot25. Normal reload showed canonical `entry_count:1` and visible1. Reload correction is not closure; the exact cache cause was not independently diagnosed here.

A separate public Character7 was created201 at03:44:27.700. Matrix attachment to book3 POST200 at03:47:35.222 persisted through normal reload, showing Attached Characters(1). Normal Detach DELETE `/characters/7/world-books/3` returned200 at03:48:49.825; reload showed zero links, canonical empty association. Captures29–40. Original book1 and book2 catalogue records stayed identical, Character6 remained version1, and its link was restored enabled.

**Entry and book deletion were not attempted after root ordered preservation of book3/entry1 for UAT274.** UAT268/269 therefore remain partial; successful backend create/attachment operations do not establish the full lifecycle. Dedicated Character7 remains detached. Book3 and its blue entry1 remain available for the repair repeat.

## Recovered background authentication observation

At03:45:29 three automatic GETs (`/auth/sessions`, `/buddies?limit=100&offset=0`, `/buddies/attachment?client_slot=default`) returned401. Two automatic refresh POSTs at.127 and.143 received one200 at.173 and one401 at.185. Automatic auth/me at.220 remained Alice2; sessions/buddies/attachment retries returned200. There was no manual login or observed failed user action, and subsequent attachment/detachment succeeded. No instantaneous UI snapshot at401 or auth error body was captured; no exact token-expiry/rotation cause is asserted.

Root's additive `root-auth-refresh-classification.json` classifies this as **no additional product defect established**. The existing winner-adoption recovery source matches the bound runtime source, and root's four unchanged winner/terminal/single-flight controls passed (56 deselected). The initial pnpm registry-resolution attempt was interrupted130 before tests, then the installed local Vitest ran successfully; both logs are retained. These controls support the recovery interpretation, not a new native expiry qualification. Full status timing remains in the safe audit.

## Honest harness record

All unsuccessful helpers remain on disk:

- 12 waited for a PUT signature; the real disable action uses POST and succeeded. No duplicate write.
- 24 waited for an assumed book-scoped entry path; the passive observer retained the real undefined-ID422.
- 26 waited for entry content while Edit selected Settings; corrected by the Entries tab in28.
- 31 tried to fill a read-only dropdown.33 End/Enter selected original4, **without submitting**.34 input click hit the selected-label overlay. After three unsuccessful selection attempts, this path stopped and the already observed Matrix was used.
- 36 `check()` asserted before the asynchronous state update; actual POST200 and checked/readback state were confirmed separately in37–38. No duplicate attachment.
- 41 listener cleanup used object iteration for an array;42 corrected cleanup successfully.

A few local reads were attempted before a pending CLI capture finished and returned no payload; only completed captures inform the audit. These are harness limitations, not concealed application passes. UI “character attachments unavailable” during initial query loading settled to zero; no persistent failure or429 occurred. The sole product422 plus the recovered background401s are all observed HTTP errors; no zero-error claim is made.

## Provenance, limits and handback

Startup proof at03:25:34.959 binds API67160/Next67372 to the reviewed source. The original profile, initialization and official PostgreSQL holder hashes match. The official runtime role remains non-superuser, without bypass-RLS/createDB/createRole. Runtime source files match the immutable prepared manifest; Character271/Metadata273 and Sources267 service match independently frozen review hashes. Actual Sources collections domain/service and WorldBook detail/chat-rag/characters domain files are included in runtime-to-manifest checks. Process receipts observed after handback still match startup bytes and cover the entire native interval; later root-owned shutdown may legitimately change receipts.

The observer recorded906 requests and906 responses; no completion/RAG/image-generation dispatch. Only normal existing browser UI actions were used. No response interception, auth injection, direct API writes, source/runtime/DB/Git/Backlog edits, new profile or inference. Private profile/credentials/process records were read only in memory and retained as hashes. Public audit projects safe facts rather than raw UI/provider/log content.

**Browser handed back explicitly at03:50:22.831UTC, with no pending command.** URL `/world-books`, book3 selected on Attachments, zero links. Own passive listeners removed; original matrix observer untouched. Root acknowledged ownership. Local captures remain available; a later hash-only packet is not a standalone native replay.
