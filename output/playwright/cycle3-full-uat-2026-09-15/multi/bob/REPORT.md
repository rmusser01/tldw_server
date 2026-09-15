# Cycle3 multi-user Bob and admin acceptance evidence

Date: 2026-09-15, approximately 09:51–10:42 UTC. Browser session: `cycle3-multi-bob-20260915`. API: `http://127.0.0.1:18201`; UI: `http://127.0.0.1:18281`. Application behavior baseline: `d40e17dc81`; UAT-only Next filesystem-cache exception committed as `c10e1464fc` during the run. This is a bounded contribution to the full multi-user run, not full acceptance.

## Execution and environment

Bob was created through the admin UI by the main runner, then signed in through the normal UI in this separate persistent profile. Notes, Media, cards and reviews below were created through visible UI controls. Independent API reads and denied foreign updates reused the current browser bearer in memory, with the expected account header. They never logged in or seeded data. Real Flashcards generation used the configured unchanged llama.cpp service on9099; no mocked inference or API-created cards.

Two disk-exhaustion interruptions occurred before Bob's ingest submission and before Biology generation. Authorized frontend cache recovery preserved API/data/browser state. The final runtime disabled only the UAT Next filesystem cache. Bob reattached the queued file through Browse after reload; no incomplete ingest or generation was counted as success. An earlier API restart corrected the harness's typed provider spelling to the visible canonical `llama.cpp` without changing saved data.

## Bob-owned fixtures and permissions

- Private Note: `a80707b3-c761-4d8a-8bbe-fe2222886bc2`, title **Cycle3 Bob private note**, version1, account3. Actual UI Save201; list survives reload; reopening shows the exact Copper Finch body/sentinel. See `note-*.txt` and `own-controls.json`.
- Public synthetic Media: numeric1, UUID `1430f1e4-37e3-473e-b6d0-764f8f922f8d`, title **uat-cycle3-multi-bob-source**. UI upload with analysis/chunking off completed job3 in2seconds. `/media?id=1` displays exact `source.txt`; independent detail/list reads show Bob's content and document version1.
- Biology Note: `9ae43c6a-2458-4910-97fd-013d888914f7`, **Cycle3 Bob Biology study note**, version1. Its complete five-fact content is retained in `final-controls.json`; UI direct list selection after the backlink failure independently opens that same content (`biology-direct-open-awaited.txt`).

| Independent control | Result |
| --- | --- |
| Bob own private Note GET | 200, exact content/version1 |
| Bob own Media1 GET | 200, exact Copper Finch content/version1 |
| Bob own profile preferences GET | 200; preference values deliberately omitted |
| Ordinary `/admin/users` | 403, required role admin |
| Scheduled tasks/results | 403, missing `tasks.read` |
| Notes monitoring alerts | 403, missing `system.logs` |
| Alice Note `af0c68d3-69a1-4860-b369-4ccace5897ca` GET | 404 |
| Same Alice Note PUT with valid content and expected-version1 | 404; Bob own Note remains exact |
| Alice Media2 GET | 404 |
| Same Alice Media2 schema-valid content PUT | 404; Bob own Media1 remains exact |

Media identifiers are per-user database keys: Bob1 and Alice1 resolve to different owned sources. That collision is expected and was not treated as a foreign404 control. Alice's actual Media2 existed while Bob had only Media1. The Media PUT schema does not accept an expected-version field/header; only the Notes control is described as versioned.

The main Alice runner separately confirmed both originals unchanged: `/private/tmp/uat-cycle3-multi-alice-note-after-bob-denied.json` and `/private/tmp/uat-cycle3-multi-alice-indigo-after-bob-denied.json`. The latter is byte-for-byte identical to Alice's original GET body, document version1. Those files belong to the main runner and are referenced rather than duplicated here.

## Biology generation, saved pairs and study

Notes → Generate flashcards prefilled the exact five-fact Note. The initial count10 was explicitly changed to5. Provider/model fields stayed blank (server defaults); the real request specified5 basic cards and the response identifies generation provider `llama.cpp`. No generated wording was edited. The UI displayed and saved five distinct cards into **Cycle3 Bob Biology**, Bob's deck1. See `biology-generation-request.json`, `biology-generation-response.json`, `biology-drafts.txt` and saved-card captures.

| Actual generated question | Actual answer | Grounding |
| --- | --- | --- |
| What organelle is described as the powerhouse of the cell? | The mitochondria is the powerhouse of the cell. | Note fact1 |
| What does DNA stand for? | DNA stands for deoxyribonucleic acid. | Note fact2 |
| What does photosynthesis convert light energy into? | Photosynthesis converts light energy into chemical energy. | Note fact3 |
| How many bones does the human body have? | The human body has 206 bones. | Note fact4 |
| At what temperature does water boil at sea level? | Water boils at 100 degrees Celsius at sea level. | Note fact5 |

Each answer was visibly revealed and inspected, then graded **Good** once. Five answer snapshots and actual UI review requests303/334/372/399/424 (all200) are retained. Session-end437 returned200. This was a specific-deck run; it does not establish the separate mixed/all-decks UAT076 control.

Independent reads at10:31:07UTC (`final-controls.json`) show:

- Exactly5 cards, all `client_id=3`, deck1, source type `note`, source ID `9ae43c6a-2458-4910-97fd-013d888914f7`; all original question/answer pairs unchanged.
- Each card has repetitions1, lapses0, version3, learning state and due time exactly10minutes after its own last review.
- Card UUIDs: `bf82742b-f2b4-4321-8d30-ecc48f655427`, `99686df6-8d5b-4a0d-a945-6d365e5d79a1`, `8b11405b-78d3-4f4c-b897-4c638b5ebfe0`, `5e931ebb-89cb-40be-9fd7-2f4f0a39c2c2`, `d258321c-97b0-42f3-a428-b32384f789f0`.
- One completed session1, `scope_key=due:deck:1`, cards_reviewed5, started10:27:47.062UTC and completed10:30:15.251UTC.

Normal reload preserves the5-card deck, learning5, reviewed-today5 and completed five-card session (`biology-after-reload.txt`). Manage shows all5 source links (`biology-manage-reloaded.txt`). The source-link destination fails to select the Note, as recorded below; persisted source metadata alone is not counted as working navigation.

## Admin-owned deletion and recovery

After Bob evidence, normal Settings Logout and admin Login switched this same browser to verified account1 (`admin-authority.json`). The admin library started0/0, without Bob/Alice records. A fresh UI file upload (`admin-source.txt`), analysis/chunking disabled, completed job6, owner1. Admin's sole Media1 has UUID `3d0e3e25-338e-4ebf-8e80-41c57d47ced2`, title **uat-cycle3-multi-bob-admin-source**; its name reflects this runner's artifact prefix, while its exact synthetic content describes the admin's Silver Wren reading room.

Canonical `/media/capabilities` returned200 with `can_delete:true`. Visible Delete item → confirmation deleted that own item (DELETE493204). The library became0/0 and hid Trash: navigation fails UAT071. After retaining that failure, the direct `/media-trash` route allowed recovery; this workaround is not a navigation pass.

Trash showed the correct item and **Deleted: Sep15,2026,3:40AM**, matching API `deleted_at=2026-09-15T10:40:09.178Z`. Visible Restore succeeded. Reopening `/media?id=1` shows the exact original content. Independent reads at10:41:58UTC confirm the sole own item, identical detail body/document version1 and empty Trash (`admin-after-restore-controls.json`). No Alice or Bob namespace item was deleted. No permanent deletion was attempted.

## Findings and limits

- **UAT078:** Home calls denied scheduled-task endpoints then labels Automation Inbox temporarily unavailable; safe403 reasons are retained in `own-controls.json`. Backend denial is expected.
- **UAT079:** successful ordinary Notes save eagerly requests privileged monitoring alerts;403 missing `system.logs`. No save failure is claimed.
- Known063 recurs: a loaded version1 Note reports “No server save status yet.”
- **UAT064 recurs:** Notes→Flashcards places the entire Biology source, title and ID in the URL (`biology-handoff.txt`).
- **UAT080:** first Bob login09:51:23UTC reaches a failed refresh around10:21; safe401 reason “Invalid or expired refresh token.” Study displays a blocking “Can't reach your tldw server” modal caused by an aborted refresh and continued sessions401 polling. No cards were graded during that failed attempt. Normal UI re-login10:23:57 restored access; the existing saved cards were reviewed without regeneration. Backend transaction root cause is tracked by the parent; this bundle independently proves the visible failure and safe401 reason.
- **UAT081:** actual Manage→Note link navigates `/notes?source_ref_id=...`, then leaves the editor New note/empty after waiting for the saved title. Settled snapshot and request inventory retained. Subsequent normal Notes list selection opens exact saved content. The earlier independent GET in the inventory is not a route-loader request.
- **UAT082:** after Bob Logout/admin Login, opening Quick Ingest exposes Bob's prior completed filename and actions despite the admin's empty library. No stale action was invoked. Visible Ingest More then continued the admin fixture. Whether full reload alone clears the stale result was not tested before this intentional new ingest.
- **UAT071/072 recur** in admin deletion: hidden Trash entry after last-item deletion; AntD deprecated notification `btn` and context-free message errors. Exact console subset retained.
- Admin storage/quotas probes return403 “Email verification required”; this is preserved as an expected account verification restriction, separate from admin deletion permission.
- Manage also emits the existing AntD List deprecation. No recent-study-list regression is inferred from that different component.
- Reading sizeL was exercised on restored Media, but no progress PUT was emitted in the retained bounded observation; this does not establish a UAT073 recurrence or pass.

Harness-only noise: the first direct Bob API probe used credentials-include and encountered CORS; switching to standard bearer/credentials-omit fixed the diagnostic without changing application CORS. One admin diagnostic guessed `/media/1/can-delete` and received404; canonical `/media/capabilities` was then verified. Neither is counted as a product failure. Initial evicted/empty response bodies and an immediate pre-settlement Note read are excluded; later independent response reads are labeled as such.

`verification.json` records checks of the captured data, not substitute application tests. `SHA256SUMS` covers every retained artifact except itself. `secret-scan.json` documents credential scanning. No runtime private configuration, passwords, bearer tokens or raw auth response was copied. Final browser remains signed in as admin on its restored own Media1; runtime/data remain running and unchanged.
