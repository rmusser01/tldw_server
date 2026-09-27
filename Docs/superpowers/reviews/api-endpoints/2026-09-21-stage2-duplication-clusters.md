# Stage 2 — Duplication clusters inside the endpoints layer

## Scope

Axis 1 (duplication / utility consolidation) across `tldw_Server_API/app/api/v1/endpoints/`.
Covers the five seed clusters that touch this module (C9, C1, C3, C6, C8), the secondary
`(page - 1) * limit` cluster, and one cluster found during verification that the briefing did not
list (a third Discord/Slack file pair, plus the test-side clones).

Two seed claims were **corrected** rather than repeated — see "Corrections to the briefing" below.
Per the audit's forbidden-destinations rule, no proposal here grows `core/Utils/Utils.py` or
`core/http_client.py`.

## Code Paths Reviewed

C9 (Discord/Slack):
`discord_support.py` (677) / `slack_support.py` (678); `discord.py` (511, working tree) /
`slack.py` (600, working tree); `discord_oauth_admin.py` (407) / `slack_oauth_admin.py` (407);
`discord_support.py:_error_response (241-245)`, `:_metric_labels (248-254)`,
`:_verify_discord_signature (273-303)`, `:_check_discord_policy (~479-530)`,
`:_normalize_discord_policy_payload (~380-418)`; the Slack counterparts at the same line numbers.

C1 (base64 cursor/token):
`audio/audio_history.py:_encode_cursor (73-76)`, `:_decode_cursor (79-91)`, caller `:155-163`;
`audio/audio_jobs.py:_decode_audio_jobs_cursor (143-165)`;
`mcp_unified_endpoint.py:_parse_safe_config_query (128-158)`;
`character_messages.py:448`; `workflows.py:2192-2210` and `:2593-2602`;
`chat.py:_urlsafe_b64encode (6474)`, `:_urlsafe_b64decode (6478-6480)`,
`:_build_knowledge_qa_share_token (6483-6493)`, `:_decode_knowledge_qa_share_token (6496-6519)`;
`notes.py:_encode_attachment_cursor (~838-854)`, `:_decode_attachment_cursor (857-912)`.

C3 (datetime):
`api/v1/utils/datetime_utils.py:coerce_datetime (21-47)`, `:parse_timed_effects (50-76)`;
`_now_iso`/`_utc_now` at `admin/admin_acp_agents.py:69`, `agent_client_protocol.py:256`,
`chunking_templates.py:58`, `family_wizard.py:182`, `media/document_annotations.py:35`,
`media/reading_progress.py:27`, `notes_graph.py:175`;
`_parse_iso_datetime` at `billing.py:30-44`, `files.py:83-92`, `reading.py:360-369`,
`user_keys.py:367-385`; plus `chat.py:_parse_iso_datetime (6246)`,
`character_chat_sessions.py:_parse_iso_timestamp (2322)`,
`sync.py:_parse_iso8601_timestamp_utc (2517)`;
`core/File_Artifacts/file_artifacts_service.py:_build_export_info_from_row (770-792)`.

C6 (truncation):
`character_chat_sessions.py:_estimate_tokens (5644-5648)`, `:_truncate_to_budget (5651-5660)`,
caller `:5929-5960`, `:5964-5974`;
`core/Persona/exemplar_prompt_assembly.py:_estimate_tokens (51-55)`, `:_truncate_to_budget (58-65)`,
`:assemble_persona_exemplar_prompt (~120-173)`;
`core/VN_Assets/prompts.py:_truncate_to_budget (180-205)`.

C8 (error to HTTP):
`workspace_memberships.py:_membership_service_error_to_http (42-50)`;
`workspaces.py:_membership_service_error_to_http (789-797)`, `:_map_db_error (~778-786)`;
`persona.py:_persona_visual_service_error_to_http (2437-2450)`,
`:_persona_visual_library_service_error_to_http (2453-2466)`,
`:_persona_visual_starter_catalog_error_to_http (2469-2482)`, `:_to_http_exception (1790)`;
`embeddings_v5_production_enhanced.py:_policy_error_to_http (2291-2305)`,
`:_embedding_domain_error_to_http (3771)`; `llamacpp.py:_supervisor_error_to_http (184)`;
`api/v1/utils/http_errors.py:map_db_error_to_http (145)`;
`core/Persona/visual_starter_catalog.py:230,246` (the `starter_copy_failed` raises).

Pagination:
`api/v1/utils/pagination.py` (3 builders); `endpoints/_pagination_utils.py`;
`kanban/_kanban_utils.py:resolve_limit_offset (19-37)`; the 37 inline `(page - 1) * <size>` sites.

## Tests Reviewed

Import-grep reachability, not measured coverage.

- `tests/Discord/test_discord_command_routing.py` / `tests/Slack/test_slack_command_routing.py` —
  protect command parsing and job enqueue; both were edited in the working tree to assert the new
  admin gate on `GET /jobs/{job_id}` (401/403). **Downgrade the risk** of the routing path.
- `tests/Discord/test_discord_oauth_lifecycle.py` / `tests/Slack/test_slack_oauth_lifecycle.py` —
  the only coverage of the `*_oauth_admin.py` pair (import-grep on the module names returns 0; these
  reach it through the `discord.py`/`slack.py` routes). 84.9% clone of each other.
- `tests/Discord/test_discord_policy_hardening.py` / `tests/Slack/test_slack_policy_hardening.py` —
  protect the policy normaliser and quota enforcement. 60.6% clone.
- `tests/Integrations/test_discord_endpoint_sanitizers.py` /
  `tests/Integrations/test_slack_endpoint_sanitizers.py` — **100% identical after the rename**
  (30/30 lines).
- `tests/Chat/unit/test_chat_share_links_api.py:234-241` — `test_share_link_resolve_rejects_malformed_token`
  sends `"not-a-valid-token"`, which has no `.` and is rejected by the arity check at `chat.py:6498`.
  It does **not** reach the base64 decode, so it does **not** downgrade finding api-endpoints-1.
- `tests/Files/test_files_endpoint.py:173,221,728` — exercise expired and unexpired
  `export_expires_at` using `datetime.isoformat()` values. They cover the happy and expired paths,
  not the unparseable path, so they do not downgrade finding api-endpoints-11.
- Per-module counts: `chat` 78, `watchlists` 47, `workflows` 39, `character_chat_sessions` 31,
  `persona` 27, `notes` 24, `sync` 19, `workspaces` 16, `mcp_unified_endpoint` 12,
  `discord`/`slack` 8 each, `jobs_admin` 7, `character_messages` 6, `user_keys` 6, `files` 5,
  `reading` 5, `billing` 4, `discord_support`/`slack_support` 3, `workspace_memberships` 2,
  `audio_jobs` 2, `audio_history` 1, `discord_oauth_admin`/`slack_oauth_admin` 0 (reached
  indirectly). Full table in `2026-09-21-stage3-test-inventory.txt`.

## Validation Commands

```
$ wc -l tldw_Server_API/app/api/v1/endpoints/{discord,slack}*.py
     407 discord_oauth_admin.py
     677 discord_support.py
     511 discord.py            (working tree; +9 uncommitted)
     407 slack_oauth_admin.py
     678 slack_support.py
     600 slack.py              (working tree; +9 uncommitted)
    3280 total

$ python3 <clone-measure script>     # full output in 2026-09-21-stage2-c9-clone-measurements.txt
discord_support.py(677) vs slack_support.py(678)           : 533 matching = 78.6%  (ratio 78.7%)
discord.py(511)         vs slack.py(600)                   : 368 matching = 61.3%  (ratio 66.2%)
discord_oauth_admin.py(407) vs slack_oauth_admin.py(407)   : 371 matching = 91.2%  (ratio 91.2%)
test_discord_command_routing.py(180)  vs ...slack...(165)  :  53 matching = 29.4%
test_discord_oauth_lifecycle.py(255)  vs ...slack...(279)  : 237 matching = 84.9%
test_discord_policy_hardening.py(350) vs ...slack...(326)  : 212 matching = 60.6%
test_discord_endpoint_sanitizers.py(30) vs ...slack...(30) :  30 matching = 100.0%

$ python3 -c "import pathlib; a=...; b=...; print(sum(1 for x,y in zip(a,b) if x==y and len(x.strip())>40))"
same-line-number byte-identical non-trivial lines: 23 of 677

$ git diff --stat -- .../endpoints/{discord,slack}.py .../tests/{Discord,Slack}/test_*_command_routing.py
 discord.py | 9 ++++++++-   slack.py | 9 ++++++++-
 test_discord_command_routing.py | 7 +++++--   test_slack_command_routing.py | 7 +++++--
 4 files changed, 26 insertions(+), 6 deletions(-)

$ grep -n 'requires-python' pyproject.toml
15:requires-python = ">=3.11"
$ python3 -c "import datetime; print(datetime.datetime.fromisoformat('2026-09-21T10:00:00Z'))"
2026-09-21 10:00:00+00:00

$ python3 -c "import base64; print(base64.urlsafe_b64decode('abc'+'='*(-3%4)))"
    # both workflows padding idioms round-trip identically for all tested lengths (12..27)

$ rg -n 'offset\s*=\s*\(?\s*page\s*-\s*1\s*\)?\s*\*' .../endpoints/watchlists.py | wc -l
      12
$ rg -n 'page\s*-\s*1\s*\)\s*\*' .../endpoints/ | wc -l
      37
$ rg -l 'api\.v1\.utils\.pagination' tldw_Server_API/app | wc -l
      20
$ rg -l '_pagination_utils' tldw_Server_API/app | wc -l
      62
$ rg -l 'utils\.datetime_utils' tldw_Server_API/app
tldw_Server_API/app/api/v1/endpoints/chat_dictionaries.py          # 1 importer
$ rg -l 'utils\.http_errors' tldw_Server_API/app | wc -l
      56
```

## Corrections to the briefing

Two seed claims did not survive verification. Recording them so they are not re-asserted:

1. **C3 / `files.py:83` does NOT fail on `Z`-suffixed input.** The briefing states it "omits the `Z`
   replace and therefore FAILS on `Z`-suffixed inputs". `pyproject.toml:15` sets
   `requires-python = ">=3.11"`, and `datetime.fromisoformat` has accepted the `Z` designator since
   3.11. Verified on the CI interpreter (3.12.11): `fromisoformat('2026-09-21T10:00:00Z')` returns
   `2026-09-21 10:00:00+00:00`. The real defect at that site is different and is recorded as
   api-endpoints-11.
2. **C1 / `workflows.py:2195`'s `+ b"=="` padding is not a bug.** It differs in idiom from
   `workflows.py:2597`'s `"=" * (-len(cursor) % 4)`, but CPython's `binascii.a2b_base64` tolerates
   excess padding; both round-trip identically for every cursor length tested (12-27 chars, all
   `len % 4` classes). Reported below as an idiom divergence, not a defect.

## Findings

### FINDING api-endpoints-3 — Discord/Slack: three source clone pairs and four test clone pairs, ~4,900 lines

```
axis:        duplication
class:       true-duplication
severity:    High
sites:       SOURCE
             discord_support.py (677) == slack_support.py (678) after the rename: 533/678 lines
               matching = 78.6%. Byte-identical at the SAME line numbers:
               _error_response (:241-245) and _metric_labels (:248-254); 23 non-trivial lines
               byte-identical at the same line number overall.
             discord_oauth_admin.py (407) == slack_oauth_admin.py (407): 371/407 = 91.2%.
               NOT in the briefing — found during verification.
             discord.py (511) == slack.py (600): 368/600 = 61.3%.
             TESTS
             tests/Integrations/test_discord_endpoint_sanitizers.py ==
               tests/Integrations/test_slack_endpoint_sanitizers.py: 30/30 = 100.0%
             tests/Discord/test_discord_oauth_lifecycle.py == tests/Slack/...: 237/279 = 84.9%
             tests/Discord/test_discord_policy_hardening.py == tests/Slack/...: 212/326 = 60.6%
             tests/Discord/test_discord_command_routing.py == tests/Slack/...: 53/180 = 29.4%
canonical:   NONE
destination: `app/api/v1/endpoints/_chatops/` — one responsibility: the transport-agnostic ChatOps
             integration shell. It owns exactly the parts that are identical: the policy document
             schema + normaliser + quota enforcement, the installation record shape, the OAuth
             install/callback/list/disable state machine, the receipt/dedupe store wiring, the
             metric-label and error-envelope helpers, and the signature-verification *scaffold*
             (timestamp extraction, staleness window, the `(ok, reason)` contract). The two
             protocol-specific pieces stay in `discord_*`/`slack_*` as injected strategies: the
             signature algorithm (Ed25519 vs HMAC-SHA256 `v0=`) and the command parser (interaction
             options tree vs slash-command form / app_mention text). It must NOT be folded into
             `core/Utils/Utils.py` or `core/http_client.py`.
knowledge:   The ChatOps policy contract. The policy document is normalised twice under two
             vocabularies that have already drifted:
               team_quota_per_minute      vs workspace_quota_per_minute
               status_scope {team, team_and_user} vs {workspace, workspace_and_user}
               default_response_mode {ephemeral, channel} vs {ephemeral, thread, channel}
             Adding a policy field, a quota dimension, or a response mode requires editing two
             normalisers, two enforcement blocks, two default-policy dicts, two env-var readers and
             two test suites — and the divergent vocabulary means a copy-paste fix silently writes
             the wrong key.
scenario:    The change-amplification cost is already visible in the uncommitted working tree. The
             IDOR fix on `GET /{discord|slack}/jobs/{job_id}` — adding
             `dependencies=[Depends(RequireRole("admin"))]` plus a 4-line rationale comment — had to
             be written four times: discord.py:355-365, slack.py:449-459,
             tests/Discord/test_discord_command_routing.py:74-79,
             tests/Slack/test_slack_command_routing.py:63-68, with byte-identical comment text in
             all four. That is one security fix costing four edits, and it only stayed consistent
             because the author remembered the sibling.
impact:      High. 3,280 source lines + 1,615 test lines with 61-100% redundancy, and the divergent
             28% is where the bugs live: the two modules have already drifted on the policy
             vocabulary (above), on bot-loop suppression (`slack_support.py:_is_bot_event` has no
             Discord counterpart), on dedupe scope (Discord has one `_INTERACTION_RECEIPTS`, Slack
             has `_EVENT_RECEIPTS` + `_COMMAND_RECEIPTS`), and on OAuth field capture (Slack records
             enterprise_id/bot_user_id/authed_user_id, Discord records refresh_token). Each of those
             is defensible on its own, but nothing in the code says which differences are
             deliberate.
tests:       Import-grep reachability, not measured coverage. discord 8 files / slack 8 files /
             discord_support 3 / slack_support 3 / *_oauth_admin 0 direct (covered indirectly by the
             oauth_lifecycle pair). Coverage is symmetric and good, which makes the extraction
             cheaper than the line count suggests — but the test suites are themselves clones, so
             the extraction must collapse them too or the duplication simply moves.
effort:      Expensive. Needs `Docs/Design/YYYY-MM-DD-chatops-shell-design.md`, an ADR (this is a
             decision about where integration policy lives), a Backlog task linking both, and
             `IMPLEMENTATION_PLAN_chatops-shell.md` with 3-5 stages. Suggested stage 1 is the
             91.2% pair (`*_oauth_admin.py`) — smallest, highest identity, already has the
             oauth_lifecycle tests on both sides.
owner-only:  yes
confidence:  confirmed (the measurements, the byte-identical helpers, the divergent vocabulary, the
             four-way working-tree fix); probable-risk (that a future policy edit lands on one side
             only — no instance of that was found in history)
```

Divergences worth recording separately, all currently justified, none documented in the code:

| Divergence | Discord | Slack | Verdict |
| --- | --- | --- | --- |
| Signature | Ed25519 over `timestamp+body` (`discord_support.py:273-303`) | HMAC-SHA256 `v0:ts:body` (`slack_support.py:273-297`) | justified-divergence — protocol |
| Unconfigured-key log | silent | `logger.warning(...)` (`slack_support.py:~276`) | drift, not a defect |
| Bot-loop guard | none | `_is_bot_event` (`slack_support.py:~310-318`) | justified — Discord interactions cannot originate from a bot the same way |
| OAuth `ok` check | none | `token_payload["ok"]` -> 502 (`slack_oauth_admin.py:~148-153`) | justified — `ok` is a Slack response field |
| Install-key source | falls back to the `guild_id` **query parameter** (`discord.py:409,415`; `discord_oauth_admin.py:~163-170`) | token payload only | justified — Discord documents `guild_id` on the redirect — but it is the one place where an attacker-influenced query value names the installation record. Worth an explicit comment. |
| `key_hint_token` | may be `None` | coerced to `""` (`slack_oauth_admin.py:~382`) | drift; check `upsert_secret`'s `None` handling |

### FINDING api-endpoints-9 — four incompatible conventions for "core service error -> HTTP" inside endpoints, one of them dropping a real error code

```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       Convention A (passthrough of exc.status_code, detail = {code, message, details}) —
               workspace_memberships.py:_membership_service_error_to_http (42-50)
               == workspaces.py:_membership_service_error_to_http (789-797)  [BYTE-IDENTICAL, 9 lines]
             Convention B (code-set -> status table, detail = {code, str(exc), details}) —
               persona.py:_persona_visual_service_error_to_http (2437-2450)
               persona.py:_persona_visual_library_service_error_to_http (2453-2466)
               persona.py:_persona_visual_starter_catalog_error_to_http (2469-2482)
             Convention C (code -> status with a metrics side effect, detail = bare string) —
               embeddings_v5_production_enhanced.py:_policy_error_to_http (2291-2305)
               embeddings_v5_production_enhanced.py:_embedding_domain_error_to_http (3771)
             Convention D (canonical) — api/v1/utils/http_errors.py:map_db_error_to_http (145),
               56 importers, used by workspaces.py:778-786 in the same file as convention A
             Also: llamacpp.py:_supervisor_error_to_http (184), persona.py:_to_http_exception (1790)
canonical:   api/v1/utils/http_errors.py:map_db_error_to_http (145) — 56 importers, already adopted
destination: Extend the existing `api/v1/utils/http_errors.py` (single responsibility: translating
             domain exceptions to HTTPException; it is 284 lines, not a junk drawer) with one
             `map_service_error_to_http(exc, *, code_status: Mapping[str, int], default: int)`.
             Convention B's three copies then differ only in the mapping literal, which is the
             point. Convention A's two byte-identical copies collapse to zero code.
knowledge:   The error response body shape and the code->status policy. Today `detail` is a dict in
             conventions A and B and a bare string in convention C, so a client parsing
             `response.json()["detail"]["code"]` works against workspaces and persona and breaks
             against embeddings. `api-response-envelope/2026-04-25-helper-contract-spec.md` owns the
             convergence design for exactly this; it never enumerated the instances. This is the
             enumeration.
scenario:    `core/Persona/visual_starter_catalog.py:230` and `:246` raise
             `PersonaVisualStarterCatalogError("starter_copy_failed", ...)` when an optimistic-
             concurrency update returns falsy — a version conflict or a failed write.
             `persona.py:_persona_visual_starter_catalog_error_to_http (2469-2482)` enumerates
             starter_pack_not_found / starter_asset_not_found / target_persona_not_found /
             invalid_starter_fixture / duplicate_starter_fixture / invalid_starter_asset /
             invalid_starter_manifest / persona_id_required / user_id_required, but NOT
             starter_copy_failed. It therefore falls through to the function's default,
             `status.HTTP_400_BAD_REQUEST`. A concurrent-write conflict on the server is reported to
             the client as "your request was malformed", where the sibling mappers in the same file
             would have returned 409. A retrying client sees 400, treats it as permanent, and gives
             up on a conflict that would have succeeded on retry.
impact:      Medium. The byte-identical pair is trivial to fix and trivially safe; the value is in
             the enumeration, which surfaced the starter_copy_failed misclassification and the
             dict-vs-string `detail` split that makes a uniform client error handler impossible.
tests:       Import-grep reachability, not measured coverage: `workspaces` 16 files,
             `workspace_memberships` 2, `persona` 27, `embeddings_v5_production_enhanced` covered
             via the Embeddings suites. No test asserts the status code for `starter_copy_failed`.
effort:      Cheap for convention A (delete one copy, import the other — or better, move the mapper
             next to `WorkspaceMembershipServiceError` in core). Moderate for B and C: each needs
             its response-shape change coordinated with the api-response-envelope plan.
             The starter_copy_failed mapping is a one-line fix and should not wait for the rest.
owner-only:  yes
confidence:  confirmed (the four conventions, the byte-identical pair, the missing code); probable-
             risk (that a client is actually harmed by the 400 today)
```

### FINDING api-endpoints-8 — persona preview re-truncates with a divergent copy and then reports `truncated: false` on content the core assembler already cut

```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       character_chat_sessions.py:_estimate_tokens (5644-5648),
               :_truncate_to_budget (5651-5660), caller loop (5929-5960)
             core/Persona/exemplar_prompt_assembly.py:_estimate_tokens (51-55),
               :_truncate_to_budget (58-65), :assemble_persona_exemplar_prompt (120-173,
               truncation applied at :157 and :167)
             core/VN_Assets/prompts.py:_truncate_to_budget (180-205)
canonical:   core/Persona/exemplar_prompt_assembly.py:_truncate_to_budget (58) — the endpoint file
             already imports from this exact module at character_chat_sessions.py:235
destination: n/a — adoption, not a new module. The correct copy is the Persona one (it clamps with
             `max(1, token_budget * 4)` and `.rstrip()`s before appending the ellipsis); the
             endpoint copy should be deleted.
knowledge:   "How many characters fit in N tokens, and what a truncated string looks like."
             Three answers live in the repo under one name:
               exemplar_prompt_assembly.py:58  (str, int) -> str,  4 chars/token, rstrip, max(1,..)
               character_chat_sessions.py:5651 (str, int) -> str,  4 chars/token, NO rstrip, NO clamp
               VN_Assets/prompts.py:180        (str, int) -> tuple[str, int], real tokenizer +
                                                binary search on word boundaries
             The third already migrated to a real tokenizer. When Persona follows, the endpoint's
             preview silently keeps approximating.
scenario:    `character_chat_sessions.py:5916` calls `_build_persona_preview_assembly`, which calls
             the core `assemble_persona_exemplar_prompt`. That function ALREADY truncates each
             section (`exemplar_prompt_assembly.py:157,167`) and returns
             `(name, truncated_content, budget)` tuples. The endpoint appends those tuples to
             `sections_raw` at :5936 and, at :5943, re-runs its own `_truncate_to_budget(text, budget)`
             on the already-truncated text with the same budget. Because core's output is
             `budget*4 + 3` characters and `_estimate_tokens` is `len // 4`, the second pass is a
             no-op — so at :5953 the endpoint computes
             `"truncated": bool(text) and truncated_text != text` as **False**. Concrete: a persona
             boundary section whose raw content exceeds `_PERSONA_BOUNDARY_SECTION_BUDGET` comes back
             from `GET`-ing the preview with `content` visibly ending in `"..."`, `tokens_estimated
             == tokens_effective == budget`, and `truncated: false`. A client that renders a
             "content was trimmed" warning off that flag never shows it.
impact:      Medium. The endpoint is a *preview* of the prompt the runtime will assemble; a preview
             that under-reports truncation is worse than no preview. The latent half — the missing
             `max(1, ...)` clamp, which for a negative budget makes `text[:budget*4]` slice from the
             END and return nearly the whole string — is not reachable today (all budgets are the
             module constants `_TOKEN_BUDGET_*` at :5457-5463, all positive) and is recorded as
             latent, not as a defect.
tests:       Import-grep reachability, not measured coverage: `character_chat_sessions` 31 files.
             The characters-backend ledger (2026-03-23, rebaselined 2026-04-15) reviewed this file
             but not this code path. No test asserts the `truncated` flag for a persona section.
effort:      Cheap. Delete `character_chat_sessions.py:5644-5660`, import `_estimate_tokens` and
             `_truncate_to_budget` from the module the file already imports at :235, and drop the
             redundant second truncation pass so `truncated` reflects what core did. Well covered
             at the route level.
owner-only:  yes
confidence:  confirmed (the double pass, the no-op, the resulting `truncated: false`); assumption
             (that a client reads the flag)
```

### FINDING api-endpoints-7 — four divergent `_parse_iso_datetime` copies in endpoints, and a canonical datetime helper that is unadoptable by construction

```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       billing.py:_parse_iso_datetime (30-44)      — accepts datetime passthrough, str()
                                                            coercion, catches (ValueError, TypeError)
             files.py:_parse_iso_datetime (83-92)        — str only, bare `except Exception`
             reading.py:_parse_iso_datetime (360-369)    — global `.replace("Z", "+00:00")`,
                                                            catches ValueError only
             user_keys.py:_parse_iso_datetime (367-385)  — type guard first, TRAILING-only Z strip,
                                                            catches ValueError
             plus differently-named siblings: chat.py:_parse_iso_datetime (6246),
               character_chat_sessions.py:_parse_iso_timestamp (2322),
               sync.py:_parse_iso8601_timestamp_utc (2517)
             and in core, the same-shaped copy that pairs with files.py:
               core/File_Artifacts/file_artifacts_service.py:_parse_iso_datetime
                 (used at :773, :776)
             _now_iso/_utc_now, verbatim `return datetime.now(timezone.utc).isoformat()`:
               admin/admin_acp_agents.py:69, agent_client_protocol.py:256,
               chunking_templates.py:58 (with a function-local re-import),
               family_wizard.py:182, media/document_annotations.py:35,
               media/reading_progress.py:27, notes_graph.py:175
canonical:   api/v1/utils/datetime_utils.py:coerce_datetime (21-47) — nominally the shared helper.
             ONE importer repo-wide (endpoints/chat_dictionaries.py).
destination: A new `app/api/v1/utils/iso_datetime.py` with one responsibility: parsing and emitting
             ISO-8601 instants at the API boundary — `utc_now_iso()` and
             `parse_iso_utc(value) -> datetime | None`. It must NOT be added to
             `datetime_utils.py` as it stands, because that module imports
             `api/v1/schemas/chat_dictionary_schemas.TimedEffects` at module level for its second
             function — a "shared datetime util" coupled to one feature's schema, which is why it
             has one importer. Splitting `parse_timed_effects` out is a precondition, not an
             afterthought.
knowledge:   "What counts as a parseable timestamp, and what happens when it isn't one." Four
             answers: accept a `datetime` object or not; strip `Z` globally, only at the end, or
             not at all; catch `Exception`, `ValueError`, or `(ValueError, TypeError)`. Every new
             timestamp column added to an API response picks one of the four at random.
scenario:    The canonical helper is itself wrong for its stated contract. `coerce_datetime`
             (datetime_utils.py:21-47) documents "conversion ... to a timezone-aware datetime", but
             tries `strptime("%Y-%m-%dT%H:%M:%S")` BEFORE `fromisoformat`, so
             `coerce_datetime("2026-09-21T10:00:00")` returns a **tz-naive** datetime while
             `coerce_datetime("2026-09-21T10:00:00+00:00")` returns a tz-aware one — output
             awareness depends on input format. Its final fallback returns
             `datetime.now(timezone.utc)`, silently substituting the current time for an
             unparseable input rather than signalling failure. Adopting it as-is would spread both
             behaviours; that is the reason the destination above is a new module and not this one.
impact:      Medium. This is the adoption-gap the briefing predicted, but inverted: the designated
             helper is not merely bypassed, it is unadoptable (wrong contract, feature-coupled
             import). Reporting it as "60 sites bypass datetime_utils" would have recommended
             spreading a bug.
tests:       Import-grep reachability, not measured coverage: `billing` 4, `files` 5, `reading` 5,
             `user_keys` 6, `chat_dictionaries` covered via the Chat suites. No test pins the
             tz-awareness contract of `coerce_datetime`.
effort:      Moderate. The `_now_iso` half is mechanical (7 identical one-liners). The parse half
             needs a decision on the single contract first, and `chunking_templates.py:58`'s
             function-local `from datetime import ...` re-import should go with it.
owner-only:  yes (all cited sites are under api/v1/)
confidence:  confirmed (the four divergent copies and the coerce_datetime behaviour); probable-risk
             (that any specific pair of copies produces a user-visible difference today)
```

### FINDING api-endpoints-10 — one base64 cursor/token idiom, eight hand-written copies, six different error contracts, across two trust classes

```
axis:        duplication
class:       true-duplication
severity:    Medium
sites:       OPAQUE PAGINATION CURSORS
               audio/audio_history.py:_encode_cursor (73-76), :_decode_cursor (79-91)
                 — no try inside the helper; the caller wraps it at :155-163 -> 400
               audio/audio_jobs.py:_decode_audio_jobs_cursor (143-165)
                 — catches an 8-exception tuple incl. binascii.Error, re-raises ValueError
               workflows.py:2192-2210  — padding `+ b"=="`; swallows into logger.debug and
                 SILENTLY IGNORES the cursor, resetting offset to 0
               workflows.py:2593-2602  — padding `"=" * (-len % 4)`; swallows into logger.debug and
                 silently ignores the cursor, replaying events from `since`
               mcp_unified_endpoint.py:_parse_safe_config_query (128-158)
                 — the strictest: `validate=True`, explicit binascii.Error catch, structured 400
               character_messages.py:448 — image data URI, not a cursor; `validate=True` at :461
                 with a size preflight at :455-459
             SIGNED / CRYPTO TOKENS (different trust class)
               notes.py:_encode_attachment_cursor (838-854),
                 :_decode_attachment_cursor (857-912) — 512-byte cap -> 413, schema pre-check,
                 canonical-base64 re-encode check, hmac.compare_digest, payload bound to
                 owner/dataset/note/state, version field
               chat.py:_urlsafe_b64encode (6474), :_urlsafe_b64decode (6478-6480),
                 :_build_knowledge_qa_share_token (6483-6493),
                 :_decode_knowledge_qa_share_token (6496-6519) — none of the above; see
                 api-endpoints-1
canonical:   NONE
destination: `app/api/v1/utils/opaque_cursor.py`, one responsibility: encode/decode a versioned,
             length-bounded, canonically-encoded opaque pagination cursor, raising one typed error
             the endpoint maps to 400. It must expose ONLY the unsigned form. The signed-token
             sites (`notes.py`, `chat.py`) must NOT be folded in: flattening "opaque pagination
             cursor" and "HMAC-signed capability token" into one helper is how a future caller gets
             an unsigned cursor where a signed one was required. If the signed form is shared at
             all it belongs beside the HMAC key derivation in `core/AuthNZ/`, not here.
knowledge:   The wire format (padding rule, canonicality, version field, max length) and the failure
             contract. Today "malformed cursor" means 400 in three places, "silently start over" in
             two, 413-or-400 in one, and 500 in one (api-endpoints-1).
scenario:    `workflows.py:2192-2210` and `:2593-2602` are the reportable behaviour: a corrupted or
             truncated cursor is caught by `_WORKFLOWS_NONCRITICAL_EXCEPTIONS`, logged at debug, and
             **ignored**. A client paging `GET /workflows/runs` whose cursor is mangled in transit
             gets HTTP 200 with page 1 again instead of a 400, and an event consumer at :2593 silently
             re-reads from `since` — an unbounded replay rather than an error. The two are in the
             same file and disagree only on the padding idiom, which shows nobody reconciled them.
impact:      Medium. No data exposure; the cost is a silent-restart paging contract in `workflows.py`
             and seven places to edit when the cursor format gains a version or a size cap — which
             `notes.py` and `mcp_unified_endpoint.py` have already independently decided it needs.
tests:       Import-grep reachability, not measured coverage: `workflows` 39, `notes` 24,
             `mcp_unified_endpoint` 12, `character_messages` 6, `audio_jobs` 2, `audio_history` 1.
effort:      Moderate. `mcp_unified_endpoint.py:128-158` is the model to promote (validate=True +
             typed 400). `notes.py:857-912` is the model for the signed class and should stay
             separate.
owner-only:  yes
confidence:  confirmed (the eight sites and six contracts); confirmed (the workflows silent-ignore
             behaviour, read directly)
```

### FINDING api-endpoints-13 — page-to-offset arithmetic inlined at 37 sites with a canonical pagination module that has no resolver

```
axis:        duplication
class:       true-duplication
severity:    Low
sites:       watchlists.py:748, 2164, 3258, 3300, 3924, 4475, 4525, 4598, 6535, 7812, 8471, 8601 (12)
             paper_search.py:290, 347, 1666, 1752, 1826, 1933, 1993, 2051, 2513, 2612, 2694, 2769,
               2846, 4856 (14)
             character_chat_sessions.py:8030, 8867 (2)
             research.py:135, 255 (2)
             collections_feeds.py:478 (`max(0, ...)`), reading.py:562 (`max(0, ...)`),
             personalization.py:185, characters_endpoint.py:968, media/versions.py:120,
             media/listing.py:1128, admin/admin_user.py:165,
             evaluations/evaluations_embeddings_abtest.py:402,
             kanban/_kanban_utils.py:resolve_limit_offset (19-37) (9)
canonical:   NONE for the request side. `api/v1/utils/pagination.py` (20 importers) owns only the
             RESPONSE side (`build_offset_pagination_meta`, `build_cursor_pagination_meta`,
             `build_page_pagination_meta`); `endpoints/_pagination_utils.py` (62 importers) owns
             Link headers and `resolve_page_pagination_metadata`. Neither converts page -> offset.
             `kanban/_kanban_utils.py:resolve_limit_offset (19-37)` is the only resolver and is
             imported by exactly the 4 kanban modules.
destination: Add the request-side resolver to `api/v1/utils/pagination.py` — it is already the
             cohesive pagination module with a single responsibility, it is already adopted, and
             `_pagination_utils.py` already re-exports from it. Promote the kanban helper there
             rather than writing a fifth one.
knowledge:   The one-based-page to zero-based-offset convention and its clamp.
             `api-pagination/2026-04-25-helper-contract-spec.md` already states the rule
             ("`page` aliases are one-based and convert to zero-based offset") and the alias
             precedence ladder. This finding is the per-site enumeration that ledger explicitly did
             not do; it is not a rediscovery of the contract.
impact:      Low, and deliberately so. Every one of the 37 sites declares `page: int = Query(1, ge=1)`
             (verified across watchlists, paper_search, characters_endpoint,
             character_chat_sessions, personalization), so the two clamped sites
             (`collections_feeds.py:478`, `reading.py:562`) are belt-and-braces, not bug fixes for
             the other 35. There is no reachable negative-offset path today. The cost is purely the
             35 edits a convention change would require.
tests:       Import-grep reachability, not measured coverage: `watchlists` 47, `paper_search`
             covered via the Research/Search suites, `kanban` via tests/Kanban.
             `api-pagination/2026-04-25-route-family-catalogue.md:48` lists `watchlists (12)` in the
             Hybrid bucket as inventory with no findings; `auth-dependencies` puts it top of the
             legacy-user-dependency table at 64 signals.
effort:      Cheap per site, but it should ride the api-pagination Phase 3.2 plan rather than land
             as a standalone sweep — that ledger explicitly says "Avoid first: media/listing,
             paper_search, admin/*".
owner-only:  yes
confidence:  confirmed
```

### FINDING api-endpoints-16 — seven verbatim `_now_iso` copies and 60 `datetime.utcnow()` calls in endpoints

```
axis:        duplication
class:       adoption-gap
severity:    Low
sites:       Verbatim `return datetime.now(timezone.utc).isoformat()`:
               admin/admin_acp_agents.py:69, agent_client_protocol.py:256,
               chunking_templates.py:58 (plus a function-local `from datetime import ...`),
               family_wizard.py:182, media/document_annotations.py:35,
               media/reading_progress.py:27, notes_graph.py:175 (named `_utc_now`)
             `datetime.utcnow()` by file: auth.py 12, watchlists.py 5, workflows.py 4,
               prompt_studio/prompt_studio_evaluations.py 4, outputs.py 4,
               embeddings_v5_production_enhanced.py 4, audio/audio_health.py 4, users.py 3,
               prompt_studio/prompt_studio_websocket.py 3, prompt_studio/prompt_studio_optimization.py 3,
               notes.py 3, characters_endpoint.py 3, outputs_templates.py 2, voice_assistant.py 1,
               prompts.py 1, llm_providers.py 1, jobs_admin.py 1, health.py 1, audio/audio_tts.py 1
               (60 total)
canonical:   NONE that is usable — see api-endpoints-7 on why
             `api/v1/utils/datetime_utils.py` is not it.
destination: `app/api/v1/utils/iso_datetime.py:utc_now_iso()` — same destination as api-endpoints-7.
knowledge:   Whether API timestamps are tz-aware and at what precision. `datetime.utcnow()` returns
             a tz-NAIVE value; `datetime.now(timezone.utc).isoformat()` returns `+00:00`. The repo
             already has a verified instance of this exact split causing incomparable timestamps
             between two sibling services (`app/services/workflows_webhook_dlq_service.py:45` vs
             `app/services/meetings_webhook_dlq_service.py:40`).
impact:      Low on its own — grouped here because it is the same destination as api-endpoints-7 and
             should be fixed in the same pass, not as a separate sweep. `auth.py`'s 12 sites are the
             ones worth reading first: naive datetimes on token-expiry arithmetic are where this
             class of bug has teeth.
tests:       Import-grep reachability, not measured coverage: `auth` covered by the AuthNZ trees
             (1,207 import-grep hits module-wide), `notes` 24, `watchlists` 47.
effort:      Cheap, mechanical, and should be gated behind the destination decision in
             api-endpoints-7.
owner-only:  yes
confidence:  confirmed (the counts and the verbatim copies); assumption (that any of the 60
             `utcnow()` sites currently mis-compares — not traced per site)
```

## Suggested Refactor/Actions

1. **C9 is the single highest-value item in this module and nobody else covers it.** Start with
   `discord_oauth_admin.py` / `slack_oauth_admin.py` (91.2%, 407 lines each, tests on both sides).
   Needs the full design-first treatment: design doc, ADR, Backlog task, staged implementation plan.
   Collapse `tests/Integrations/test_{discord,slack}_endpoint_sanitizers.py` (100% identical) in the
   same change or the duplication simply relocates.
2. **Before any C9 extraction, write down which divergences are deliberate.** The table above is the
   starting list. The unified policy vocabulary is the first thing the shell must fix; leaving
   `team_quota_per_minute` and `workspace_quota_per_minute` as two names for one concept guarantees
   the shell grows a compatibility layer on day one.
3. **Fix `starter_copy_failed` -> 409 at `persona.py:2469-2482` now**, independently of the error-
   mapper consolidation. One line, confirmed misclassification.
4. **Delete `character_chat_sessions.py:5644-5660`** and use the truncator from the module the file
   already imports at `:235`, dropping the redundant second pass so `truncated` is honest. Cheap,
   well covered.
5. **Do not adopt `api/v1/utils/datetime_utils.py`.** Split `parse_timed_effects` back to the chat-
   dictionary feature, fix or retire `coerce_datetime`, and create
   `api/v1/utils/iso_datetime.py` for findings 7 and 16 together.
6. **Promote `mcp_unified_endpoint.py:128-158` as the opaque-cursor model** into
   `api/v1/utils/opaque_cursor.py`; keep the signed-token sites out of it. Make `workflows.py:2192`
   and `:2593` return 400 instead of silently restarting.
7. **Route finding 13 through the api-pagination Phase 3.2 plan**, not as a standalone sweep, and
   promote `kanban/_kanban_utils.py:resolve_limit_offset` rather than writing a new resolver.
