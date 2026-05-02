# Phase 3.4 Auth Risk Scan

Date: 2026-04-25

Scope: `tldw_Server_API/app/api/v1/endpoints`

This scan expands the Phase 3.4 auth dependency inventory with static signals for legacy user dependencies, raw user-dictionary handling, duplicate/manual admin checks, and dependency-ordering risk. Counts are text-scan counts, not unique endpoints.

## Summary

- `210` endpoint modules have at least one auth or auth-adjacent signal.
- `99` modules still reference legacy user dependencies such as `get_request_user`, `get_current_user`, or `get_current_active_user`.
- `24` modules have raw user-dictionary signals.
- `57` modules have manual admin-check signals.
- `120` modules have ordering-sensitive signals such as rate limits, quotas, audit, `request.state`, or scope-context handling.

## Raw User-Dictionary Signals

These route families should be reviewed before replacing dependencies with `AuthPrincipal` aliases because route code may expect dictionary-like users or direct `user_id` keys.

| Route family | Signals |
| --- | ---: |
| `auth` | 29 |
| `family_wizard` | 6 |
| `prompt_studio/prompt_studio_test_cases` | 6 |
| `admin/admin_ops` | 5 |
| `jobs_admin` | 5 |
| `prompt_studio/prompt_studio_prompts` | 5 |
| `agent_client_protocol` | 4 |
| `chunking_templates` | 4 |
| `persona` | 4 |
| `chat_workflows` | 3 |
| `prompt_studio/prompt_studio_projects` | 3 |
| `rag_unified` | 3 |
| `chat` | 2 |
| `data_tables` | 2 |
| `acp_permissions` | 1 |
| `authnz_debug` | 1 |
| `discord` | 1 |
| `embeddings_v5_production_enhanced` | 1 |
| `mcp_unified_endpoint` | 1 |
| `orgs` | 1 |
| `sandbox` | 1 |
| `slack_oauth_admin` | 1 |
| `vector_stores_openai` | 1 |
| `voice_assistant` | 1 |

## Manual Admin-Check Signals

Top manual-admin families:

| Route family | Signals |
| --- | ---: |
| `admin/admin_ops` | 59 |
| `workflows` | 56 |
| `embeddings_v5_production_enhanced` | 35 |
| `orgs` | 28 |
| `resource_governor` | 22 |
| `admin/admin_data_ops` | 21 |
| `integrations_control_plane` | 20 |
| `admin/admin_byok` | 18 |
| `mcp_catalogs_manage` | 17 |
| `scheduler_workflows` | 14 |
| `sandbox` | 13 |
| `admin/admin_tools` | 12 |
| `privileges` | 12 |
| `vector_stores_openai` | 12 |
| `flashcards` | 11 |
| `storage` | 11 |

These are not automatically bugs. Some duplicate checks are intentional defense in depth where service functions can be called outside FastAPI. Phase 3.4 should preserve service-layer checks unless ownership proves they are redundant.

## Ordering-Sensitive Signals

Top ordering-sensitive families:

| Route family | Signals |
| --- | ---: |
| `chat` | 157 |
| `chatbooks` | 140 |
| `audio/audio_streaming` | 125 |
| `workflows` | 125 |
| `sandbox` | 105 |
| `storage` | 103 |
| `embeddings_v5_production_enhanced` | 87 |
| `agent_client_protocol` | 85 |
| `admin/admin_storage_quotas` | 72 |
| `sharing` | 72 |
| `users` | 67 |
| `writing_manuscripts` | 64 |
| `notes` | 58 |
| `persona` | 53 |
| `auth` | 52 |
| `audit` | 47 |

Treat these as later migration candidates. Reordering dependencies in these families could alter rate-limit, quota, audit, or request-state behavior.

## Legacy User Dependency Signals

Top legacy dependency families:

| Route family | Signals |
| --- | ---: |
| `watchlists` | 64 |
| `notes` | 58 |
| `claims` | 42 |
| `persona` | 42 |
| `agent_client_protocol` | 31 |
| `embeddings_v5_production_enhanced` | 31 |
| `workflows` | 30 |
| `writing` | 27 |
| `character_chat_sessions` | 26 |
| `reading` | 22 |
| `sandbox` | 22 |
| `agent_orchestration` | 21 |
| `vector_stores_openai` | 21 |
| `workspaces` | 20 |
| `chat` | 18 |
| `family_wizard` | 18 |
| `sharing` | 18 |
| `voice_assistant` | 18 |
| `chatbooks` | 16 |
| `storage` | 16 |

## Pilot Implications

- `skills` remains a good first Phase 3.4 candidate: it has only `get_request_user` references and no role, permission, quota, org/team, or setup-local dependency mix.
- `data_tables` is not a first auth-cleanup candidate despite being a useful response/pagination pilot, because it has raw-user signals plus principal, permission, and rate-limit dependencies.
- `slides` should wait for stronger deny-path tests before auth cleanup because it mixes legacy user dependencies, permission checks, and `rbac_rate_limit`.
- `storage` should not be the first auth pilot because ordering-sensitive and manual-admin signals are higher than they looked in the initial dependency count.
- Admin, org/team, setup, workflows, chat, and MCP surfaces should be separate late slices.

## Next Action

Before adding standard auth aliases, add contract tests for the identity-only path that `skills` needs:

- missing credentials
- single-user API key
- multi-user JWT
- API-key fallback through `get_request_user`
- TEST_MODE dependency override
- `request.state.auth` and `_auth_user` reuse

Then migrate only `get_skills_service` to the chosen identity alias in a Phase 3.4-specific PR.
