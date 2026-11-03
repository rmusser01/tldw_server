# ADR-083: The LLM quotas neither count nor check a call on the user's own provider key

**Status:** Accepted
**Date:** 2026-10-26
**Backfilled from:** not backfilled
**Decision owner:** Repository owner (@rmusser01)
**Related task:** TASK-13955
**Related spec/plan:** `Docs/superpowers/plans/2026-10-26-byok-quota-exemption.md`
**Amends:** ADR-064 (what the per-user LLM limits count)

## Decision

A call made with the user's own stored key for a provider that bills that key, sent to that vendor's own host, is recorded in `llm_usage_log` with its real tokens and cost and `credential_source = 'user'`. `limits.llm_tokens_per_month`, `limits.llm_cost_per_day_usd`, `limits.llm_cost_per_month_usd` and the hosted billing plan's `llm_tokens_month` do not count it, and the per-user limits never refuse it. `/chat/completions` and the Messages API admit an over-limit user whose provider resolves to their own key.

## Context

The owner's decision of 2026-10-26: "BYOK users own keys shuold not count against their own costs". The LLM usage accounting design (`Docs/Design/2026-10-09-llm-usage-accounting-design.md`, owner decision 3) had gated BYOK calls "for now", because most call paths did not say which key they used. The final review of TASK-13936 found that a user's own-key calls still counted against that user's limits. When the vendor bills the user's key, the operator pays nothing for such a call. A key stored for an endpoint the operator runs is different: the operator's model answers whatever key is sent.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Record an own-key call at $0 | No new column, but the token limit still counts the call, the user and the operator lose its real cost, and a $0 row cannot be told from a free local call. |
| Exempt the cost limits only | The token limit would still refuse a user who works on their own key, which is the complaint being fixed. The token and cost limits budget the same thing, the operator's provider bill, in two units. |
| Do not record own-key calls | The user and the operator would lose the record, and ADR-067 records every call made for a user. |
| Treat every stored user key as the user's own | A key stored for a custom OpenAI-compatible slot or a local server reaches the operator's own endpoint, so a junk key would switch the limits off on the operator's hardware. |
| Treat a user key on any provider that bills per key as the user's own, and tell operators to take such a provider off `BYOK_ALLOWED_PROVIDERS` when they point its base URL elsewhere | The operator can point any of those providers at its own model server (`config.txt` or an environment variable). One missed provider would give every BYOK user unlimited use of the operator's hardware. |
| At every request boundary, let through any over-limit user who has an own key for some provider | The refusal would move from the boundary to the call on about 30 endpoints, and the "checked once" guarantees of benchmark runs, evaluation batches and media jobs would no longer hold for those users. |

## Consequences

- **The column.** `llm_usage_log.credential_source` holds `user`, `user_unmetered`, `team`, `org`, `server_default` or `none`. It is NULL when the recorder cannot tell, and on rows written before the column existed. SQLite gets it from an AuthNZ migration, PostgreSQL from the usage-table bootstrap. A bootstrap statement that fails with one of its handled errors (a timeout or a socket-level connection error) is now logged as a warning, not at debug; a database error such as a missing privilege, or asyncpg's own closed-connection error, stops startup, as before. On a schema without the column a row is kept without its source, every reader counts every row, and the server logs a warning once.
- **Whose key is the user's own.** Only a key for one of the providers that bill per key, and only while that provider reaches its vendor: every base URL the configuration gives it must be on the host its adapter calls by default. That covers `config.txt` (for example `qwen_api_base_url`, `moonshot_api_base_url`, `zai_api_base_url`, `huggingface_api_base_url`), environment variables such as `OPENAI_API_BASE_URL`, and a base URL a platform admin stored with a key. A key stored for a custom OpenAI-compatible slot, a local provider, a provider the server does not know, or a billed provider pointed at a self-hosted model server or a gateway is recorded as `user_unmetered` and counts. A vendor host other than the one its adapter defaults to (OpenAI's data-residency hosts such as `eu.api.openai.com`, `api.cohere.com`, `api.moonshot.ai`, Bedrock's FIPS endpoints) also counts, until it is added to `VENDOR_HOSTS`.
- **What counts.** The three limits, the profile's `llm` usage figures and the hosted plan's monthly LLM tokens count every row except `user`. So org and team shared keys, server keys, operator endpoints, keyless local providers and unknown rows still count. On PostgreSQL the hosted plan's sum counted nothing before this change: it bound a timezone-aware month start against a `TIMESTAMP` column, and the driver refused it. It now counts.
- **Where the source comes from.** A runtime-issued credential handle carries its source. A caller that resolves the key itself and passes a raw API key marks the call context. A call with neither counts, as the auto-router's call does today.
- **At the call.** A call on the user's own key is not checked, whoever started it: a job, a workflow, a WebSocket or an endpoint. A refused call on a fallback provider is the user's limit, not the provider's failure, so it does not mark the provider unhealthy.
- **At the boundary.** `/chat/completions` and the Messages API know the provider before any LLM work. Suppose the user is over a limit, BYOK is on, and the provider bills per key, reaches its vendor, is allowlisted and has an active key of the user's. Then the request is let through without the admitted mark, so each of its calls is checked where its key is known. `/chat/completions` also skips the hosted plan's check for such a request. The other boundary endpoints and a media ingest job's start still refuse an over-limit user before the key is known.
- **Other limits are unchanged.** The Resource Governor's token caps, virtual-key budgets, `limits.evaluation_tokens_per_day` and the `llm_cost_dollars` metric still count own-key calls.

## Follow-up

TASK-13955's final summary lists the follow-up task: the auto-router's usage row and the other boundary endpoints (TASK-13956). The owner decided on 2026-10-29 that org and team shared keys keep counting.

## Amendment (2026-10-31): the vendors' other hosts, and a trusted caller's per-request base URL

TASK-13959 closes two gaps that TASK-13955's final review found in "Whose key is the user's own".

- **The vendors' other hosts are theirs.** A vendor's own host is no longer only the one its adapter calls by default. `VENDOR_HOSTS` also lists the hosts each vendor documents for the same API: OpenAI's data-residency hosts (`<region>.api.openai.com`, such as `eu.api.openai.com`), `api.cohere.com`, `api.moonshot.ai`, and Bedrock's FIPS endpoints (`bedrock-runtime-fips.<region>.amazonaws.com`). A user key sent to one of them is the user's own, so the sentence in Consequences that such a host counts no longer holds. A `*` in a host pattern matches one DNS label, so `bedrock-runtime.<region>.amazonaws.com` no longer also matches a deeper host.
- **A per-request base URL is read too.** A platform admin, a service principal or the single user may pass a `base_url` with the request (ADR-025), and every billed adapter prefers it to the configured one. Off the vendor's hosts, that call is `user_unmetered` and counts, like a configured base URL. Before, it was recorded as `user`.
