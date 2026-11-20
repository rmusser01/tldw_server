# Refresh commercial model defaults and selectable catalogs

November 19, 2026 · TASK-15206.1

## Problem and scope

The current catalog can correctly discover an account's available models while the shipped config still selects a retired model. Several bundled output budgets also predate the currently supported models. The picker needs current model IDs, and requests that omit a model or output budget need a usable default.

Audit every supported commercial text-generation provider in the adapter registry, including direct providers, API routers, and cloud services with region- or account-dependent model access. A provider gets a bundled model where it already has a shipped default. For providers that require endpoint-specific or account-specific configuration, retain that requirement and record it in the provider audit. Local model defaults and the separate catalogs for embeddings, images, audio, and reranking are outside the scope.

## How to determine availability

Use the existing [provider-authoritative inventory contract](../../Design/provider-authoritative-model-inventory-20261004.md). For providers with authenticated inventory, the selectable catalog continues to reflect the effective credential, endpoint, and allowlist. The provider response determines what that account can select. For providers with a static fallback, use the provider's official endpoint-specific model catalog and lifecycle notices as evidence. Keep the existing permission, caching, and failure behavior at both call boundaries.

Record an audit row per commercial provider. Include the supported endpoint or region, current inventory mechanism, old and new bundled default when applicable, affected selectable IDs and their dispositions, advertised output limit, effective default output budget, official links, and verification date. An account-restricted model can remain selectable when live discovery includes it. A confirmed retirement is removed from new selection on the affected endpoint. A future shutdown is recorded separately from a completed retirement.

Keep exact IDs and historical prices for usage reporting. Use the existing hidden or compatibility representation where static catalogs and pricing share a file. Previously saved selections and explicit model configuration remain readable. If a saved ID has retired, use the existing unavailable-model error so users can choose a replacement.

The initial audit found these cases.

| Provider | Finding |
| --- | --- |
| OpenAI | The GPT‑4o model page advertises 16,384 output tokens. Its specific May 2024 snapshot has shut down. Verify alias availability separately when selecting the current bundled default. [GPT‑4o model details](https://developers.openai.com/api/docs/models/gpt-4o) · [API lifecycle](https://developers.openai.com/api/docs/deprecations) |
| Anthropic | The bundled `claude-opus-4-20250514` ID retired on June 15, 2026. Current replacements have different output and sampling constraints. [Claude model lifecycle](https://platform.claude.com/docs/en/about-claude/model-deprecations) · [Claude model limits](https://platform.claude.com/docs/en/models/overview) |
| Google | Google restricts access to `gemini-2.5-flash` and will route the ID to `gemini-3.5-flash-lite` starting December 1, 2026. The replacement advertises 65,536 output tokens. [Gemini lifecycle](https://aistudio.google.com/docs/deprecations) · [Gemini 3.5 Flash-Lite](https://aistudio.google.com/docs/models/gemini-3.5-flash-lite) |
| DeepSeek | The shipped `deepseek-chat` alias is legacy. Use `deepseek-flash` as the bundled direct API default with a 200,000-token request budget. DeepSeek advertises a 384K output ceiling. [DeepSeek release notes](https://api-docs.deepseek.com/updates/) · [DeepSeek model limits](https://api-docs.deepseek.com/quick_start/pricing/) |

## Bundled models and output budgets

Pick a stable model supported by each provider's existing generation endpoint for deployments using the bundled configuration. Match the existing default's general purpose and choose among current options using compatibility, latency, and cost. Check the bundled prompt, sampling, streaming, and request fields against the selected endpoint. Keep the sample configuration and any runtime fallbacks aligned. Existing installation configuration and request overrides keep their precedence.

Record the documented model output ceiling in model metadata. Use a separate runtime default when a model's maximum is unsuitable for a provider transport or the service's generation path. Resolve the effective model first. Use the provider's native output field and the following precedence for the budget.

1. An explicit request value, including a supported alias such as `max_completion_tokens`.
2. An explicit operator or provider configuration value.
3. A verified bundled budget applicable to the resolved provider, model, and request path.

For a model without a verified bundled budget, keep the existing provider fallback behavior. Router-selected and custom-endpoint models use the rules of the endpoint that serves them. Where the request path already computes remaining context, bound its implicit budget to that remaining capacity. Preview, region-specific, streaming-only, and batch-only output allowances apply only to the corresponding endpoint and mode. Provider and validation limits still apply to explicit requests.

Keep this work inside existing config, metadata, and adapter boundaries. The relevant output budget has to reach the provider payload from normal Chat and analysis dispatch, including streaming and non-streaming paths where both exist.

## Verification and PRs

- Verify that each audited bundled model is selectable where the provider advertises it, and confirm that configured eligibility narrows live results.
- Test retired-ID exclusion, retained historical price lookup, and the existing unavailable-model response for a saved retired ID.
- Capture outbound payloads with mocked providers. Cover an omitted output limit, a smaller explicit limit, a provider configuration override, an alternate model on the same provider, and the appropriate streaming or non-streaming limits. Verify the DeepSeek Flash default sends 200,000. Exercise the Chat and analysis entry points that depend on the defaults.
- Run focused configuration, inventory, pricing, and adapter tests with the repository's lint and security checks. Record per-provider evidence and any account or region limits on the verification.
- Open the public PR against `dev` and backport the verified behavior through the hosted patch workflow on its pinned backend. Verify the hosted materialization and equivalent payload behavior. Link the two PRs with their separate coverage and Base/HEAD revisions.

## ADR assessment

A new ADR is unnecessary. The changes refresh data and make bundled defaults consistent within [ADR-025's provider routing and override boundary](../../ADR/025-llm-provider-adapter-routing-and-overrides.md). Existing credential scoping, catalog discovery, provider transport, and usage accounting govern the implementation.
