# MCP Tool Availability And Deferred Search Design

## Goal

Add Claude-style profile-scoped tool availability and deferred tool-search behavior to the standalone MCP gateway without expanding the initial executable tool surface. Profiles should expose direct tools immediately, keep additional installed-but-deferred tools searchable, and show recommendation-only tools as unavailable setup targets.

## Current State

`mcp_unified.gateway.tool_discovery` already builds a profile-filtered catalog from backend tools and profile recommendation metadata. It ranks with standard-library BM25 and documents the intended order: profile grants, installation status, category filtering, BM25, category priority, and tool id. `ProfileAwareGatewayRuntime` also exposes bridge tools: `tool_categories.list`, `profile.tools.list`, `tool_search`, `tool_describe`, and conditional `tool_call`.

The missing behavior is exposure control. Any installed tool allowed by profile policy is currently returned by `list_tools()`, even if its category is intended for progressive disclosure. That makes direct and deferred categories metadata-only instead of controlling model-visible tools.

## Design

The gateway will classify each visible discovery entry into one of three exposure states:

- `direct`: installed and allowed by profile policy, and its category is in the profile's direct categories.
- `deferred_installed`: installed and allowed by profile policy, but its category is not direct or is explicitly listed as deferred.
- `recommended_unavailable`: recommendation metadata with no callable backend tool.

The existing `installation_status` remains separate from exposure. Installed deferred tools keep `installation_status: "installed"` and add `exposure: "deferred"` or an equivalent public field. Recommendation-only tools keep `installation_status: "recommended_unavailable"`.

`ProfileAwareGatewayRuntime.list_tools()` will return only:

- direct installed tools,
- the read-only discovery bridge tools,
- `tool_call` when the profile has deferred categories or deferred installed tools.

`tool_search`, `tool_describe`, and `profile.tools.list` will continue to include direct, deferred installed, and recommended-unavailable entries. This gives clients progressive disclosure without losing awareness of useful tools. `tool_call` can execute deferred installed tools after resolving through the same profile policy path used for direct backend calls. Recommended-unavailable tools still return `tool_not_enabled`.

## Ranking And Filtering

No semantic search is introduced. Search remains deterministic and local:

1. Filter by effective profile grants first.
2. Classify installation and exposure.
3. Apply requested category filter.
4. Rank with BM25.
5. Tie-break by category priority and tool id.

Denied tools are not returned from discovery, description, or call resolution. Denials should look like `tool_not_found` to the model-facing bridge so fully denied tools are not advertised.

## Availability Metadata

Catalog payloads should include scalar metadata that helps front-ends explain availability:

- `direct_count`, `deferred_installed_count`, and `recommended_unavailable_count` per category.
- Global `availability` or `summary` counts for the whole profile catalog.
- Per-tool `exposure` and `availability_reason_code`, without absolute paths, secrets, or denied-tool names.
- Existing ranking metadata must continue to state `semantic_search: false`.

This metadata should be safe for chat and ACP sessions.

## Wait For MCP Servers Interaction

This slice will not implement server startup waiting. It will reserve a readiness field for future integration, such as `readiness_status` and `readiness_reason_code`. Existing external-runtime startup and install/update tasks can later populate this without changing the bridge response shape.

## Testing

Focused tests should cover:

- Deferred installed tools are hidden from initial `list_tools()` but visible through `tool_search` and `profile.tools.list`.
- Direct installed tools remain listed normally.
- `tool_call` can delegate a deferred installed tool through normal profile policy checks.
- Recommendation-only tools remain discoverable but not callable.
- Category counts and global availability counts distinguish direct, deferred installed, and recommended unavailable.
- Denied tools do not appear in discovery or descriptions.
- Ranking metadata remains non-semantic and deterministic.

## Non-Goals

- New semantic embeddings or vector search.
- New install/update behavior.
- New permission-rule parser syntax.
- Hook enforcement changes.
- LSP, shell, notebook, web, or monitor tool implementations.
