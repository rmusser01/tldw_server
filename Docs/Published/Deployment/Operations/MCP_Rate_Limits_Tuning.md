# MCP Rate Limits - Operations Tuning Guide

This guide explains how to tune MCP Unified rate limits in production.

## Overview

MCP provides:
- A global rate limiter (RPM + burst) for all requests
- Per-category limiters (e.g., `ingestion`, `read`) applied per tool
- Config-driven tool→category mapping via JSON env or YAML file
- Optional RG Redis backend for multi-node deployments

## Key Environment Variables

Global:
- `MCP_RATE_LIMIT_ENABLED` - Enable/disable limiter (default on)
- `MCP_RATE_LIMIT_RPM` - Global requests per minute
- `MCP_RATE_LIMIT_BURST` - Allowed burst tokens

Distributed (multi-node):
- `RG_BACKEND=redis`
- `REDIS_URL=redis://host:6379/0`

Category-specific (optional):
- `MCP_RATE_LIMIT_RPM_INGESTION` / `MCP_RATE_LIMIT_BURST_INGESTION`
- `MCP_RATE_LIMIT_RPM_READ` (burst falls back to global burst)

Tool→category mapping (choose one):
- JSON env: `MCP_TOOL_CATEGORY_MAP='{"ingest_media":"ingestion","media.search":"read"}'`
- YAML file: `MCP_TOOL_CATEGORY_MAP_FILE=tldw_Server_API/Config_Files/mcp_tool_categories.yaml`

Sample YAML (checked in):
- `tldw_Server_API/Config_Files/mcp_tool_categories.yaml`

## Recommended Defaults

- Ingestion: RPM 20-60, burst 3-10 (depending on workload and backend capacity)
- Read: RPM 120-600, burst 10-50 (depending on client needs)
- Redis: enable in any horizontally-scaled environment

## Change Management

1. Update environment variables
   - For docker/k8s: edit Deployment/ConfigMap/Secret as appropriate
2. Reload or rolling restart
   - Changes to env vars require an application restart to take effect
3. Verify
   - Check Prometheus metrics: `mcp_rate_limit_hits_total{key_type="user|client|anonymous"}`
   - Watch p95 latency: `histogram_quantile(0.95, sum(rate(mcp_request_duration_seconds_bucket[5m])) by (le, method))`
   - Observe module operation counts and errors per module

## Troubleshooting

- Too many 429 responses
  - Increase `MCP_RATE_LIMIT_RPM_INGESTION` or refine tool mapping to classify hot tools as `read` (if safe)
- Inconsistent rates across nodes
  - Ensure `RG_BACKEND=redis` and a shared `REDIS_URL` are set for all instances
- Unexpected tool classification
  - Confirm `MCP_TOOL_CATEGORY_MAP` or `MCP_TOOL_CATEGORY_MAP_FILE` entries and that the tool names match module definitions exactly

## Security Notes

- Keep the Prometheus endpoint gated by `RequirePermission("system.logs")` / admin-style principals. Do not expose it unauthenticated; use credentials or an ingress/auth proxy for Prometheus scraping on internal networks.

## References

- MCP README (Rate Limits section): `tldw_Server_API/app/core/MCP_unified/README.md`
- Tool mapping file example: `tldw_Server_API/Config_Files/mcp_tool_categories.yaml`
- Metrics Cheatsheet: `Docs/Deployment/Monitoring/Metrics_Cheatsheet.md`
