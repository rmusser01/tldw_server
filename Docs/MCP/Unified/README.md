# MCP Unified Documentation

The MCP Unified stack is the production Model Context Protocol surface that ships with TLDW. Use this directory as the starting point for all current development, deployment, and client-integration work.

## Available Guides

- `Developer_Guide.md` - System architecture, extension patterns, testing strategy
- `System_Admin_Guide.md` - Installation, configuration, monitoring, and security hardening
- `User_Guide.md` - HTTP and WebSocket workflows, authentication, troubleshooting
- `Modules.md` - Reference for creating and managing pluggable MCP modules
- `CodeGraph.md` - Native CodeGraph module setup, indexing modes, tools, and operations
- `Governance_Operations.md` - Rollout modes, compatibility guarantees, and runbook for unified governance operations
- `Documentation_Ingestion_Playbook.md` - Step-by-step workflow for ingesting project docs and exposing them through MCP tools
- `Client_Snippets.md` - Minimal JS/Python examples for initialize → tools/list → tools/call
- `Adding_Tools.md` - Step-by-step guide to add new tools (modules) and register them
- `External_Federation.md` - Enabling and operating external MCP server federation (read + write policy controls)
- `../../API-related/Tools_API_Documentation.md` - REST facade for listing/executing tools via MCP

## When to Use MCP Unified

- You are contributing new modules, endpoints, or protocol features.
- You are deploying or operating the latest TLDW server builds.
- You want the definitive HTTP/WebSocket contract for the current MCP implementation.
