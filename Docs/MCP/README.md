# MCP Documentation Hub

This directory contains the Model Context Protocol (MCP) documentation for the TLDW server. MCP Unified is the only supported implementation; all guides in this folder describe the production stack.

## Structure

- **Unified/** - MCP Unified documentation set
  - `README.md` - Orientation and quick links
  - `Developer_Guide.md` - Architecture, module authoring, testing guidance
  - `System_Admin_Guide.md` - Deployment, observability, hardening
  - `User_Guide.md` - HTTP/WebSocket usage patterns and examples
  - `Modules.md` - Authoring reference for pluggable MCP modules
  - `Documentation_Ingestion_Playbook.md` - How to ingest project docs and query them via MCP tools
- `mcp_hub_management.md` - MCP Hub UI and API management surface (ACP profiles + external servers)
- `mcp_tool_catalogs.md` - Tool catalog governance and scope-aware APIs
- `mcp_prompts.md` - Protocol-level prompt discovery and rendering

## Picking the Right Guide

- Building or extending MCP? Start with `Unified/README.md`, then the developer and modules guides.
- Operating the service in production? Use the system admin guide.
- Integrating a client or testing workflows? The user guide covers HTTP and WebSocket flows.
- Building a prompt-capability client? Use `mcp_prompts.md` for `prompts/list` and `prompts/get`.

> Tip: The backend README at `tldw_Server_API/app/core/MCP_unified/README.md` links back to these guides for fast navigation.
