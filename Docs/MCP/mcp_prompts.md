# MCP Prompts

This guide covers protocol-level MCP prompt discovery and rendering through
`prompts/list` and `prompts/get`.

This is separate from MCP tools named `prompts.search` and `prompts.get`. Those
tools search and fetch records through the tool-execution API. Protocol prompts
are discovered with MCP prompt methods and rendered as MCP prompt messages.

## Capability

During MCP initialization the server advertises prompt support as:

```json
{
  "prompts": {
    "listChanged": false
  }
}
```

`listChanged: false` means v1 does not send
`notifications/prompts/list_changed`. Clients should call a fresh
`prompts/list` when they need current prompt discovery results.

## Sources

`prompts/list` includes:

- readable, non-deleted prompts from the authenticated user's regular Prompt
  Library
- config prompts explicitly allowlisted in
  `tldw_Server_API/Config_Files/mcp_modules.yaml`

Prompt Studio prompts are excluded in this version.

Config prompt publishing defaults to an empty allowlist:

```yaml
settings:
  config_prompts:
    enabled: true
    entries: []
```

Single config prompt example:

```yaml
settings:
  config_prompts:
    enabled: true
    entries:
      - id: rag.retrieval_guidance
        module: rag
        key: retrieval_guidance
        title: Retrieval Guidance
```

Grouped config prompt example:

```yaml
settings:
  config_prompts:
    enabled: true
    entries:
      - id: chat.summary
        title: Conversation Summary
        messages:
          - role: system
            module: chat
            key: summary_system
          - role: user
            module: chat
            key: summary_user
```

Config roles may be `system`, `developer`, `user`, or `assistant`. MCP prompt
messages only permit `user` and `assistant`, so `system` and `developer`
content is folded into labeled `user` text when rendered.

## Stable Names

Prompt protocol names are stable identifiers:

- `library:<uuid>` for regular Prompt Library prompts
- `config:<id>` for allowlisted config prompts

The `title` field carries the human-readable name. Renaming a library prompt
changes the title but does not change its `library:<uuid>` name.

For config prompts, `id` is the allowlist entry id. `module` and `key` locate
the source prompt content loaded from config prompt files. Recommended id
values use module/key-style names such as `rag.retrieval_guidance`,
`chat.summary`, or `mcp.search_knowledge`.

## Arguments

Structured Prompt Library prompts expose their declared variables as MCP prompt
arguments.

Legacy library prompts and config prompts infer variables from placeholders such
as:

- `{{topic}}`
- `{context}`
- `$query`
- `<input>`

Argument values must be strings. Missing required arguments or non-string values
return MCP invalid params errors. Unknown extra arguments may be ignored unless
the structured prompt renderer rejects them.

## Pagination And Routes

`prompts/list` accepts the MCP `cursor` parameter and may return `nextCursor`.
The cursor is opaque to clients.

JSON-RPC list:

```json
{
  "jsonrpc": "2.0",
  "method": "prompts/list",
  "params": {
    "cursor": "<nextCursor>"
  },
  "id": 1
}
```

HTTP list convenience route:

```text
GET /api/v1/mcp/prompts
GET /api/v1/mcp/prompts?cursor=<nextCursor>
```

Use `/api/v1/mcp/request` for `prompts/get`; there is no single-prompt HTTP
convenience route in v1.

```json
{
  "jsonrpc": "2.0",
  "method": "prompts/get",
  "params": {
    "name": "library:2f5cf2fd-0000-4000-8000-000000000000",
    "arguments": {
      "topic": "MCP prompts"
    }
  },
  "id": 2
}
```

## Permissions And Scope

Authenticated MCP callers need `prompts.read` for protocol-level
`prompts/list` and `prompts/get`.

Protocol-level prompt access does not require `modules.read`.

Library prompts are read from the authenticated user's per-user Prompt Library
database. Persona prompt scopes filter library list and get results.

Allowlisted config prompts are visible to authenticated callers with
`prompts.read`.

## Safety

`prompts/list` responses include names, titles, descriptions, arguments, and
metadata, but not prompt bodies.

Rendered argument values and config override file paths are not included in
errors or logs.
