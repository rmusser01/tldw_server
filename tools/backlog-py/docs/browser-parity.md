# Browser Parity Decision

This document records browser UI parity requirements for a full Backlog.md clone
and separates them from the first local-file agent cutover candidate. Browser
support is valuable for human project management, but it is not required for
agent workflows that use plain CLI output and pure MCP helpers.

## Decision

Browser parity is intentionally deferred for the first agent cutover candidate.
The Python clone must not silently pretend to support browser behavior until the
items below are implemented and tested. Each browser item remains required for a
full clone unless explicitly rejected.

## Browser Requirements

| Requirement | Classification | Agent cutover impact | Rationale |
| --- | --- | --- | --- |
| Responsive Kanban board | Required for full clone | Not required for agent cutover | Agents use `board`, task listing, and MCP tools; responsive browser layout is a human UI requirement. |
| drag-and-drop task movement | Required for full clone | Intentionally deferred | Drag semantics need browser tests, conflict handling, and status persistence coverage before claiming browser parity. |
| Task create/edit forms | Required for full clone | Not required for agent cutover | CLI and MCP task create/edit cover agent workflows; browser forms can follow after API and storage parity stabilize. |
| Acceptance criteria editor | Required for full clone | Not required for agent cutover | Agent workflows already mutate AC checks through safe CLI/MCP paths; rich browser editing is later UI work. |
| Definition of Done settings | Required for full clone | Not required for agent cutover | DoD defaults exist through config helpers, CLI, and MCP; browser settings can layer on the same safe core later. |
| Real-time updates | Required for full clone | Intentionally deferred | Live browser state needs a service process or polling contract and concurrent mutation tests. |
| Archive confirmations | Required for full clone | Not required for agent cutover | Agents use explicit archive/complete operations; browser confirmation UX belongs to a later human UI milestone. |
| Rich Markdown editing | Required for full clone | Intentionally deferred | Rich editing must preserve unknown Markdown and frontmatter exactly, so it needs round-trip visual and parser tests. |
| mermaid rendering | Required for full clone | Intentionally deferred | Mermaid rendering is browser-only presentation behavior and does not affect CLI/MCP file correctness. |
| service mode | Required for full clone | Intentionally deferred | A browser service mode needs lifecycle, port, logging, and shutdown policy that should not be bundled with local-file cutover. |
| Mobile behavior | Required for full clone | Intentionally deferred | Mobile layout should be verified with real browser screenshots after the browser implementation exists. |

## Acceptance For Full Browser Clone

A later browser milestone should not be marked complete until it has:

- End-to-end browser tests for create/edit, drag-and-drop, archive confirmation,
  and settings flows.
- Responsive checks for desktop and mobile viewports.
- A clear service mode lifecycle with startup, shutdown, logging, and port
  collision behavior.
- Round-trip tests proving rich Markdown editing does not damage frontmatter,
  owned sections, unknown body text, mermaid blocks, or checklist markers.
- Documentation that states whether the browser uses polling, server-sent
  events, WebSockets, or static reloads for real-time updates.

## Rejected For Agent Cutover

No browser-only capability is allowed to block the first agent cutover. The
first candidate is limited to local file operations through plain CLI output and
pure MCP helper functions.
