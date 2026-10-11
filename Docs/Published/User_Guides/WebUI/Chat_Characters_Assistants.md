# Chat, Characters, And Assistants

Use these pages when you want to talk with models, run character or persona workflows, manage assistant context, or build repeatable chat flows.

## Pages And Feature Sets

| Page or feature | Surface/status | What it lets you do | Common uses |
| --- | --- | --- | --- |
| `/chat` | WebUI, extension options, sidepanel | Run the main conversation workspace with model selection, history, web search, context, tools, and helper controls. | General assistants, model-backed chat, current-page chat from the extension. |
| `/chat/agent` | WebUI | Open an agent-oriented chat route. | Agent chat experiments and tool-backed conversation. |
| `/quick-chat-popout` | Advanced self-hosted | Open focused chat in a smaller route. | Fast prompts, popout chat, low-friction model checks. |
| `/persona` | WebUI, extension options, sidepanel | Use persona-backed assistant behavior. | Personal assistant flows, persona continuity, voice/persona setup. |
| `/companion`, `/companion/conversation` | WebUI, extension options, sidepanel | Use companion-oriented assistant pages and compact sidepanel conversation. | Ongoing assistant state, browser-adjacent companion sessions. |
| `/characters` | WebUI, extension options | Manage character cards, imports, generation helpers, tags, and roleplay setup. | Character roleplay, persona libraries, SillyTavern-compatible cards. |
| `/agents`, `/agent-tasks` | Advanced self-hosted | Inspect agent registry and agent task status. | Agent orchestration experiments, task monitoring. |
| `/chat-workflows` | Advanced self-hosted | Manage chat workflow entry points. | Guided conversations, repeatable assistant steps. |
| `/chat-workspace` | Advanced self-hosted; WebUI and extension options | Chat in the active workspace with explicit source staging and read-only runtime/inspector rails. | Project conversation, staged source retrieval, send-state inspection. |
| `/dictionaries` | WebUI, extension options | Manage chat dictionary entries and replacement behavior. | Terminology expansion, roleplay context, acronym replacement. |
| `/world-books` | WebUI, extension options | Manage lorebook/world-book context. | Roleplay settings, reusable background, character worlds. |
| `/settings/chat`, `/settings/chat-dictionaries`, `/settings/characters`, `/settings/world-books` | Shared UI | Configure chat, dictionaries, character, and lore settings. | Defaults and behavior tuning. |

## Larger Systems

The chat system is not one page. It spans Chat, sidepanel chat, character libraries, persona state, dictionaries, world books, workflow templates, model/provider settings, and optional tool or web-search controls. For ordinary conversation and model/assistant selection start with `/chat`; for persistent roleplay or assistant identity start with `/characters` or `/persona`.

For focused conversation in an active workspace, use [Chat Workspace](Chat_Workspace.md). Select the workspace in Research Workspace first, and choose a model in Chat. Chat Workspace has no workspace/model picker or persona editor. Its inspector reports scope, staged sources, effective model/persona, and runtime state; it does not offer approval controls. Browsing a source does not stage or send it, and **Insert context summary** inserts only source-list text, not source content.

For cited library questions use Knowledge QA (`/knowledge`); for source organization and Studio outputs use Research Workspace (`/research-workspace`); for document-centered reading and chat use Document Workspace (`/document-workspace`). The [Chat Workspace guide](Chat_Workspace.md) explains these boundaries, persona inheritance, context submission, and unsent-state lifetime.

## Extension Differences

The extension sidepanel chat is optimized for browser context. It can open from the toolbar or context menu, can use the current page as context, and has narrower layout constraints. If sidepanel chat cannot connect, verify the extension server URL, API key or login state, browser host permission, and backend reachability.

## Related Docs

- [Chat Workspace](Chat_Workspace.md)
- [Chat pages](../WebUI_Extension/Chat_Pages.md)
- [Character roleplay quickstart](../WebUI_Extension/Character_Roleplay_Quickstart.md)
- [Effective character roleplay](../WebUI_Extension/Effective_Character_Roleplay_and_You.md)
- [Advanced character roleplay](../WebUI_Extension/Advanced_Character_Roleplay_Guide.md)
- [Chat dictionaries guide](../WebUI_Extension/Chat_Dictionaries_Guide.md)
- [Persona user guide](../Server/Personas_User_Guide.md)
- [Buddy and Persona management](Buddy_And_Persona_Management.md)
- [Extension sidebar docs](https://github.com/rmusser01/tldw_server/blob/main/apps/extension/docs/sidebar/index.md)
