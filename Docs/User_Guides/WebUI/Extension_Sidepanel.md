# Extension Sidepanel

The browser extension adds a compact sidepanel, context menu actions, background request handling, and browser-page capture flows around the same tldw_server backend. Some extension option routes reuse the same shared UI components as the WebUI; sidepanel routes are compact extension-only experiences.

## Extension Surfaces

| Surface | What it does | Common uses |
| --- | --- | --- |
| Extension options page | Full-page extension UI for setup and shared option routes. | Server URL/API key setup, settings, Knowledge QA, shared feature pages. |
| Sidepanel home `/` | Compact entry state for the sidepanel. | Starting from the browser toolbar or context menu. |
| Sidepanel `/chat` | Compact chat surface. | Chat with the current page, ask quick questions, use model/RAG helpers. |
| Sidepanel `/clipper` | Page clipping and capture flow. | Save page content, send links or snippets to the server. |
| Sidepanel `/persona` | Compact persona assistant. | Persona-backed browser sessions. |
| Sidepanel `/companion`, `/companion/conversation` | Compact companion workflows. | Ongoing browser-adjacent assistant state. |
| Sidepanel `/agent` | Agent-focused sidepanel route. | Agent experiments from the browser. |
| Sidepanel `/flashcards` | Compact flashcard review. | Quick review while browsing. |
| Sidepanel `/settings` | Extension settings in the compact shell. | Connection and behavior adjustments. |
| Sidepanel `/error-boundary-test` | Internal QA/debug route. | Extension error-boundary testing only. |

## Browser Context Features

The extension can add context menu actions for browser workflows:

- Open the sidepanel or WebUI.
- Send the current page or link to tldw_server.
- Process content without saving it.
- Transcribe video or audio.
- Transcribe and summarize video or audio.

Those actions require a reachable server and valid credentials. Chromium-based browsers may also require host permission for the configured server origin so background requests can include authentication headers.

## Common Setup Problems

| Problem | What to check |
| --- | --- |
| Sidepanel opens but cannot reach the server | Server URL, API version, host permission, backend health. |
| API key or login does not stick | Extension storage, auth mode, configured server origin. |
| Current-page chat has no page context | Content-script permission, page type, extension host access. |
| Context menu action fails | Server reachability, action configuration, backend feature availability. |
| WebUI works but extension fails | Extension-specific permissions and background request path. |

## Related Docs

- [Extension user docs](https://github.com/rmusser01/tldw_server/tree/main/apps/extension/docs)
- [Sidebar docs](https://github.com/rmusser01/tldw_server/blob/main/apps/extension/docs/sidebar/index.md)
- [Extension shortcuts](https://github.com/rmusser01/tldw_server/blob/main/apps/extension/docs/shortcuts.md)
- [Current WebUI user guide](../WebUI_Extension/User_Guide.md)
- [Knowledge QA guide](../WebUI_Extension/Knowledge_QA_Guide.md)
- [Flashcards study guide](../WebUI_Extension/Flashcards_Study_Guide.md)
