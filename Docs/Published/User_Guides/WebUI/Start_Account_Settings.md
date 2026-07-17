# Start, Account, And Settings

Use these pages when you are connecting to a server, setting up authentication, checking health, or changing WebUI and extension behavior.

## Main Pages

| Page or feature | Surface/status | What it lets you do | Common uses |
| --- | --- | --- | --- |
| `/` | WebUI, extension options, sidepanel entry | Resolve to the right starting state for setup, home, or sidepanel. | First launch, returning-user entry, sidepanel home. |
| `/setup` | WebUI, extension options | Configure server connection and complete first-run readiness checks. | Local setup, Docker WebUI onboarding, API key entry. |
| `/login`, `/signup` | Hosted-only or multi-user | Authenticate or create an account when multi-user login is enabled. | Shared server login, hosted account entry. |
| `/account`, `/profile` | Hosted-only or authenticated user | Inspect identity, role, permissions, and current account state. | Permission debugging, profile review. |
| `/privileges` | Advanced self-hosted | Inspect authorization privileges when RBAC features are enabled. | Admin-assisted troubleshooting. |
| `/config` | Advanced self-hosted | Review server configuration and capability state outside the settings workspace. | Deployment diagnostics. |
| `/billing` and billing callbacks | Hosted-only | Manage hosted billing and subscription flows. | Hosted deployments. |
| `/404` | WebUI recovery | Recover from unknown or stale routes. | Broken links, old bookmarks. |

## Settings Areas

| Page or feature | Surface/status | What it lets you do | Common uses |
| --- | --- | --- | --- |
| `/settings` | WebUI, extension options, sidepanel settings | Edit general connection and application settings. | Server URL, auth mode, defaults. |
| `/settings/tldw` | Shared UI | Configure tldw server behavior exposed to the UI. | Server integration preferences. |
| `/settings/model` | Shared UI | Manage model and provider selection defaults. | Chat model defaults, local model setup handoff. |
| `/settings/provider-keys` | WebUI | Manage provider keys where the deployment supports user-managed keys. | BYOK and provider credentials. |
| `/settings/chat`, `/settings/chat-dictionaries` | Shared UI | Configure chat behavior and dictionary/lore helpers. | Chat defaults, replacements, roleplay context. |
| `/settings/prompt`, `/settings/prompt-studio` | Shared UI | Configure prompt-library and prompt-studio behavior. | Prompt authoring preferences. |
| `/settings/knowledge`, `/settings/rag` | Shared UI | Tune knowledge and retrieval behavior. | RAG defaults, source search behavior. |
| `/settings/speech` | Shared UI | Configure speech, transcription, and TTS behavior. | STT/TTS defaults. |
| `/settings/evaluations` | Shared UI | Configure evaluation defaults. | RAG/model evaluation workflows. |
| `/settings/health` | WebUI, extension options | Inspect connection, readiness, and degraded backend state. | Troubleshooting backend reachability. |
| `/settings/ui`, `/settings/splash` | Shared UI | Customize interface and splash behavior. | UI preferences. |
| `/settings/quick-ingest` | Shared UI | Configure quick ingest defaults. | Browser capture and upload defaults. |
| `/settings/image-generation`, `/settings/image-gen` | Advanced self-hosted | Configure image generation providers; `/settings/image-gen` is a compatibility route. | Image provider setup. |
| `/settings/share` | Shared UI | Configure sharing behavior. | Shared resources and link defaults. |
| `/settings/processed` | Shared UI | Inspect processed data settings or state. | Cleanup and processing diagnostics. |
| `/settings/about` | Shared UI | View version and application information. | Support and diagnostics. |

## Setup Notes

- Single-user deployments usually rely on an API key. The WebUI and extension send that key automatically after configuration.
- Multi-user deployments use login/JWT flows and may expose account, profile, organization, and privilege information.
- Extension setup can additionally require browser host permissions for the configured server origin.
- If pages report backend unreachable or degraded status, start with `/settings/health` before changing feature settings.

## Related Docs

- [Self-hosting profiles](../../Getting_Started/README.md)
- [Authentication setup](../Server/Authentication_Setup.md)
- [BYOK user guide](../Server/BYOK_User_Guide.md)
- [OpenAI OAuth first-time setup](../Server/OpenAI_OAuth_First_Time_Setup.md)
- [Current WebUI user guide](../WebUI_Extension/User_Guide.md)
