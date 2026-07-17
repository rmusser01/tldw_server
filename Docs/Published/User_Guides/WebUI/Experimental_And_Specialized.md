# Experimental And Specialized Pages

Some routes exist for beta workflows, specialized tools, hosted-only account flows, legacy compatibility, or internal QA. These pages are useful, but they are not all primary day-to-day workflows.

## Experimental And Specialized Routes

| Page or feature | Surface/status | What it lets you do | Common uses |
| --- | --- | --- | --- |
| `/vn-assets` | Experimental/labs | Manage visual novel asset packs and generation/review state. | VN asset workflows. |
| `/vn-scripts` | Experimental/labs | Author and manage visual novel scripts. | Branching stories, script editing. |
| `/vn-play` | Experimental/labs | Run visual novel play sessions. | Runtime playback, testing. |
| `/vn-play/sessions/[sessionId]/generations` | Experimental/labs | Inspect session generation output. | VN runtime generation review. |
| `/prototype-workspaces` | Experimental/labs | Use prototype workspace collaboration routes. | Workspace experiments. |
| `/for/journalists`, `/for/osint`, `/for/researchers` | Public/hosted-oriented | Show audience-specific public pages. | Explaining use cases to specific user groups. |
| `/composer-variants-preview` | Internal QA/debug | Preview composer UI variants. | Design and QA only. |
| `/onboarding-test` | Internal QA/debug | Test onboarding states. | QA and regression checks. |
| `/__debug__/sidepanel-chat` | Internal QA/debug | Debug sidepanel chat rendering in a web route. | Developer diagnostics. |
| `/__debug__/mermaid-chat-cards` | Internal QA/debug | Debug Mermaid chat-card rendering. | Developer diagnostics. |
| `/__debug__/sidepanel-error-boundary` | Internal QA/debug | Debug sidepanel error-boundary behavior. | Developer diagnostics. |

## Hosted-Only And Account Callback Routes

Hosted or multi-user deployments can expose account, billing, and auth callback routes that do not matter for ordinary local single-user use:

- `/billing`, `/billing/success`, `/billing/cancel`
- `/auth/magic-link`
- `/auth/reset-password`
- `/auth/verify-email`
- `/login`, `/signup`, `/account`, `/profile`

If you are running local single-user mode, these pages may be hidden, irrelevant, or useful only for testing route behavior.

## Legacy Aliases

| Alias | Canonical direction | Notes |
| --- | --- | --- |
| `/audio` | `/speech` | Compatibility route for the speech page family. |
| `/search` | `/knowledge` | Compatibility route for Knowledge QA. |
| `/prompt-studio` | `/prompts?tab=studio` | Prompt Studio now lives inside Prompts. |
| `/review` | Media review workflow | Older review entry retained for media review workflows. |
| `/moderation-playground` | Moderation/safety testing | Legacy safety testing route. |

## Related Docs

- [VN API](../../API-related/VN_PLATFORM_API.md)
- [VN asset packs API](../../API-related/VN_ASSET_PACKS_API.md)
- [VN platform API](../../API-related/VN_PLATFORM_API.md)
- [VN play API](../../API-related/VN_PLAY_API.md)
- [Prototype workspaces](../Prototype_Workspaces.md)
