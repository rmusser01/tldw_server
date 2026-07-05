# WebUI Setup Choice Design

Backlog task: TASK-12123

## Purpose

New self-hosted users can follow the install steps, start the API server, start
the WebUI, open the WebUI first, and then wonder where configuration is
supposed to happen. The project now has two setup surfaces:

- the API server `/setup` surface for server/operator configuration and
  recovery
- the WebUI `/setup` surface for guided first-run provider setup and first chat

This design makes that split explicit before the WebUI first-run wizard starts.
It is written for nontechnical users and for users who skipped or skimmed the
setup instructions.

## Current Baseline

The backend already exposes the setup state and first-run APIs the WebUI needs:

- `GET /api/v1/setup/first-run/state`
- `GET /api/v1/setup/first-run/metadata`
- `POST /api/v1/setup/first-run/*`
- `GET /api/v1/setup/readiness/*`
- API server `/setup` recovery/configuration endpoints guarded by local setup
  access unless remote setup is explicitly enabled

The WebUI already has a thin `pages/setup.tsx` wrapper that renders shared
`@/routes/option-setup`. That route currently shows a manual "Connect your tldw
server" card, a setup recovery panel, and the existing `UnifiedSetupWizard`.

The implementation should extend that route instead of adding a second setup
system.

## Product Decision

When WebUI `/setup` loads and backend setup is incomplete, show a plain-language
choice screen before the existing WebUI wizard.

The choice screen explains that the user is in the WebUI, that tldw also has an
API server setup page, and that most users should start with WebUI setup unless
they need server/operator configuration.

Primary action:

- **Set up in WebUI**
- Use this path to add a chat provider, choose a model, send a first test chat,
  and finish first-run onboarding.

Secondary action:

- **Open API server setup**
- Open the API server `/setup` page separately.
- Use this path for server/operator settings, recovery, local/remote setup
  access, and file-backed configuration issues.

The API setup surface is linked out or opened separately. It is not embedded in
an iframe and is not treated as a child step of the WebUI wizard.

## Trigger Rules

Show the setup choice only when all of these are true:

- `metadata.setup_required === true`
- `metadata.setup_completed === false`
- `state.status` is one of:
  - `not_started`
  - `in_progress`
  - `first_chat_complete`
  - `blocked`

Do not show it after completed or skipped setup.

Treat `blocked` as a recovery state, not a normal WebUI wizard state. The
backend rejects first-run writes while state is `blocked`, so the choice screen
should emphasize API server setup/recovery and explain that WebUI setup can
continue after recovery. The "Set up in WebUI" action should be disabled in
blocked state until a re-check returns a mutable state.

If first-run state or metadata cannot be loaded, do not show the choice screen.
Keep the existing recovery/manual connection UI available so users can repair
their connection.

## Component Design

Add a small shared component, provisionally named `SetupEntryChoice`, under the
existing WebUI/shared onboarding area.

Inputs:

- `state: FirstRunState | null`
- `metadata: FirstRunMetadata | null`
- `configuredServerUrl?: string | null`
- `onStartWebUiSetup: () => void`
- `onRefreshSetupState: () => Promise<void> | void`

Responsibilities:

- explain the two setup surfaces in plain language
- resolve a browser-openable API server setup URL when possible
- warn when API setup is local-only or may need to be opened on the server
  machine
- open the API setup page in a new tab/window with `noopener,noreferrer` safety
- keep the current choice screen visible after opening API setup
- after opening API setup, show an "I finished API server setup" action that
  calls `onRefreshSetupState` to refresh first-run state and metadata in the
  WebUI
- provide a "Continue with WebUI setup" path into the existing wizard

State stays local to the route:

- `entryChoiceDismissed` or `setupEntryMode: "choice" | "webui"`
- Choosing WebUI flips that state and renders the existing
  `UnifiedSetupWizard`.
- A low-emphasis "Back to setup choices" action lets users return from the
  wizard to the choice screen without reloading.

## API Setup URL Resolution

Do not blindly link to `metadata.connection.api_origin`.

Reason: in Docker quickstart the WebUI may call the API through a Next.js
same-origin proxy backed by an internal container origin. That internal origin
can be correct for server-side rewrites but not openable by the user's browser.

Resolution order:

1. Use `metadata.connection.api_origin` only when it is an absolute HTTP(S)
   origin that passes the browser-openable predicate below.
2. Otherwise use the configured WebUI connection server URL only when it is an
   absolute HTTP(S) origin that passes the same predicate.
3. If no browser-openable URL is available, show a non-button fallback:
   "Open the API server setup page on the machine running tldw. For the default
   local install this is usually `http://127.0.0.1:8000/setup`."

Browser-openable predicate for this feature:

- The candidate must parse as an absolute `http:` or `https:` URL.
- The candidate origin must not equal the current WebUI page origin or
  `metadata.connection.frontend_origin`. Linking to the WebUI origin would send
  the user back to the WebUI `/setup` route, not the API server setup route.
- The candidate hostname is accepted when it is one of:
  - `localhost`, `127.0.0.1`, `[::1]`, or another loopback address
  - the same hostname as the WebUI page, with a different port
  - an RFC1918/private LAN IP address
  - a public or local DNS name with at least one dot, such as
    `tldw.local.example`
- A single-label non-loopback hostname, such as `app`, `api`, or `server`, is
  rejected because in the documented Docker quickstart that usually represents
  an internal container DNS name rather than a browser-openable host. The only
  exception is the same-hostname/different-port rule above, because the browser
  has already proven that hostname is reachable.

Examples:

| Candidate | Current WebUI origin | Result | Reason |
| --- | --- | --- | --- |
| `http://127.0.0.1:8000` | `http://127.0.0.1:8080` | Link to `http://127.0.0.1:8000/setup` | local API server origin |
| `http://localhost:8000` | `http://localhost:8080` | Link to `http://localhost:8000/setup` | local API server origin |
| `http://192.168.1.20:8000` | `http://192.168.1.20:8080` | Link to `http://192.168.1.20:8000/setup` | same LAN host, different port |
| `http://server:8000` | `http://server:8080` | Link to `http://server:8000/setup` | same browser-proven hostname, different port |
| `http://app:8000` | `http://127.0.0.1:8080` | Fallback guidance | likely Docker-internal origin |
| `http://127.0.0.1:8080` | `http://127.0.0.1:8080` | Fallback guidance | same WebUI origin would loop back to WebUI setup |
| missing or invalid | any | Fallback guidance | no safe API setup link |

The initial implementation can keep this resolver client-side. If future UAT
shows the fallback is too weak for Docker quickstart, add an explicit
browser-facing API setup URL to the WebUI runtime config endpoint.

## Local And Remote Setup Copy

Backend `/setup` is local-only by default. The choice screen must say this
before the user clicks the API setup option.

Copy rules:

- If `metadata.connection.browser_access === "local"`, say API setup should open
  locally.
- If browser access is not local and `metadata.remote_setup_enabled !== true`,
  say API setup may need to be opened on the server machine or enabled for
  remote setup by the operator.
- If `metadata.remote_setup_enabled === true`, say remote API setup access is
  enabled and may still be restricted by the server's setup allowlist.

This is explanatory text, not a hard client-side security decision. The backend
remains authoritative.

## Existing UI Adjustments

When the setup choice is active, hide the existing manual "Connect your tldw
server" card. Showing both would create three competing paths for the exact
novice workflow this change is meant to clarify.

Keep the manual connection card for recovery states:

- first-run metadata cannot be loaded
- first-run state cannot be loaded
- WebUI is not connected to the API server
- setup is not incomplete according to the strict trigger rules

The existing `SetupReadinessPanel` can remain visible after the user chooses
WebUI setup. It should not be required on the choice screen unless it can be
shown without adding noise.

## Error Handling

- If opening API setup is blocked by the browser, keep the URL visible as plain
  text so the user can copy it.
- If the resolver cannot produce a URL, show the default local URL as guidance
  but do not render it as an authoritative link.
- Returning from API server setup does not automatically complete the WebUI
  state. The choice screen should offer an explicit re-check action that calls
  the existing setup refresh path and then either keeps showing the choice,
  opens the WebUI wizard, or leaves setup according to the refreshed backend
  state.
- If WebUI setup save/validation fails, existing wizard error handling remains
  responsible.
- If backend returns `403` from API setup, the separate API server page owns that
  message. The WebUI choice screen should have already warned about local-only
  setup.

## Accessibility And UX

- Use one `h1` for the `/setup` route.
- Make the WebUI setup action visually primary.
- Make API server setup visibly secondary but not hidden.
- Button/link labels should be explicit: "Set up in WebUI" and "Open API server
  setup".
- New-tab API setup links must include accessible "opens in new tab" context.
- The layout must fit mobile and desktop without horizontal overflow.

## Testing

Add focused tests at the shared UI level:

- choice screen appears when setup is required and state is incomplete
- choice screen does not appear when setup is completed or skipped
- blocked state shows recovery-oriented copy and does not route directly into
  the normal WebUI wizard
- choosing WebUI setup renders the existing `UnifiedSetupWizard`
- "Back to setup choices" returns to the choice screen
- API setup link uses a browser-openable URL when metadata provides one
- fallback guidance appears when no browser-openable API setup URL exists
- same-hostname/different-port API setup URLs are accepted even when the
  hostname is single-label
- the "I finished API server setup" action refreshes first-run state and
  metadata
- local-only warning appears for remote browser access when remote setup is not
  enabled
- manual connection card is hidden while the choice screen is active and remains
  available for recovery/error states

Extend existing WebUI onboarding or setup-route Playwright coverage:

- first-run `/setup` opens the choice screen on desktop and mobile
- choosing WebUI setup proceeds through the existing mocked first-chat path
- API setup action is exposed as a safe new-tab link or fallback guidance,
  depending on the mocked metadata

## Non-Goals

- Do not embed API server `/setup`.
- Do not redesign the full WebUI first-run wizard.
- Do not change backend setup security.
- Do not add new backend first-run state transitions.
- Do not make remote API setup available from the client if the backend blocks
  it.
- Do not replace documentation with the setup choice screen; keep README and
  Getting Started docs as the source of full install instructions.

## Acceptance Criteria

- First-run users who land on WebUI `/setup` before backend setup is complete
  see a clear choice between WebUI setup and API server setup before the wizard.
- The UI explains which setup path to use in nontechnical language.
- The API server setup action opens separately when a browser-openable URL is
  available.
- The UI gives accurate fallback guidance when the API setup URL cannot be
  safely resolved.
- Local-only setup behavior is explained before the user leaves WebUI.
- Existing WebUI wizard behavior remains unchanged after choosing WebUI setup.
- Tests cover trigger rules, URL fallback, local-only warnings, and the WebUI
  wizard handoff.
