# Research Studio Capability Health Contract Design

## Summary

Research Studio currently allows users into the app when aggregate backend
health is degraded, but it does not have a backend-owned contract that proves
which Research Studio actions are safe. This design adds a PR-scoped
capability health contract and a verification requirement for real local text
artifact generation using existing saved LLM credentials.

The central design choice is ownership: the backend derives capability states;
the frontend consumes those states and gates only the relevant action
boundaries. The frontend must not infer Research Studio capability safety from
raw subsystem health payloads.

## Goals

- Add a stable Research Studio capability endpoint or payload that describes
  action-level status for source browsing, chat, text artifact generation,
  slides generation, audio summary, export/download, and sync/share.
- Keep route entry permissive for reachable degraded backend health while
  making expensive or unavailable actions honest at the point of use.
- Preserve the current `/research-studio` canonical route and legacy alias
  behavior from PR #1616.
- Require local/manual CDP verification against an authenticated backend.
- Require local/manual real text artifact generation with an actual configured
  LLM provider using existing saved credentials.
- Record provider/source/artifact evidence in the Backlog task and PR body.

## Non-Goals

- Do not require real slides generation or real audio summary generation in
  this PR unless the local environment already has those providers configured
  and cheap to run.
- Do not make real LLM generation a required CI gate. It depends on local
  credentials and provider configuration.
- Do not migrate internal `workspace-playground` storage, telemetry, export, or
  test helper identifiers.
- Do not block the whole WebUI because one Research Studio capability is
  degraded or unavailable.

## Backend Contract

Add a backend-owned capability surface for Research Studio. The preferred route
is:

```text
GET /api/v1/research-studio/capabilities
```

If router layout makes this awkward, an equivalent `/api/v1/health/research-studio`
route is acceptable, but the contract should remain product-scoped rather than
burying product semantics in generic aggregate health.

The endpoint must be authenticated and rate-limited like other user-facing app
capability surfaces. It can expose coarse capability status, but it must not be
public by default because it may reveal whether local providers, slides, audio,
or sync features are configured. Implement it through a service/helper that
uses lightweight local health collectors rather than HTTP-calling sibling
endpoints; this avoids auth recursion, network overhead, and double
serialization.

The endpoint returns a stable summary:

```json
{
  "status": "degraded",
  "ttl_seconds": 30,
  "capabilities": {
    "source_browse": {
      "status": "ready",
      "mode": "allow",
      "dependencies": ["database", "chacha_notes"]
    },
    "chat": {
      "status": "degraded",
      "mode": "warn",
      "dependencies": ["source_browse", "rag", "llm"],
      "reason_code": "rag_degraded"
    },
    "artifact_text_generation": {
      "status": "ready",
      "mode": "allow",
      "dependencies": ["source_browse", "llm"]
    },
    "slides_generation": {
      "status": "unavailable",
      "mode": "block",
      "dependencies": ["source_browse", "llm", "slides"],
      "reason_code": "slides_unavailable"
    },
    "audio_summary": {
      "status": "unknown",
      "mode": "warn",
      "dependencies": ["source_browse", "llm", "tts"],
      "reason_code": "tts_unknown"
    },
    "export_download": {
      "status": "ready",
      "mode": "allow",
      "dependencies": ["local_artifact_state"]
    },
    "sync_share": {
      "status": "unknown",
      "mode": "warn",
      "dependencies": ["sync"]
    }
  },
  "timestamp": "2026-05-13T00:00:00Z"
}
```

Stable values:

- `status`: `ready`, `degraded`, `unavailable`, or `unknown`.
- `mode`: `allow`, `warn`, or `block`.
- `reason_code`: optional, machine-readable, user-safe, and never a raw
  exception string.
- `dependencies`: stable dependency identifiers, not raw implementation
  details.
- `ttl_seconds`: optional cache hint for clients. If omitted, the frontend
  should use a short default and refresh before expensive actions.

Capability semantics:

| Status | Mode | Meaning |
|---|---|---|
| `ready` | `allow` | Backend has positive evidence that required dependencies are available. |
| `degraded` | `warn` | Backend has evidence of partial degradation, but the action can still be attempted. |
| `unavailable` | `block` | Backend has positive evidence that a required dependency is unavailable. |
| `unknown` | `warn` | Backend cannot derive a reliable answer; avoid false claims and let request-level errors remain authoritative. |

Overall `status` is for status display and diagnostics only; the frontend must
gate from the per-capability `mode`. The overall value should summarize the
worst meaningful capability state:

- `ready` when all known capabilities are ready.
- `degraded` when at least one capability is warning but no required default
  action is blocked.
- `unavailable` when core read/write workflows are blocked.
- `unknown` only when the endpoint cannot derive reliable semantics.

## Capability Derivation

The first implementation should derive conservatively from existing health
sources:

- `source_browse`: use aggregate database and ChaChaNotes/media-source
  availability where available. Aggregate AuthNZ database health alone is not
  enough to claim source browsing is ready. If exact media/source health is not
  exposed, return `unknown` with `mode: "warn"` rather than over-claiming.
- `chat`: depend on `source_browse`, RAG health, and LLM health/provider
  availability.
- `artifact_text_generation`: depend on `source_browse` and LLM health/provider
  availability.
- `slides_generation`: depend on `source_browse`, LLM health/provider
  availability, and slides health.
- `audio_summary`: depend on `source_browse`, LLM health/provider availability,
  and TTS health. Do not depend on STT unless the action being certified really
  requires transcription.
- `export_download`: allow local export/download for already-generated
  artifacts unless a backend export dependency is known unavailable.
- `sync_share`: return `unknown/warn` unless there is a real sync/share
  capability signal.

Derivation must fail closed only when a required dependency is known
unavailable. Missing or ambiguous dependency evidence should produce
`status: "unknown", mode: "warn"`, not a fabricated `block`.

Provider and subsystem details must be sanitized. The endpoint may return
stable reason codes such as `llm_unavailable`, `rag_degraded`, or
`slides_unavailable`, but it should avoid raw provider responses, secret
presence detail, local filesystem paths, and exception text.

## Frontend Consumption

Add a small Research Studio capability client/helper in the shared UI layer.
It should:

- Fetch the backend capability payload after Research Studio route entry.
- Cache the latest payload for the current page session with a short TTL.
- Refresh on route focus, manual retry, and immediately before expensive
  actions when the cached payload is stale.
- Treat unreachable or malformed capability payloads as `unknown/warn`.
- Expose a typed lookup helper for action boundaries.
- Avoid duplicating backend derivation rules in the frontend.

The route remains open when aggregate app readiness permits entry. Capability
state applies only at action boundaries.

The existing source-selection gate remains higher priority than capability
gating. If no source is selected, show the no-source guidance first. Only show
capability warnings or blocks once the user is otherwise eligible to attempt
the action. This avoids stacking unrelated warnings on first-time users.

## Action Boundary Behavior

Source browsing:

- `allow`: normal browsing.
- `warn`: browsing remains available with scoped degraded copy.
- `block`: source details show a recovery state, but the workspace shell stays
  usable.

Chat:

- `allow`: normal send behavior.
- `warn`: keep Send available with degraded status near the composer.
- `block`: disable Send and show the backend-provided reason code translated
  into user-safe copy.

Text artifact generation:

- Applies to summary, report, compare sources, timeline, flashcards, quiz, mind
  map, and data table.
- `warn`: generation remains available, but the UI tells the user the server is
  degraded before the expensive action. Prefer inline or adjacent warning copy
  over a blocking confirmation modal unless the operation is known to be costly
  enough to need explicit confirmation.
- `block`: generation buttons and regeneration menus are disabled.

Slides generation:

- Applies only to Slides.
- Slides being blocked must not block text artifacts.

Audio summary:

- Applies only to Audio Summary.
- Audio being blocked must not block text artifacts or slides.

Export/download:

- Existing generated content remains viewable locally.
- Browser-local downloads remain available when they do not need backend
  services.
- Backend-dependent exports respect `export_download`.

Sync/share:

- Show warning or block share/sync affordances only when that capability is
  explicit. Do not use generic health as a sync/share proxy.

## Error Handling

If the capability endpoint is unreachable, malformed, unauthorized, or times
out:

- Keep Research Studio open if aggregate readiness already allowed route entry.
- Treat capabilities as `unknown/warn`.
- Keep read-only UI available.
- Preserve request-level errors for the exact action that fails.
- Do not claim a specific capability is down unless the backend contract says
  so.

If the endpoint returns `block` for an action:

- Block only that action.
- Keep neighboring independent actions available.
- Show user-safe recovery copy based on `reason_code`.
- Include a retry/refresh path when the blocked state may recover without a
  configuration change.

## Local Manual Verification

This PR requires local/manual verification that is recorded in Backlog and the
PR body. It is not a CI gate.

Required authenticated CDP checks:

- Start the backend with a valid API key.
- Start WebUI against that backend.
- Use CDP/Playwright, not Computer Use.
- Seed the frontend with the valid key from the existing local configuration or
  saved credentials path.
- Verify `/research-studio` renders.
- Verify `/workspace-playground` aliases to `/research-studio`.
- Verify `/workspace-studio?tab=studio` aliases to
  `/research-studio?tab=studio`.
- Verify mobile `/research-studio?tab=studio` opens Studio.
- Verify the page can call `/api/v1/health` and the Research Studio capability
  endpoint successfully with auth.

Required real generation check:

- Use existing saved local LLM credentials; do not introduce a fake provider or
  new CI secret.
- If existing saved credentials are missing or invalid, treat that as a manual
  verification blocker for this PR rather than silently downgrading to a mock.
- Select or create a small deterministic local source.
- Generate at least one `summary` text artifact through Research Studio.
- Verify the artifact enters generation, completes, and has non-empty output.
- Record provider name, model if visible/safe, source type/title, artifact
  type, completion status, screenshot path if captured, and caveats.

Do not record secrets, full API keys, private file paths containing sensitive
names, full generated text, or raw provider error bodies. It is enough to
record a short non-sensitive excerpt length or character count plus the
completion state.

## Implementation Status

Implemented route:

```text
GET /api/v1/research-studio/capabilities
```

The shipped endpoint follows the preferred product-scoped route from this
design. It is authenticated, requires `media.read`, is rate-limited as
`research_studio.capabilities`, and uses `collect_research_studio_capabilities`
instead of HTTP-calling sibling endpoints.

Frontend implementation:

- `WorkspacePlayground` fetches the payload after route entry and passes the
  normalized capability state into Chat and Studio.
- Chat treats `chat.warn` as inline degraded copy and `chat.block` as a send
  boundary block without conflating it with a disconnected-server state.
- Studio keeps source selection as the first gate, then applies
  `artifact_text_generation`, `slides_generation`, and `audio_summary` at the
  matching output button and regeneration boundary.
- Studio refresh-checks capability state before expensive generation actions.
- Unknown or malformed capability fetches normalize to `unknown/warn`, so the
  route remains usable and request-level errors remain authoritative.

Operational adjustment from live verification:

- Audio Summary capability now uses config-level TTS readiness rather than the
  full setup health path. This keeps the capability endpoint lightweight and
  avoids initializing or downloading TTS provider assets during route entry.
- The PR includes an opt-in manual Playwright spec,
  `apps/tldw-frontend/e2e/workflows/research-studio-live-generation.manual.spec.ts`,
  for local authenticated route/capability checks and real Summary generation
  against a saved LLM provider credential.

## Automated Tests

Backend tests:

- Capability derivation maps healthy subsystem payloads to `allow`.
- RAG degraded maps chat to `warn` without blocking text generation when LLM is
  ready.
- LLM unavailable blocks chat and text generation.
- Slides unavailable blocks only slides generation.
- TTS unavailable blocks or warns only audio summary.
- Unknown/malformed dependency evidence produces `unknown/warn`, not an
  over-confident allow or unrelated block.
- Response schema excludes raw exception text, secrets, and filesystem paths.

Frontend tests:

- `allow` keeps chat and generation enabled.
- `warn` keeps actions enabled but surfaces degraded copy.
- `block` disables only the matching action boundary.
- Slides block does not disable text artifact generation.
- Audio block does not disable text artifact generation or slides.
- Capability payload failure falls back to `unknown/warn`.

CDP/local smoke:

- Authenticated route and alias checks pass.
- Capability endpoint is called with the same valid auth context as the app.
- Real summary artifact generation completes with existing saved credentials.

## Documentation And Evidence

Update:

- `Docs/Operations/Research_Studio_Trust_Status_Telemetry_Runbook.md` with the
  implemented endpoint and final semantics.
- The Research Studio implementation plan/task notes with the exact verification
  commands and local manual generation evidence.
- PR body with the capability endpoint summary and local manual generation
  evidence.

Backlog final notes must distinguish:

- automated checks,
- authenticated CDP checks,
- real local LLM generation evidence,
- local environment caveats,
- any intentionally deferred slides/audio real-provider certification.

## Risks

- Health checks can be expensive or slow if the endpoint probes too much. The
  endpoint should derive from lightweight existing health/status surfaces and
  avoid warming models or generating content.
- Provider availability can be configured but still fail on generation. The
  endpoint should communicate readiness, while request-level errors remain the
  final authority for an individual generation attempt.
- Over-blocking can make degraded but usable local workflows feel broken. Block
  only on known unavailable required dependencies.
- Under-blocking can trigger expensive failed calls. Warn on ambiguous states
  and block only explicit unavailable states.
- Stale capability payloads can over-block after recovery or over-allow after a
  provider goes down. Use TTL-based refresh plus request-level error handling.

## Open Questions

- `source_browse` remains conservative: aggregate health and ChaChaNotes/source
  database signals can allow it, but missing exact media-source evidence should
  still produce `unknown/warn`.
- Chat is currently gated as source/RAG/LLM dependent because Research Studio's
  selected-source chat path is expected to be grounded. A future direct
  selected-source-only chat path should get a separate capability if it has
  different safety conditions.
- Manual verification uses existing local single-user auth and provider
  configuration discovered from the local environment. The runbook and Backlog
  notes record evidence without copying secrets.
