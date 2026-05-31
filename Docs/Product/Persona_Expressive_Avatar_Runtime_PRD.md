# Persona Expressive Avatar Runtime PRD

Status: Draft

Owner: Persona module / visual runtime integration

Tracking: #1916, split from #1902

Backlog: TASK-470

## Summary

Define the Persona-owned runtime contract for expressive avatars: how Persona live state, speech activity, tool activity, approval waits, errors, and future viseme/lip-sync cues become renderer-neutral visual intent. This PRD does not implement Buddy animation. It defines the boundary between Persona runtime events and visual renderers so current sprite-frame packs, future Live2D/Rive/3D adapters, and any Buddy shell implementation can consume the same safe contract.

The current Persona Visual Packs foundation already handles user-owned packs, manifest-backed assets, review-first generation/import, explicit activation, renderer capability reporting, and basic live visual state resolution. The remaining product gap is an explicit expressive runtime layer for richer state, speech timing, fallback, capability negotiation, and observability without creating a parallel avatar system.

## Problem

Persona Visual Packs can render current states such as idle, listening, thinking, speaking, tool-running, approval-needed, wake-armed, error, and offline. That is enough for V1 sprite-frame behavior, but it does not yet define a durable contract for richer expression:

- speech-driven mouth/viseme cues,
- high-frequency animation hints,
- renderer-specific capabilities,
- cross-renderer fallback from advanced models to static/sprite states,
- accessibility preferences and reduced-motion behavior,
- trace-safe debugging when a visual renderer cannot honor an intent.

If this is not specified, future work can drift into renderer-specific or Buddy-specific behavior embedded directly in Persona runtime code. The Persona module should emit safe expressive intent; renderer adapters should decide how to render it.

## Goals

- Define a renderer-neutral Persona Visual Runtime Intent envelope.
- Preserve the existing Persona Visual Pack ownership and activation model.
- Extend current visual state semantics without replacing the pack manifest format.
- Support future viseme/lip-sync and speech activity cues as optional renderer inputs.
- Let renderers advertise which expressive capabilities they can honor.
- Require deterministic fallback when a renderer does not support a requested cue.
- Keep runtime events bounded, trace-safe, duration-limited, and scoped to persona/session.
- Keep the first implementation backend/contract-oriented.

## Non-goals

- No Buddy-specific animation implementation in this PRD.
- No design-system backlog work.
- No implementation in this PRD slice.
- No executable user-supplied visual plugins.
- No arbitrary renderer JavaScript or unsafe SVG animation logic.
- No Live2D, Rive, Spine, Lottie, WebGL, or 3D runtime adapter as a required V1 deliverable.
- No replacement for Persona Visual Pack import/export, activation, or review flows.
- No marketplace, shared community library, or cross-user pack publishing.
- No scheduled Persona work, broad personalization memory, or multi-agent collaboration.

## Current Contract Evidence

- `Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md` records the current user-owned animated 2D visual-pack feature for Persona Buddy and Persona Live.
- `Docs/Code_Documentation/Persona_Visual_Packs.md` documents one-persona default attachment, manifest-backed packs, explicit activation, import preview, renderer capability reporting, Codex pet imports, and current `sprite_frames` limits.
- `tldw_Server_API/app/core/Persona/visuals.py` defines built-in visual states, custom `state_catalog` validation, authored triggers, fallback validation, frame timing limits, and sprite-frame manifest validation.
- `tldw_Server_API/app/core/Persona/visual_renderer_capabilities.py` exposes the renderer capability registry. `sprite_frames` is currently activatable and runtime-supported; `live2d` is listed as a disabled future renderer with static fallback requirements.
- `tldw_Server_API/app/api/v1/endpoints/persona.py` extracts bounded `persona_visuals.trigger_state` runtime overrides into trace-safe `visual_state_override` payloads scoped to persona/session.
- `tldw_Server_API/app/api/v1/schemas/persona.py` exposes renderer capability responses and Persona Visual pack/asset response contracts.
- `tldw_Server_API/app/core/Persona/README.md` defines Persona live sessions, policy/scope enforcement, websocket events, and the distinction between profile, state docs, memory, policy/scope, and exemplars.

## Product Shape

Persona Expressive Avatar Runtime should add a thin contract layer between live Persona behavior and visual renderers:

```json
{
  "type": "persona_visual_runtime_intent",
  "schema_version": 1,
  "persona_id": "persona-id",
  "session_id": "session-id",
  "event_id": "event-id",
  "state": "speaking",
  "priority": 50,
  "duration_ms": 1200,
  "reason": "assistant_delta",
  "speech": {
    "activity": "speaking",
    "sample_rate_hz": null,
    "visemes": [],
    "text_span_ref": null
  },
  "expression": {
    "mood": null,
    "intensity": 0.5,
    "gaze": null,
    "gesture": null
  },
  "fallback": {
    "state": "speaking",
    "static_asset_role": "fallback_preview"
  }
}
```

V1 does not need to fill every field. The key contract is that Persona emits bounded intent and the renderer resolves that intent according to active pack manifest, renderer capability, user preferences, and fallback rules.

## Persona-Owned Boundary

Persona owns:

- active Persona identity and selected visual pack reference,
- current live/session state,
- policy-safe runtime event emission,
- mapping live events to semantic visual intent,
- trace-safe runtime diagnostics,
- capability requirements for expressive cues.

Renderer/Buddy implementation owns:

- frame interpolation,
- animation playback,
- lip-sync rendering,
- Live2D/Rive/Spine/Lottie/WebGL/3D runtime internals,
- canvas/WebGL lifecycle,
- local performance tuning and rendering fallback details,
- visual layout and shell positioning.

This boundary keeps Persona expressive behavior reusable across the floating Buddy shell, future Persona Live surfaces, ordinary Persona-backed chat affordances, and non-sprite renderers without turning Persona runtime code into a renderer.

## Runtime State Semantics

The current built-in states remain the stable baseline:

- `idle`
- `wake_armed`
- `listening`
- `thinking`
- `speaking`
- `tool_running`
- `approval_needed`
- `error`
- `offline`

Future expressive runtime work may add optional custom states through the existing manifest `state_catalog`, but custom state IDs must keep the current safety rules: bounded identifiers, no unsafe prefixes, no secret markers, and explicit fallback to built-in states.

State priority should be deterministic. Suggested precedence:

1. `error`
2. `approval_needed`
3. explicit bounded MCP runtime override
4. `tool_running`
5. `speaking`
6. `listening`
7. `thinking`
8. `wake_armed`
9. `offline`
10. `idle`

Implementation planning should verify this order against current Buddy runtime behavior before adopting it.

## Speech, Visemes, And Lip-Sync

Speech-driven expression should be optional and capability-gated. The Persona runtime may expose:

- coarse `speaking` state,
- speech activity start/stop events,
- text-span references for generated assistant deltas,
- optional bounded viseme timelines when a TTS/STT backend provides them,
- optional amplitude/activity hints when local audio analysis exists.

V1 should not require real microphone input, a TTS provider, or an external lip-sync library. If no speech timing is available, renderers fall back to the existing `speaking` state animation. Viseme data must be bounded by count, duration, and allowed labels; raw audio buffers should not be placed in Persona runtime intent events.

## Renderer Capability Extensions

The existing renderer capability endpoint should remain the source of truth. Future capability fields can be additive:

- `supports_runtime_intent`
- `supported_intent_schema_versions`
- `supported_visual_states`
- `supports_custom_state_catalog`
- `supports_visemes`
- `supported_viseme_sets`
- `supports_speech_activity`
- `supports_expression_intensity`
- `supports_gaze`
- `supports_gesture`
- `supports_reduced_motion`
- `max_runtime_event_rate_hz`
- `max_visemes_per_event`
- `requires_static_fallback`

Unsupported renderers should be visible, disabled, and explicit, just as `live2d` is currently reported with `runtime_adapter_not_implemented`. A renderer appearing in API type unions must not be treated as supported until capability reporting, manifest validation, import preview, activation, and runtime playback agree.

## Data And API Direction

Preferred V1 contract additions:

- Runtime intent schema in Persona API schemas.
- Websocket event for visual runtime intent, emitted only when Persona Visual runtime is enabled for the session.
- Optional REST/debug endpoint for the latest bounded runtime visual diagnostics.
- Renderer capability response extensions for expressive features.
- Tests proving unsupported fields degrade to current state-based rendering.

The runtime event should not create new durable pack rows. Durable visual changes should continue to use existing draft/review/activation flows. Transient runtime intent is session-scoped and should expire by duration or state transition.

## Fallback And Accessibility

Every expressive cue must degrade:

- unsupported viseme timeline -> `speaking` state,
- unsupported gesture/gaze -> current semantic state,
- unsupported custom state -> declared fallback chain or built-in state,
- broken active pack -> derived/static Buddy summary,
- missing renderer adapter -> fallback preview/static/sprite path,
- reduced motion enabled -> static pose or low-frame-rate state animation.

Accessibility preferences should be treated as renderer inputs, not pack mutations. Reduced motion must not require editing or duplicating a visual pack.

## Policy And Safety

Runtime visual intent is not a permission grant. It must not authorize tool execution, write to memory, modify active packs, activate drafts, or bypass review. MCP-triggered runtime state changes remain transient, bounded, persona/session-scoped, and auditable.

Safety rules:

- no raw secrets, local paths, provider payloads, or prompt text in visual runtime diagnostics,
- bounded reason strings and identifiers,
- maximum duration and event-rate limits,
- no user-supplied executable renderer code,
- no automatic activation from runtime events,
- no cross-user or cross-persona asset resolution.

## Staged Delivery

### Stage 1: Contract Audit And Schema Design

Goal: define the expressive runtime intent envelope and renderer capability extensions.

Deliverables:

- Runtime intent schema proposal.
- Renderer capability extension proposal.
- State precedence and fallback rules.
- Safety limits for durations, event rate, viseme count, reason text, and custom labels.

### Stage 2: Backend Intent Emission

Goal: emit current state-based runtime intent without changing rendering behavior.

Deliverables:

- Websocket event for bounded runtime intent.
- Tests for speaking/listening/thinking/tool/error/approval state emission.
- Diagnostics for unsupported renderer capabilities and fallback decisions.
- Compatibility proof that existing sprite-frame rendering still works.

### Stage 3: Speech Activity And Optional Viseme Contract

Goal: add optional speech timing inputs without requiring a TTS/STT backend.

Deliverables:

- Bounded viseme timeline schema.
- Coarse speech activity fallback.
- Tests for missing timing, malformed viseme data, excessive viseme count, and reduced-motion mode.

### Stage 4: Renderer Adapter Readiness

Goal: make future advanced renderers implementable without changing Persona runtime semantics.

Deliverables:

- Capability-driven renderer adapter checklist.
- Live2D/Rive/3D disabled-state diagnostics.
- Static fallback requirements for advanced renderers.
- Import-preview/activation gates tied to renderer capability truth.

### Stage 5: Runtime QA And Observability

Goal: make expressive runtime behavior testable and debuggable.

Deliverables:

- Trace-safe visual runtime diagnostics.
- Browser/E2E coverage for fallback and reduced-motion behavior.
- Contract tests for renderer capability negotiation.
- Performance thresholds for event rate and render fallback.

## Risks

- Expressive avatar work can blur into Buddy implementation unless the Persona runtime emits intent rather than drawing frames.
- Type unions can imply support before renderer capability, validation, activation, and runtime playback are implemented.
- Viseme/lip-sync data can become high-volume or privacy-sensitive if raw audio or unbounded text is included.
- Advanced renderers may require licenses, native dependencies, WebGL, or large assets that conflict with local-first deployment.
- Reduced-motion and fallback behavior can be overlooked if the first implementation targets only animated happy paths.

## Open Questions For Implementation Planning

- Should runtime visual intent be emitted as a new websocket event type or folded into existing live status events?
- Should viseme labels use a provider-neutral set first, with provider-specific mappings in renderer adapters?
- What should the default maximum visual intent event rate be for local/self-hosted deployments?
- Should reduced-motion preferences live in Persona session preferences, user settings, or renderer-local settings?
- Which future renderer should be used as the first adapter proof: Live2D, Rive, or a minimal 3D/WebGL adapter?

## Acceptance Criteria

- Persona expressive avatar runtime is defined as renderer-neutral intent, not Buddy-specific animation implementation.
- Existing Persona Visual Pack ownership, review, activation, and portability contracts remain intact.
- Current sprite-frame state rendering remains the fallback baseline.
- Future speech/viseme and advanced renderer support are capability-gated and optional.
- Runtime visual events are bounded, trace-safe, persona/session-scoped, and duration-limited.
- Unsupported renderer features degrade deterministically.
- Buddy animation, design-system backlog work, scheduled work, personalization memory, and multi-agent collaboration remain out of scope.

## Verification Plan

- Schema tests for valid and invalid runtime intent envelopes.
- Renderer capability tests for supported, disabled, and unknown expressive features.
- Websocket tests for state-to-intent emission and disabled-mode silence.
- Fallback tests for unsupported visemes, unsupported custom states, missing active pack, broken pack, and disabled renderer adapter.
- Safety tests for overlong reason strings, unsafe custom state IDs, excessive duration, excessive event rate, and oversized viseme timelines.
- Browser/E2E tests for current sprite-frame fallback and reduced-motion behavior when a UI/runtime implementation is scoped.

## References

- `Docs/Product/Persona_Agent_Design.md`
- `Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md`
- `Docs/Code_Documentation/Persona_Visual_Packs.md`
- `Docs/Product/Persona_Backed_Chat_Startup_PRD.md`
- `Docs/Product/Workspace_Persona_Defaults_PRD.md`
- `Docs/Product/Persona_Scheduled_Work_PRD.md`
- `tldw_Server_API/app/core/Persona/README.md`
- `tldw_Server_API/app/core/Persona/visuals.py`
- `tldw_Server_API/app/core/Persona/visual_renderer_capabilities.py`
- `tldw_Server_API/app/api/v1/endpoints/persona.py`
- `tldw_Server_API/app/api/v1/schemas/persona.py`
