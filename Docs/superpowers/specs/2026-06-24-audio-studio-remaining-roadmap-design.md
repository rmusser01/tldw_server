# Audio Studio Remaining Work Roadmap

Date: 2026-06-24
Status: Reviewed, pending user review
Backlog: TASK-2355
Related spec: Docs/superpowers/specs/2026-06-23-audio-studio-design.md
Related implementation plan: Docs/superpowers/plans/2026-06-23-audio-studio-mvp-implementation-plan.md

## Summary

Audio Studio now has the core server-backed MVP shape: `/audio-studio`, `/audiobook-studio` compatibility routing, project persistence, provider-backed generation jobs, render/export backend services, migration compatibility, and an initial timeline editor slice.

The remaining work should prioritize a release-stable creator experience for the first-class spoken-audio workflows while hardening the shared platform contracts underneath them. Narration, Podcast, and Briefings are product pillars. Music and sound generation, including ACE-Step, should validate and extend the adapter model without pulling priority away from dependable spoken-audio creation.

This roadmap supersedes the 2026-06-23 Audio Studio spec's MVP timing for first-class music generation and ACE-Step. It preserves the adapter-pattern, external HTTP provider, allowlisting, and secret-handling requirements from that spec, but moves ACE-Step/music expansion behind creator MVP stabilization and platform hardening. It does not move basic timeline placement, per-clip volume, mute state, or fade-in/out controls out of MVP stabilization; those remain part of the current Audio Studio baseline and should be hardened where needed.

The roadmap is organized around five phases:

1. Stabilize the creator MVP.
2. Harden the platform.
3. Complete spoken workflows.
4. Add practical editing depth.
5. Expand music and sound generation.

## Goals

- Make Narration, Podcast, and Briefings usable end-to-end: create, generate, play, arrange, render, and export.
- Keep `/audiobook-studio` as a compatibility route until migration is stable and trustworthy.
- Add authorized artifact playback and download as a foundation for review, migration, timeline preview, waveform work, render confidence, and exports.
- Add a minimum versioned provider capability contract before deeper workflow UI depends on provider-specific behavior.
- Preserve strict external-service allowlisting and secret handling.
- Keep ACE-Step as one external HTTP adapter behind the shared provider system.
- Define clear follow-up slices that are reviewable and testable.

## Non-Goals

- No full DAW scope in the near term.
- No standalone music composition surface before spoken workflows and platform contracts are stable.
- No arbitrary provider URLs, headers, callback URLs, or secrets from client payloads.
- No automatic deletion of legacy local Audiobook Studio data.
- No silent paid-provider retries that can duplicate charges or artifacts.
- No provider-specific frontend branching where the capability schema can describe the control.

## Guiding Decisions

### Workflow Priority

Narration, Podcast, and Briefings remain first-class. They can share storage, job, artifact, provider, and timeline primitives, but they should not feel like labels on the same generic generation form.

- Narration is section/chapter driven.
- Podcast is speaker/segment driven.
- Briefings are source/context driven.
- Music and SFX are supporting generation types until the spoken workflows are stable.

### Adapter Pattern

Adapters describe provider capabilities, not provider brands. ACE-Step should be registered through the same external HTTP adapter registry as later music or SFX providers.

The provider capability schema should be versioned and conservative:

- Include a `schema_version`.
- Include provider `kind` values such as `speech`/`tts`, `music`, `sfx`, `voice_conversion`, `render`, and later `mastering`.
- Include supported workflows, output formats, duration limits, async/sync mode, required inputs, optional controls, presets, and safety constraints.
- Unknown capability fields must not break the UI.
- Unknown required fields should disable the affected action with a clear message instead of rendering a broken form.

Implementation planning should align the final speech capability names with the existing Audio Studio/provider vocabulary, such as `speech.synthesize.v1`, instead of introducing a parallel naming scheme. The `tts` label in this roadmap is shorthand for speech synthesis where the existing contracts already use speech-oriented names.

### Artifact Access

Artifact playback and download are infrastructure, not polish. Most remaining work depends on this.

Requirements:

- Backend-authorized access by user, project, and artifact.
- Support both single-user API-key mode and multi-user JWT mode.
- No raw filesystem paths exposed to the browser.
- MIME, size, and extension validation.
- Browser playback support, with range requests for longer audio where practical.
- Metadata for provider, job, source section or clip, duration, created time, MIME type, and checksum when available.
- Retention and cleanup policy for generated artifacts, render outputs, exports, and orphaned files.
- If signed temporary URLs are used, they must have short TTLs, be scoped to user/project/artifact access, and never be persisted in project JSON, export JSON, logs, or Jobs payloads.

### Jobs

Audio Studio should expose one understandable job model across workflows and providers.

Requirements:

- Stable job IDs.
- Status polling now, with future SSE/WebSocket compatibility.
- Honest progress: percentage where available, otherwise clear queued/running states.
- Sanitized failure categories: validation, provider unavailable, provider rejected request, timeout, artifact write failure, render failure, export failure, and unknown.
- Terminal-state replay so completed work is visible after reload.
- Retry rules that distinguish local/free operations from external paid calls.
- Explicit user action before retrying a paid external call that could duplicate charges or artifacts.
- Idempotency keys and source revision pins for generation, render, and export requests.

### Security

Strict allowlisting and secret handling stay non-negotiable:

- External base URLs come only from server config or approved environment settings.
- Client payloads never supply provider URLs, callback URLs, headers, or secrets.
- Logs redact secrets, auth headers, signed URLs, provider tokens, and other sensitive fields.
- SSRF protections apply to every external HTTP adapter.
- Provider/admin configuration UI can come later. Runtime provider metadata should start read-only.
- Security tests ship with each affected provider, artifact, and job slice instead of being deferred to a final cleanup pass.

### Briefing Source Authorization

Briefings must not accidentally expose source content across users or projects.

Requirements:

- Source references must be authorized before they are attached to a briefing section.
- Citation/provenance metadata must identify source ids without leaking inaccessible content.
- Regeneration from changed source material must re-check source authorization.
- Exports should include source provenance appropriate to the user's permissions.
- Source text is untrusted content. It must not be treated as instructions for LLM-assisted outline, script, or briefing generation.
- Helper generation should use the project's existing safe source/RAG prompt handling patterns so source content cannot override system, developer, application, or user instructions.
- Tests should cover source prompt-injection and instruction-override attempts for Briefing helper generation.

### Waveform Strategy

Timeline editing slice 2 should not begin until the waveform generation strategy is chosen.

Open decision before implementation:

- Server-generated waveform metadata as artifact sidecar files.
- Client-generated waveform data from authorized playable audio.
- Hybrid approach: server-generated for long files and persisted projects, client-generated for quick previews.

The decision should consider mobile performance, storage overhead, range requests, cache invalidation, and whether waveforms need to be available in exports.

### Operator Documentation

External provider slices must include setup documentation as part of the slice:

- Environment variables.
- Allowlist examples.
- Local HTTP sidecar guidance.
- Timeout and retry expectations.
- Troubleshooting for health and provider errors.
- Explicit warning against broad internal-network allowlists.

## Phase 1: Stabilize The Creator MVP

Purpose: make `/audio-studio` usable and trustworthy for spoken-audio creation.

Work:

1. Artifact playback/download foundation.
2. Minimum provider capability contract.
3. Migration blob import and compatibility reports.
4. Render/export UI integration.
5. Thin but complete MVPs for Narration, Podcast, and Briefings.

Definition of done:

- Users can create, generate, play, arrange, render, and export in Narration, Podcast, and Briefings.
- Migrated audiobook projects either play correctly or show an actionable compatibility report.
- `/audiobook-studio` remains a compatibility interstitial/fallback route with no confusing dead ends; no hard redirect should be required until migration exit criteria are met.
- The UI does not depend on provider-specific hardcoding for basic controls.
- External provider URLs and secrets remain server-controlled.
- Artifact access works in single-user API-key mode and multi-user JWT mode.
- Generated artifacts have a documented retention and cleanup policy.
- Phase 1 Briefings are limited to already-authorized or user-entered sources, source references are checked before attach, regeneration, render, or export, and helper generation treats source text as untrusted.

### Phase 1 Slice Order

1. **Artifact playback/download foundation**
   Add authorized artifact media endpoints or a signed temporary URL strategy, browser playback support, MIME/range handling, and tests for ownership and invalid artifacts.

2. **Minimum provider capability contract**
   Add a versioned read-only capability endpoint with the metadata the UI needs now: generation kind, workflow support, required fields, optional controls, output formats, and availability.

3. **Migration blob import and compatibility reports**
   Import legacy local blobs when the user explicitly provides them, dedupe imported audio, report missing blobs, and preserve local Dexie data unless the user explicitly cleans it up.

4. **Render/export UI integration**
   Wire render settings, queue render/export jobs, poll status, show completed artifacts, and expose download/open actions.

5. **Thin complete workflow MVPs**
   Keep Narration, Podcast, and Briefings visible and usable before deepening any one workflow. Each workflow should have a distinct starting structure, generation path, playback/review path, and render/export action. The Briefing MVP must use already-authorized or user-entered sources only.

## Phase 2: Harden The Platform

Purpose: make providers, jobs, artifacts, and security durable enough for expansion.

Work:

- Baseline external-provider security hardening.
- Provider health, model, and preset endpoints.
- Job lifecycle hardening.
- Adapter conformance hardening.
- Artifact authorization, range, download, retention, and orphan cleanup hardening.

Definition of done:

- Adapters are tested against a shared contract.
- External HTTP providers are allowlisted and secret-safe.
- Jobs have consistent statuses, errors, terminal replay, and safe retry/cancel semantics where supported.
- Artifact access is authorized, browser-friendly, and does not expose filesystem details.
- Provider health/capability metadata is reliable enough for UI decisions.
- Paid-call retry and idempotency rules are documented and tested.

## Phase 3: Complete Spoken Workflows

Purpose: make Narration, Podcast, and Briefings feel purpose-built instead of generic.

Narration:

- Chapter and section structure.
- Voice consistency controls.
- Pronunciation controls.
- Section-level regeneration.
- Audiobook export presets.
- Capability parity with the old Audiobook Studio where still relevant.

Podcast:

- Host and guest roles.
- Per-speaker voice profiles.
- Multi-speaker script sections.
- Intro, outro, transition, and bed slots.
- Episode templates for repeated production.

Briefings:

- Source-linked sections.
- Citation/source display.
- Regeneration from changed source context.
- Briefing presets for summaries, updates, analyst notes, and similar formats.
- Source authorization checks on attach, regenerate, and export.

Shared:

- Project templates.
- Reusable voice/provider presets.
- Batch generation.
- Section-level status.
- Regenerate one section without rebuilding the full project.

Definition of done:

- Each workflow has a distinct data structure, UX, and export defaults.
- Users can regenerate isolated sections without rebuilding full projects.
- Briefings retain source traceability without leaking unauthorized content.
- Podcast speaker setup supports repeated episode production.
- Narration remains at least as capable as the legacy Audiobook Studio.

## Phase 4: Add Practical Editing Depth

Purpose: make timeline editing useful without trying to clone a DAW.

Slice 2:

- Waveform display.
- Trim handles.
- Split.
- Keyboard nudging.

Phase 4 does not reclassify basic clip volume, mute, or fade-in/out controls as future work. Those controls are part of the existing timeline baseline. Phase 4 is for editing depth beyond that baseline.

Slice 3:

- Crossfades and richer fade editing beyond basic fade-in/out.
- Clip gain automation or normalization beyond basic per-clip volume.
- Mute and solo improvements.
- Richer preview mixing.

Later:

- Multi-clip drag/drop across tracks.
- Timeline revision conflict UI.
- Track ordering.
- Better mobile and tablet behavior.

Definition of done:

- Timeline edits persist predictably.
- Preview timing and volume align closely with final render.
- Editing operations are understandable and recoverable.
- The editor stays simple enough for the target GarageBand-lite experience.
- Waveform strategy is documented before implementation begins.

## Phase 5: Expand Music And Sound Generation

Purpose: broaden Audio Studio after spoken workflows and platform contracts are stable.

Work:

- ACE-Step external HTTP adapter through the shared provider system.
- Music and SFX capability-driven controls.
- Sound effects, beds, stingers, loops, intros, outros, and ambience.
- Integration into Podcast and Briefings before a standalone music surface.
- Standalone music composition only after supporting spoken-workflow use cases are stable.

Definition of done:

- ACE-Step is one adapter, not a special backend path.
- Music/SFX generation uses the same jobs, artifacts, allowlisting, secrets, and UI capability contracts.
- Podcast and Briefings can use generated music/SFX without disrupting the core spoken workflow.
- Provider setup docs ship with the ACE-Step slice.

## Concrete Backlog

Recommended order:

1. Artifact playback/download foundation.
2. Minimum provider capability contract.
3. Migration blob import and compatibility reports.
4. Render/export UI integration.
5. Thin complete MVP for Narration, Podcast, and Briefings.
6. Baseline external-provider security hardening.
7. Provider health/model/preset endpoints.
8. Workflow completion pass for Narration, Podcast, and Briefings.
9. Job lifecycle hardening.
10. Adapter conformance hardening.
11. Timeline editing slice 2: waveform plus trim/split.
12. Timeline editing slice 3: fades/crossfades plus richer preview mixing.
13. ACE-Step/music adapter slice.
14. Music/SFX integration into Podcast and Briefings.
15. Standalone music composition surface, only after the above is stable.

Security, auth, and redaction tests should be attached to every affected provider/job/artifact task, not deferred to one hardening task.

## Recommended Next Implementation Slice

The next implementation plan should target **Phase 1, Slice 1: artifact playback/download foundation** and include the minimum provider capability contract if it remains small enough for one reviewable change. If that combined slice grows beyond a focused PR, split it into:

1. Artifact playback/download foundation.
2. Minimum provider capability contract.

Initial acceptance criteria:

- Authorized artifact media endpoint or signed temporary URL strategy exists.
- Browser playback works for generated artifacts.
- Artifact metadata includes enough information for UI playback, render review, and future waveform work.
- Media responses use safe `Content-Type` and size metadata, never expose raw filesystem paths, and either support HTTP range/partial-content seek behavior or document a deliberate fallback for providers/artifacts that cannot support it yet.
- If the implementation uses signed temporary URLs, they are short-lived, scoped to the current user/project/artifact, and excluded from project JSON, export JSON, logs, and Jobs payloads.
- Artifact access is tested in both auth modes or the remaining auth mode gap is explicitly blocked by existing test infrastructure.
- Invalid artifact ids, wrong-project artifacts, wrong-user artifacts, unsupported MIME types, and URL/path injection attempts are rejected.
- Minimal provider capability metadata is versioned and consumed by the UI without provider-specific branching.

## Review Risks

- Phase 1 can still become too broad if artifact access, capability metadata, migration blob import, render/export UI, and all three workflow MVPs are attempted in one PR. The implementation plan should split these into narrow reviewable tasks.
- Provider capability metadata can become a premature schema if it tries to model every future music/SFX option now. Keep the first version small and extensible.
- Migration blob import can become risky if it silently mutates or deletes local data. Keep it explicit and report-oriented.
- Paid external-provider retries can surprise users. Treat paid retries as explicit user actions unless an adapter proves idempotent behavior.
- Briefing source refs need early authorization coverage because source-backed workflows are easy to accidentally over-share.
- Waveform implementation should not start until artifact playback and waveform strategy are settled.
