# Persona Transparent-Video Packs Stage 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reviewed native VP9-alpha WebM Persona packs, fallback-first browser rendering, guided local conversion Jobs, safe dsh-pet import, and current-Chatbook-compatible raster export without making video a Buddy dependency.

**Architecture:** Stage 2 consumes Stage 1's immutable revisions, complete review fingerprints, companion engine, and authenticated Blob loader. The server adds renderer-specific `video_clips` v1 validation and bounded conversion/import pipelines; the browser video adapter remains presentation-only and always brings up the nested strict `sprite_frames` v1 fallback before attempting an authenticated video Blob.

**Tech Stack:** FastAPI, Pydantic, SQLite/PostgreSQL migrations, Jobs WorkerSDK, FFmpeg/ffprobe, Pillow, Python zipfile/tarfile streaming, React 18, TypeScript, HTMLVideoElement, Vitest/Testing Library, Playwright, pytest, Bandit.

**Spec:** `Docs/superpowers/specs/2026-08-23-persona-ambient-companion-transparent-video-design.md`

## Global Constraints

- Complete and release the Stage 1 plan first; Stage 2 must use its engine, preference, generation, review, activation, and authenticated-asset interfaces unchanged unless renewed design review approves an interface change.
- Keep archive envelope `tldw.persona_visual_pack.v1` and dispatch renderer validation by `(renderer_type, manifest_version)`.
- Preferred video is silent VP9-alpha WebM, muted, inline, and without native controls.
- Activation requires at least one valid transparent idle clip, a complete strict `sprite_frames` v1 fallback for all nine built-in states, and a genuinely non-animated PNG static selection for each built-in state.
- Fallback is authoritative for reduced motion, unsupported codec/alpha, asset-auth failure, decode/play/stall failure, and missing optional state.
- Native image/video elements never receive protected API URLs directly; all bytes flow through Stage 1's bounded authenticated Blob loader.
- The browser performs a known-alpha session probe; reduced motion bypasses the probe and never loads video.
- Conversion and import use server-owned storage IDs/checksums, never client filesystem paths, shell interpolation, remote assets, npm, scripts, or package lifecycle hooks.
- FFmpeg runs with `-nostdin`, a local-file protocol allowlist, bounded resources, complete process-tree cancellation, and a final wait.
- Default cleanup occurs only after durable immutable publication and durable Job completion; `retain_source=true` preserves the source.
- dsh-pet ZIP/npm TGZ ingestion is streaming, signature-based, declarative, review-first, and rejects traversal, links, devices, duplicates, ambiguity, nested archives, bombs, oversized media, and remote references.
- Chatbook export is a separate fallback-only current `.tldw-persona-vpack`; it omits video and server-only behavior/provenance from the strict pack body.
- Licensing remains entirely outside the workflow.

---

### Task 1: Add the Native Video Contract and Cross-Database Persistence

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py`
- Create: `tldw_Server_API/app/core/Persona/visual_video.py`
- Modify: `tldw_Server_API/app/core/Persona/visuals.py`
- Modify: `tldw_Server_API/app/core/Persona/visual_manifest_assets.py`
- Modify: `tldw_Server_API/app/core/Persona/visual_renderer_capabilities.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/persona.py`
- Create: `tldw_Server_API/tests/DB_Management/test_chacha_migration_v53_persona_video.py`
- Create: `tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v53_persona_video.py`
- Create: `tldw_Server_API/tests/Persona/test_persona_visual_video.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_visual_manifest_assets.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_visuals_core.py`

**Interfaces:**
- Consumes: Stage 1 `normalize_companion_behavior`, strict `sprite_frames` v1 validator, review fingerprint, asset metadata, and immutable activation.
- Produces: schema version 53 constraints for renderer `video_clips` and asset role `video_clip`; `validate_video_clips_manifest(manifest, assets) -> PersonaVisualValidationResult`; `collect_video_manifest_asset_ids(manifest) -> set[str]`; `remap_video_manifest_assets(manifest, asset_id_map) -> dict[str, Any]`; `resolve_video_fallback_manifest(manifest) -> dict[str, Any]`; renderer-specific dispatch; TypeScript/Pydantic video manifest shapes.

- [ ] **Step 1: Write SQLite/PostgreSQL migration constraint tests**

```python
def test_v53_allows_video_renderer_and_video_asset(migrated_db, persona_id):
    pack_id = insert_pack(migrated_db, persona_id=persona_id, renderer_type="video_clips")
    insert_asset(
        migrated_db,
        pack_id=pack_id,
        persona_id=persona_id,
        asset_role="video_clip",
        mime_type="video/webm",
    )


def test_v53_still_rejects_unknown_renderer(migrated_db, persona_id):
    with pytest.raises(Exception):
        insert_pack(migrated_db, persona_id=persona_id, renderer_type="executable_pet")
```

Repeat against the established PostgreSQL migration fixture and assert existing raster rows survive.

- [ ] **Step 2: Run migration tests red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_chacha_migration_v53_persona_video.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v53_persona_video.py -v`

Expected: FAIL because schema 53 and constraints are absent.

- [ ] **Step 3: Implement migration 52→53 for both engines**

Rebuild the constrained SQLite tables using the repository's copy/rename pattern and alter equivalent PostgreSQL constraints. Preserve indexes, foreign keys, rows, `companion_behavior_json`, reviews, and active-pack uniqueness. Add only `video_clips` and `video_clip`; set `_CURRENT_SCHEMA_VERSION = 53`.

- [ ] **Step 4: Write exact video-manifest validation tests**

```python
def test_video_manifest_requires_complete_strict_fallback(video_assets):
    result = validate_video_clips_manifest(
        manifest=video_manifest(fallback_states={"idle": "fallback.still"}),
        assets=video_assets,
    )
    assert "fallback_missing_state:wake_armed" in result.errors


def test_shared_png_still_is_valid_for_all_nine_states(video_assets):
    result = validate_video_clips_manifest(
        manifest=video_manifest(fallback_states=all_states("fallback.still")),
        assets=video_assets,
    )
    assert result.is_valid is True
    assert "shared_static_fallback" in result.warnings
```

Cover exact v1 keys, state/animation references, at least one idle clip, `video/webm`/`video_clip` asset role, finite baseline `0..1`, `loop` boolean, absent `mirror_safe` => false, no audio metadata, strict nested fallback, nine states, non-animated PNG static selection order, unknown optional state filtering, and no duplicate checksums in the manifest.

- [ ] **Step 5: Implement renderer-specific validation and traversal**

```python
def validate_persona_visual_manifest(
    renderer_type: str,
    manifest_version: int,
    manifest: Mapping[str, Any],
    assets: Sequence[Mapping[str, Any]],
) -> PersonaVisualValidationResult:
    validator = {
        ("sprite_frames", 1): validate_sprite_frames_manifest,
        ("video_clips", 1): validate_video_clips_manifest,
    }.get((renderer_type, manifest_version))
    if validator is None:
        return PersonaVisualValidationResult.invalid("unsupported_renderer_manifest_pair")
    return validator(manifest=manifest, assets=assets)
```

Video asset traversal includes animation `asset_id`, optional preview/poster IDs, and every nested fallback frame/preview reference. Remapping returns a deep copy. Feed resolvable native/fallback state IDs into Stage 1 behavior validation and fingerprinting.

- [ ] **Step 6: Add capability and schema types**

Register `video_clips` v1 with `video_clip` plus raster roles, bounded totals, required static fallback, import/export support, and browser runtime support. Keep local FFmpeg/libvpx creator capability separate: missing conversion tooling disables create/import conversion routes, not playback of an already-reviewed pack. Add exact Pydantic/TypeScript state, animation, alignment, and nested fallback types; do not loosen the sprite schema.

- [ ] **Step 7: Run contract and migration suites**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_chacha_migration_v53_persona_video.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v53_persona_video.py tldw_Server_API/tests/Persona/test_persona_visual_video.py tldw_Server_API/tests/Persona/test_persona_visual_manifest_assets.py tldw_Server_API/tests/Persona/test_persona_visuals_core.py -v`

Expected: PASS, with only the established PostgreSQL environment skip when unavailable.

- [ ] **Step 8: Commit the contract**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py tldw_Server_API/app/core/Persona/visual_video.py tldw_Server_API/app/core/Persona/visuals.py tldw_Server_API/app/core/Persona/visual_manifest_assets.py tldw_Server_API/app/core/Persona/visual_renderer_capabilities.py tldw_Server_API/app/api/v1/schemas/persona.py tldw_Server_API/tests/DB_Management/test_chacha_migration_v53_persona_video.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v53_persona_video.py tldw_Server_API/tests/Persona/test_persona_visual_video.py tldw_Server_API/tests/Persona/test_persona_visual_manifest_assets.py tldw_Server_API/tests/Persona/test_persona_visuals_core.py
git commit -m "feat(persona): add video clips visual contract"
```

### Task 2: Add the Fallback-First Browser Video Adapter

**Files:**
- Modify: `apps/packages/ui/src/types/persona-visuals.ts`
- Modify: `apps/packages/ui/src/services/persona-visual-assets.ts`
- Create: `apps/packages/ui/src/components/Common/PersonaBuddy/personaVideoCapability.ts`
- Create: `apps/packages/ui/src/components/Common/PersonaBuddy/VideoClipRenderer.tsx`
- Add: `apps/packages/ui/src/assets/persona-alpha-probe.webm`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/personaCompanionEngine.ts`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualRenderers.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualDiagnostics.ts`
- Create: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVideoCapability.test.ts`
- Create: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/VideoClipRenderer.test.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaCompanionEngine.test.ts`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualRenderers.test.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts`

**Interfaces:**
- Consumes: Task 1 video types, Stage 1 renderer props/generation, `SpriteFrameRenderer`, and `acquirePersonaVisualAsset`.
- Produces: `probePersonaVideoAlphaCapability(signal) -> Promise<VideoSessionCapability>`; session cache; `VideoClipRenderer`; failure classes `session_incompatible|clip_invalid|asset_unavailable|play_rejected|stalled|stale`; renderer callbacks shared with raster.

- [ ] **Step 1: Write known-alpha probe and reduced-motion tests**

```typescript
it("bypasses the alpha probe and video loading for reduced motion", async () => {
  render(<VideoClipRenderer {...props} reducedMotion />)
  await screen.findByRole("img", { name: /persona buddy/i })
  expect(probePersonaVideoAlphaCapability).not.toHaveBeenCalled()
  expect(screen.queryByTestId("persona-buddy-video")).not.toBeInTheDocument()
})
```

Add a tiny bundled known-alpha WebM fixture/probe canvas assertion, one probe per session, unsupported codec, alpha compositing failure, and aborted probe. The test fixture remains under 100 KiB.

- [ ] **Step 2: Write fallback-first transition tests**

```typescript
it("keeps fallback visible until play and a real video frame succeed", async () => {
  render(<VideoClipRenderer {...props} />)
  await screen.findByTestId("persona-video-fallback")
  expect(screen.queryByTestId("persona-buddy-video")).not.toBeVisible()
  resolvePlay()
  presentVideoFrame()
  expect(screen.getByTestId("persona-buddy-video")).toBeVisible()
})
```

Cover previous visual until fallback ready, authenticated clip Blob capped at 64 MiB, `muted`/`playsInline`/no controls, `play()` promise, `requestVideoFrameCallback` and guarded `loadeddata/playing` fallback, maximum two video elements, same-state idempotence, one stall retry, per-clip invalidation, session disablement, transient auth failure not marking corrupt, stale callback cleanup, and URL revocation. Add engine integration cases proving successful turn completion commits facing, failed turn plus mirror-safe movement may commit facing, and failed turn plus `mirror_safe=false` preserves facing.

- [ ] **Step 3: Run adapter tests red**

Run: `cd apps/packages/ui && bunx vitest run src/components/Common/PersonaBuddy/__tests__/personaVideoCapability.test.ts src/components/Common/PersonaBuddy/__tests__/VideoClipRenderer.test.tsx src/components/Common/PersonaBuddy/__tests__/personaVisualRenderers.test.tsx`

Expected: FAIL because video adapter/probe are absent.

- [ ] **Step 4: Implement session probe and scoped failure cache**

```typescript
export type VideoSessionCapability =
  | { enabled: true }
  | { enabled: false; reason: "codec_unsupported" | "alpha_probe_failed" | "probe_decode_failed" }

export async function probePersonaVideoAlphaCapability(
  signal?: AbortSignal,
): Promise<VideoSessionCapability> {
  if (cachedCapability) return cachedCapability
  cachedCapability = await runKnownAlphaProbe(signal)
  return cachedCapability
}
```

Do not log Blob URLs, auth headers, or local paths. Cache session incompatibility globally; cache corrupt clips by `packId:packVersion:assetId`; keep network/auth failures retryable.

- [ ] **Step 5: Implement fallback-first VideoClipRenderer**

Render the nested strict fallback with `SpriteFrameRenderer` first. Acquire the video Blob only after fallback readiness. Fence every async boundary by generation. Set `muted`, `playsInline`, `preload="auto"`, and no `controls`. Swap only after successful play and a presented frame. Release stale sources promptly and retain at most current/next elements.

```typescript
const isCurrent = () => generation === generationRef.current
await video.play()
await waitForPresentedFrame(video, signal)
if (!isCurrent()) {
  clipHandle.release()
  return
}
setPresented({ stateId: requestedState, assetId: animation.asset_id, clipHandle })
```

Pass renderer completion/failure back to the Stage 1 engine's existing turn transition. Mirroring applies only when `mirror_safe === true`; the outer Buddy owns position/facing.

- [ ] **Step 6: Add safe diagnostics and renderer registration**

Diagnostics include pack/revision/state/renderer/failure class and counters for fallback success, video swap, session disablement, decode, stall, play rejection, and stale generation. Register `video_clips` without changing Stage 1 engine selection.

- [ ] **Step 7: Run adapter suite**

Run: `cd apps/packages/ui && bunx vitest run src/components/Common/PersonaBuddy/__tests__/personaVideoCapability.test.ts src/components/Common/PersonaBuddy/__tests__/VideoClipRenderer.test.tsx src/components/Common/PersonaBuddy/__tests__/personaCompanionEngine.test.ts src/components/Common/PersonaBuddy/__tests__/personaVisualRenderers.test.tsx src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts src/services/__tests__/persona-visual-assets.test.ts`

Expected: PASS.

- [ ] **Step 8: Commit browser video fallback**

```bash
git add apps/packages/ui/src/types/persona-visuals.ts apps/packages/ui/src/services/persona-visual-assets.ts apps/packages/ui/src/assets/persona-alpha-probe.webm apps/packages/ui/src/components/Common/PersonaBuddy/personaVideoCapability.ts apps/packages/ui/src/components/Common/PersonaBuddy/VideoClipRenderer.tsx apps/packages/ui/src/components/Common/PersonaBuddy/personaCompanionEngine.ts apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualRenderers.tsx apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualDiagnostics.ts apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVideoCapability.test.ts apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/VideoClipRenderer.test.tsx apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaCompanionEngine.test.ts apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualRenderers.test.tsx apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts
git commit -m "feat(ui): render persona video with fallback first"
```

### Task 3: Build Bounded Media Inspection, Proposal, Conversion, and Validation

**Files:**
- Create: `tldw_Server_API/app/core/Persona/visual_media.py`
- Create: `tldw_Server_API/app/core/Persona/visual_conversion.py`
- Create: `tldw_Server_API/app/core/Persona/visual_subprocess.py`
- Create: `tldw_Server_API/tests/Persona/test_persona_visual_media.py`
- Create: `tldw_Server_API/tests/Persona/test_persona_visual_conversion.py`
- Create: `tldw_Server_API/tests/Persona/test_persona_visual_subprocess.py`
- Add: `tldw_Server_API/tests/Persona/fixtures/video/green_screen_short.webm`
- Add: `tldw_Server_API/tests/Persona/fixtures/video/transparent_short.webm`
- Add: `tldw_Server_API/tests/Persona/fixtures/video/static_fallback.png`
- Add: `tldw_Server_API/tests/Persona/fixtures/video/animated_not_static.webp`

**Interfaces:**
- Consumes: Server-resolved local source/output paths and fixed executable discovery.
- Produces: `PersonaVisualMediaProbe`; `PersonaVisualConversionControls`; `PersonaVisualProposal`; `inspect_persona_visual_media(path, limits)`; `propose_chroma_key(samples, controls)`; `build_persona_visual_ffmpeg_argv(input_path, output_path, controls) -> list[str]`; `run_bounded_visual_process(argv, limits, cancel_check)`; `validate_persona_video_output(path, limits)`; `validate_persona_static_png(path)`.

- [ ] **Step 1: Write deterministic inspection/proposal tests**

```python
def test_hsv_border_proposal_uses_multiple_timestamps(media_probe):
    proposal = propose_chroma_key(
        samples=media_probe.sampled_frames,
        controls=PersonaVisualProposalControls(sample_count=5),
    )
    assert proposal.key_color == "#00ff00"
    assert 0.0 <= proposal.confidence <= 1.0
    assert proposal.sample_timestamps_ms == [0, 250, 500, 750, 1000]


def test_low_confidence_never_auto_confirms(mixed_border_probe):
    proposal = propose_chroma_key(mixed_border_probe.sampled_frames)
    assert proposal.requires_manual_confirmation is True
```

Cover deterministic border HSV sampling, crop/scale/baseline proposals, bounded timestamps, dimensions/duration/frame count/stream inventory, and no model/network invocation.

- [ ] **Step 2: Write argv and process-tree tests**

```python
def test_ffmpeg_argv_is_fixed_and_local_only(tmp_path):
    argv = build_persona_visual_ffmpeg_argv(
        input_path=tmp_path / "source.webm",
        output_path=tmp_path / "output.webm",
        controls=valid_controls(),
    )
    assert argv[1:5] == ["-nostdin", "-protocol_whitelist", "file", "-v"]
    assert "-an" in argv
    assert "libvpx-vp9" in argv
    assert all("http://" not in item and "https://" not in item for item in argv)
```

Mock the process to assert `start_new_session=True` on POSIX/new process group on Windows, CPU/memory/wall/output bounds, TERM then KILL escalation, whole-group cancellation, and final `wait()` after success/cancel/timeout. Reject non-allowlisted executable/paths/protocols before spawn.

- [ ] **Step 3: Write output rejection tests**

Test all-opaque, all-transparent, empty visible bounds, audio-bearing, animated static, wrong codec/pixel format, excessive canvas/duration/frame count/bytes, and invalid baseline. A valid output must have sampled transparent and visible pixels, VP9 alpha metadata, no audio, sane visible bounds, and a non-animated PNG fallback.

- [ ] **Step 4: Run media/conversion tests red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_media.py tldw_Server_API/tests/Persona/test_persona_visual_conversion.py tldw_Server_API/tests/Persona/test_persona_visual_subprocess.py -v`

Expected: FAIL because the modules are absent.

- [ ] **Step 5: Implement controls and fixed FFmpeg arrays**

```python
@dataclass(frozen=True)
class PersonaVisualConversionControls:
    key_color: str
    tolerance: float
    spill_suppression: float
    crop_x: float
    crop_y: float
    crop_width: float
    crop_height: float
    scale: float
    baseline: float


def build_persona_visual_ffmpeg_argv(input_path: Path, output_path: Path, controls: PersonaVisualConversionControls) -> list[str]:
    return [
        resolve_ffmpeg_executable(),
        "-nostdin",
        "-protocol_whitelist", "file",
        "-v", "error",
        "-i", str(input_path),
        "-an",
        "-vf", build_fixed_filter_graph(controls),
        "-c:v", "libvpx-vp9",
        "-pix_fmt", "yuva420p",
        "-auto-alt-ref", "0",
        "-y", str(output_path),
    ]
```

Normalize and bound tolerance/spill `0..1`, crop values `0..1`, scale `0.1..4`, baseline `0..1`. Filter graph fragments are produced only from parsed numeric values and validated `#RRGGBB`; never accept raw filter text.

- [ ] **Step 6: Implement process and decode validation**

Use `asyncio.create_subprocess_exec`, no shell. On POSIX apply `RLIMIT_CPU` and `RLIMIT_AS` in a narrowly tested child setup plus `start_new_session=True`; on Windows use the repository's supported process-group path and capability blocker when enforceable memory limits are unavailable. Poll Job cancellation and output size; terminate the group, escalate, and await process exit. Probe output with fixed ffprobe JSON arguments and sample bounded decoded RGBA frames.

- [ ] **Step 7: Run deterministic tests and optional real FFmpeg smoke**

Run unit tests:

`source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_media.py tldw_Server_API/tests/Persona/test_persona_visual_conversion.py tldw_Server_API/tests/Persona/test_persona_visual_subprocess.py -v`

Run real smoke when FFmpeg capability is present:

`source .venv/bin/activate && PERSONA_VISUAL_REAL_FFMPEG_TEST=1 python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_conversion.py -v -m external_tool`

Expected: unit tests PASS; real smoke PASS when enabled and capable, otherwise its explicit capability skip.

- [ ] **Step 8: Commit the converter core**

```bash
git add tldw_Server_API/app/core/Persona/visual_media.py tldw_Server_API/app/core/Persona/visual_conversion.py tldw_Server_API/app/core/Persona/visual_subprocess.py tldw_Server_API/tests/Persona/test_persona_visual_media.py tldw_Server_API/tests/Persona/test_persona_visual_conversion.py tldw_Server_API/tests/Persona/test_persona_visual_subprocess.py tldw_Server_API/tests/Persona/fixtures/video
git commit -m "feat(persona): add bounded transparent video conversion"
```

### Task 4: Add Staging, Preview, Conversion Jobs, Publication, and API Lifecycle

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/PersonaVisualConversion_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/Persona/visual_jobs.py`
- Create: `tldw_Server_API/app/core/Persona/visual_conversion_worker.py`
- Modify: `tldw_Server_API/app/core/Persona/visual_jobs_worker.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/persona.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/persona.py`
- Create: `tldw_Server_API/tests/Persona/test_persona_visual_conversion_db.py`
- Create: `tldw_Server_API/tests/DB_Management/test_chacha_migration_v54_persona_visual_conversion.py`
- Create: `tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v54_persona_visual_conversion.py`
- Create: `tldw_Server_API/tests/Persona/test_persona_visual_conversion_worker.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_visual_jobs.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`

**Interfaces:**
- Consumes: Tasks 1–3 contracts, Stage 1 review/publication APIs, existing Jobs WorkerSDK and user database/storage roots.
- Produces: schema version 54 user-owned conversion source/preview records; Job types `persona_visual_analyze_source`, `persona_visual_preview_conversion`, `persona_visual_convert_publish`; capability/source/analyze/preview/submit/status/cancel/retry routes; durable inactive pack plus review; cleanup/expiry lifecycle.

- [ ] **Step 1: Write repository and API lifecycle tests**

```python
def test_source_upload_returns_storage_identity_not_local_path(client, auth_headers, green_clip):
    response = client.post(
        "/api/v1/persona/profiles/persona-1/visual-conversions/sources",
        headers=auth_headers,
        files={"file": ("green.webm", green_clip, "video/webm")},
    )
    assert response.status_code == 201
    assert "source_id" in response.json()
    assert "path" not in response.json()


def test_capability_reports_actionable_blockers(client, auth_headers, monkeypatch):
    monkeypatch.setattr(capability, "resolve_ffmpeg_executable", lambda: None)
    response = client.get("/api/v1/persona/visuals/video-capability", headers=auth_headers)
    assert response.json()["creator_enabled"] is False
    assert "ffmpeg_not_found" in response.json()["blockers"]
```

Add ownership in API-key/bearer modes, upload MIME/signature/size limits, separate preview and final-conversion rate limits, source checksum revalidation, normalized controls, low-confidence manual confirmation, cancel/retry/status, expiry cleanup, and no client path acceptance.

- [ ] **Step 2: Write worker publication/idempotency/cleanup tests**

Use a fake converter and Job repo to prove idempotency key contains source checksum, normalized controls, fallback mapping, and converter version. Assert retry returns the same pack/review. Assert default source/intermediates deletion only after pack, assets, review, and completed Job are durable; `retain_source=true` preserves source; failed/interrupted work is retained until expiry; cleanup never deletes accepted WebM/fallback/manifest/review.

- [ ] **Step 3: Run lifecycle tests red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_chacha_migration_v54_persona_visual_conversion.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v54_persona_visual_conversion.py tldw_Server_API/tests/Persona/test_persona_visual_conversion_db.py tldw_Server_API/tests/Persona/test_persona_visual_conversion_worker.py tldw_Server_API/tests/Persona/test_persona_visual_jobs.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py -v`

Expected: FAIL because persistence, Job types, worker, and routes are absent.

- [ ] **Step 4: Add conversion persistence and Job payloads**

Add migration 53→54 for both SQLite and PostgreSQL. Store `source_id`, owner/persona, storage key, checksum, bytes/MIME, retention flag, status, expiry, and timestamps. Store preview generation, normalized controls/proposal JSON, confidence/manual confirmation, preview asset IDs, and status. Implement the repository through the existing backend abstraction rather than the SQLite-only portability table bootstrap, register both migrations, and set `_CURRENT_SCHEMA_VERSION = 54`.

```python
PERSONA_VISUAL_CONVERT_PUBLISH_JOB_TYPE = "persona_visual_convert_publish"


def build_visual_conversion_idempotency_key(
    *,
    user_id: str,
    source_sha256: str,
    controls: Mapping[str, Any],
    fallback_mapping: Mapping[str, Any],
    converter_version: str,
) -> str:
    digest = canonical_payload_fingerprint({
        "source_sha256": source_sha256,
        "controls": controls,
        "fallback_mapping": fallback_mapping,
        "converter_version": converter_version,
    })
    return f"persona-visuals:user:{user_id}:convert:{digest}"
```

- [ ] **Step 5: Implement capability and lifecycle routes**

Use these exact routes:

- `GET /api/v1/persona/visuals/video-capability`
- `POST /api/v1/persona/profiles/{persona_id}/visual-conversions/sources`
- `POST /api/v1/persona/profiles/{persona_id}/visual-conversions/sources/{source_id}/analyze`
- `POST /api/v1/persona/profiles/{persona_id}/visual-conversions/sources/{source_id}/previews`
- `POST /api/v1/persona/profiles/{persona_id}/visual-conversions/sources/{source_id}/jobs`
- `GET /api/v1/persona/profiles/{persona_id}/visual-conversions/jobs/{job_id}`
- `POST /api/v1/persona/profiles/{persona_id}/visual-conversions/jobs/{job_id}/cancel`
- `POST /api/v1/persona/profiles/{persona_id}/visual-conversions/jobs/{job_id}/retry`

Controls expose only key color, tolerance, spill suppression, normalized crop, scale, and baseline. Mapping covers required fallback states and optional idle/turn/move/click/drag/weighted entries.

- [ ] **Step 6: Implement worker publication and complete cleanup boundary**

Resolve the source from owner-scoped storage identity and verify checksum. Run bounded analysis/conversion, validate video/static/fallback/behavior, write accepted assets, create one immutable inactive pack, create its complete review, then mark the Job completed. Only after both durable pack transaction and durable Job completion invoke default source/intermediate cleanup.

```python
publication = publish_inactive_video_pack(
    user_id=user_id,
    persona_id=persona_id,
    source=verified_source,
    video_output=validated_video,
    fallback_output=validated_fallback,
    manifest=validated_manifest,
    companion_behavior=validated_behavior,
)
completed_job = jobs.complete(job_id, result={"pack_id": publication.pack_id, "review_id": publication.review_id})
if completed_job and not source.retain_source:
    cleanup_conversion_source(source.id, include_intermediates=True)
```

Cancellation checks run before/after each probe, subprocess, asset write, and transaction. The worker registers with existing worker startup and reports user-safe failure codes.

- [ ] **Step 7: Run lifecycle suite**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_chacha_migration_v54_persona_visual_conversion.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v54_persona_visual_conversion.py tldw_Server_API/tests/Persona/test_persona_visual_conversion_db.py tldw_Server_API/tests/Persona/test_persona_visual_conversion_worker.py tldw_Server_API/tests/Persona/test_persona_visual_jobs.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py -v`

Expected: PASS.

- [ ] **Step 8: Commit Job lifecycle**

```bash
git add tldw_Server_API/app/core/DB_Management/PersonaVisualConversion_DB.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/Persona/visual_jobs.py tldw_Server_API/app/core/Persona/visual_conversion_worker.py tldw_Server_API/app/core/Persona/visual_jobs_worker.py tldw_Server_API/app/api/v1/schemas/persona.py tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/tests/DB_Management/test_chacha_migration_v54_persona_visual_conversion.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v54_persona_visual_conversion.py tldw_Server_API/tests/Persona/test_persona_visual_conversion_db.py tldw_Server_API/tests/Persona/test_persona_visual_conversion_worker.py tldw_Server_API/tests/Persona/test_persona_visual_jobs.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py
git commit -m "feat(persona): add reviewed video conversion jobs"
```

### Task 5: Add the Guided Local Creator and Review UI

**Files:**
- Create: `apps/packages/ui/src/components/PersonaGarden/PersonaVideoPackWizard.tsx`
- Create: `apps/packages/ui/src/components/PersonaGarden/PersonaVideoPreview.tsx`
- Create: `apps/packages/ui/src/components/PersonaGarden/personaVideoWizardState.ts`
- Modify: `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
- Modify: `apps/packages/ui/src/services/persona-visuals.ts`
- Create: `apps/packages/ui/src/components/PersonaGarden/__tests__/PersonaVideoPackWizard.test.tsx`
- Create: `apps/packages/ui/src/components/PersonaGarden/__tests__/PersonaVideoPreview.test.tsx`
- Create: `apps/packages/ui/src/components/PersonaGarden/__tests__/personaVideoWizardState.test.ts`
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

**Interfaces:**
- Consumes: Task 4 API, Stage 1 review/activation client, and Stage 2 video renderer/fallback.
- Produces: a finite wizard `upload → analyze → adjust → map fallback/actions → convert → review → inactive draft`; generation-fenced preview requests; capability blockers; cancel/retry; source-retention choice.

- [ ] **Step 1: Write reducer and stale-preview tests**

```typescript
it("ignores an older preview generation", () => {
  const state = reduceWizard(initialState, { type: "previewRequested", generation: 2 })
  const next = reduceWizard(state, {
    type: "previewReady",
    generation: 1,
    preview: oldPreview,
  })
  expect(next.preview).toBeNull()
  expect(next.previewGeneration).toBe(2)
})
```

Cover finite transitions, aborted prior request, low-confidence manual confirmation, invalid controls, required nine-state fallback coverage, optional action mapping, retain-source default false, cancel/retry, and immutable result IDs.

- [ ] **Step 2: Write guided UI tests**

Render capability blockers; upload by file identity; before/after previews; only the six approved control groups; confidence/manual-confirm warning; state/action mapping review; shared-still warning; source-retention toggle; conversion progress; cancel/retry; completed inactive pack/review; explicit separate activation. Assert no licensing attestation or rights copy.

- [ ] **Step 3: Run wizard tests red**

Run: `cd apps/packages/ui && bunx vitest run src/components/PersonaGarden/__tests__/PersonaVideoPackWizard.test.tsx src/components/PersonaGarden/__tests__/PersonaVideoPreview.test.tsx src/components/PersonaGarden/__tests__/personaVideoWizardState.test.ts src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

Expected: FAIL because wizard modules are absent.

- [ ] **Step 4: Implement finite wizard state and service calls**

```typescript
export type PersonaVideoWizardStep =
  | "upload"
  | "analyze"
  | "adjust"
  | "map"
  | "convert"
  | "review"
  | "complete"

export type PersonaVideoConversionControls = {
  key_color: string
  tolerance: number
  spill_suppression: number
  crop: { x: number; y: number; width: number; height: number }
  scale: number
  baseline: number
}
```

Every preview request increments generation and aborts its predecessor. The final request sends source ID, normalized controls, mapping, manual confirmation, and `retain_source`; it never sends a path.

- [ ] **Step 5: Implement accessible previews and mapping review**

Show fallback immediately. Use the video adapter for accepted preview output. Label sliders/inputs, keep keyboard operation, announce progress/errors through a polite live region, and expose user-safe recovery actions. The review lists video alpha, all nine fallback states, proposed idle/turn/move/click/drag entries, normalization, duration/dimensions/size, omitted actions, and retention.

- [ ] **Step 6: Mount the wizard from VisualPackEditor**

Add `Create transparent video pack` only when server capability reports enabled; otherwise keep Stage 1 controls usable and show blockers. Completion selects the inactive pack and review record. Activation remains the existing separate reviewed-fingerprint action.

- [ ] **Step 7: Run wizard suite**

Run: `cd apps/packages/ui && bunx vitest run src/components/PersonaGarden/__tests__/PersonaVideoPackWizard.test.tsx src/components/PersonaGarden/__tests__/PersonaVideoPreview.test.tsx src/components/PersonaGarden/__tests__/personaVideoWizardState.test.ts src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

Expected: PASS.

- [ ] **Step 8: Commit creator UI**

```bash
git add apps/packages/ui/src/components/PersonaGarden/PersonaVideoPackWizard.tsx apps/packages/ui/src/components/PersonaGarden/PersonaVideoPreview.tsx apps/packages/ui/src/components/PersonaGarden/personaVideoWizardState.ts apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx apps/packages/ui/src/services/persona-visuals.ts apps/packages/ui/src/components/PersonaGarden/__tests__/PersonaVideoPackWizard.test.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/PersonaVideoPreview.test.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/personaVideoWizardState.test.ts apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
git commit -m "feat(ui): add guided persona video pack creator"
```

### Task 6: Add Streaming Safe Archives and the dsh-pet Review Adapter

**Files:**
- Create: `tldw_Server_API/app/core/Utils/jsonc.py`
- Modify: `tldw_Server_API/app/core/CodeGraph/extractors/js_ts_imports.py`
- Create: `tldw_Server_API/app/core/Persona/visual_portability/safe_archive.py`
- Create: `tldw_Server_API/app/core/Persona/visual_portability/dsh_pet.py`
- Modify: `tldw_Server_API/app/core/Persona/visual_portability/preview.py`
- Modify: `tldw_Server_API/app/core/Persona/visual_portability/importer.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/persona.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/persona.py`
- Create: `tldw_Server_API/tests/Utils/test_jsonc.py`
- Create: `tldw_Server_API/tests/Persona/test_persona_visual_safe_archive.py`
- Create: `tldw_Server_API/tests/Persona/test_persona_visual_dsh_pet.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_visual_portability_worker.py`
- Add: `tldw_Server_API/tests/Persona/fixtures/dsh_pet/minimal-dsh-pet.zip`
- Add: `tldw_Server_API/tests/Persona/fixtures/dsh_pet/minimal-dsh-pet.tgz`

**Interfaces:**
- Consumes: Tasks 1, 3, and 4 video inspection/conversion/review pipeline; existing import preview/commit Jobs.
- Produces: shared `strip_jsonc_comments(text) -> str`; `SafeArchiveLimits`; `inspect_safe_archive(path, limits) -> SafeArchiveIndex`; `open_archive_member(index, name) -> BinaryIO`; `inspect_dsh_pet_archive(path, limits) -> DshPetImportProposal`; dsh preview/commit routes and proposed mapping.

- [ ] **Step 1: Promote the state-machine JSONC stripper with regression tests**

```python
def test_jsonc_stripper_preserves_comment_markers_inside_strings():
    raw = '{"url":"https://example.invalid/a//b","note":"/* text */"}// tail\n'
    assert json.loads(strip_jsonc_comments(raw)) == {
        "url": "https://example.invalid/a//b",
        "note": "/* text */",
    }
```

Move the existing state-machine implementation from CodeGraph to `core/Utils/jsonc.py` and import it from both CodeGraph and dsh adapter. Test line/block comments, escapes, unterminated comments, and strings. Do not add JSON5.

- [ ] **Step 2: Write hostile streaming archive tests**

Construct ZIP/TGZ cases for traversal, absolute/backslash paths, symlink, hard link, device, duplicate normalized path, encrypted member, nested archive suffix/signature, remote URI, ambiguous configs, member count, per-member bytes, total expanded bytes, compression ratio, and chunked reads. Assert inspection does not call `read()` without a bounded size and accepts root or `package/` prefix.

- [ ] **Step 3: Write complete dsh mapping tests**

```python
def test_dsh_plural_pools_map_to_native_states(dsh_archive):
    proposal = inspect_dsh_pet_archive(dsh_archive, limits=test_limits())
    states = {item.native_state for item in proposal.mappings}
    assert "idle" in states
    assert any(state.startswith("ambient.idle.") for state in states)
    assert any(state.startswith("ambient.turn.") for state in states)
    assert any(state.startswith("ambient.move.") for state in states)
    assert any(state.startswith("reaction.click.") for state in states)
    assert any(state.startswith("reaction.drag.") for state in states)
```

Cover `animations.idle`, `animations.turn`, `animations.moves.default`, `animations.moves.actions`, `animations.clicks`, `animations.drag`, `categories`, top-level `animationWeights`, `noMirror`, invalid numerics, Unicode labels separate from IDs, stable short-digest collision prevention, lead/tail duration ratios, ignored position/multiplicity, first size as review hint, and relative weights not totaling 100.

- [ ] **Step 4: Run archive/adapter tests red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Utils/test_jsonc.py tldw_Server_API/tests/Persona/test_persona_visual_safe_archive.py tldw_Server_API/tests/Persona/test_persona_visual_dsh_pet.py -v`

Expected: FAIL because shared JSONC/archive/adapter modules are absent.

- [ ] **Step 5: Implement streaming ZIP/TGZ boundary**

Detect format by signature. Build an index without extraction; normalize with `PurePosixPath`; reject links/devices/duplicates/nested archives and limit count/member/total/ratio. `open_archive_member` returns a wrapper that stops after declared/allowed bytes. Never invoke npm, JavaScript, shell, or remote access.

```python
@dataclass(frozen=True)
class SafeArchiveLimits:
    max_members: int = 512
    max_member_bytes: int = 64 * 1024 * 1024
    max_total_expanded_bytes: int = 256 * 1024 * 1024
    max_compression_ratio: float = 100.0


def inspect_safe_archive(path: Path, limits: SafeArchiveLimits) -> SafeArchiveIndex:
    with path.open("rb") as handle:
        signature = handle.read(4)
    if signature.startswith(b"PK\x03\x04"):
        return inspect_zip_file(path, limits)
    if is_gzip_signature(signature):
        return inspect_tgz_file(path, limits)
    raise SafeArchiveError("unsupported_archive_signature")
```

Keep every archive read behind a bounded file handle; do not materialize an archive or all expanded members in memory.

- [ ] **Step 6: Implement declarative dsh mapping and review handoff**

Sanitize state IDs to ASCII slug plus the first 10 hex characters of SHA-256 over the original label/path. Inspect every referenced local media member through Task 3 limits. Pass through only proven VP9-alpha/no-audio outputs; route every other clip into normal conversion. Produce proposed manifest, behavior, controls, clamps, ignored fields, fallback requirements, warnings, and source fingerprint. Commit only after user review and required fallback completion.

- [ ] **Step 7: Add dsh preview/commit routes and worker dispatch**

Use:

- `POST /api/v1/persona/profiles/{persona_id}/visual-packs/dsh-import-previews`
- `GET /api/v1/persona/profiles/{persona_id}/visual-packs/dsh-import-previews/{preview_id}`
- `POST /api/v1/persona/profiles/{persona_id}/visual-packs/dsh-import-previews/{preview_id}/commit`

Upload creates a user-owned staged archive/import-preview Job. GET returns mapping/clamps/ignored fields/choices. Commit revalidates source fingerprint and submits Task 4 conversion/publication; it never activates automatically.

- [ ] **Step 8: Run adapter and portability worker suites**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Utils/test_jsonc.py tldw_Server_API/tests/Persona/test_persona_visual_safe_archive.py tldw_Server_API/tests/Persona/test_persona_visual_dsh_pet.py tldw_Server_API/tests/Persona/test_persona_visual_portability_worker.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py -v`

Expected: PASS.

- [ ] **Step 9: Commit safe dsh import**

```bash
git add tldw_Server_API/app/core/Utils/jsonc.py tldw_Server_API/app/core/CodeGraph/extractors/js_ts_imports.py tldw_Server_API/app/core/Persona/visual_portability/safe_archive.py tldw_Server_API/app/core/Persona/visual_portability/dsh_pet.py tldw_Server_API/app/core/Persona/visual_portability/preview.py tldw_Server_API/app/core/Persona/visual_portability/importer.py tldw_Server_API/app/api/v1/schemas/persona.py tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/tests/Utils/test_jsonc.py tldw_Server_API/tests/Persona/test_persona_visual_safe_archive.py tldw_Server_API/tests/Persona/test_persona_visual_dsh_pet.py tldw_Server_API/tests/Persona/test_persona_visual_portability_worker.py tldw_Server_API/tests/Persona/fixtures/dsh_pet
git commit -m "feat(persona): import dsh pet packs through safe review"
```

### Task 7: Export Chatbook Fallbacks and Prove Stage 2 End to End

**Files:**
- Create: `tldw_Server_API/app/core/Persona/visual_portability/chatbook_exporter.py`
- Modify: `tldw_Server_API/app/core/Persona/visual_jobs.py`
- Modify: `tldw_Server_API/app/core/Persona/visual_jobs_worker.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/persona.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/persona.py`
- Create: `tldw_Server_API/tests/Persona/test_persona_visual_chatbook_export.py`
- Add: `tldw_Server_API/tests/Persona/fixtures/chatbook/video-fallback-golden.tldw-persona-vpack`
- Modify: `apps/packages/ui/src/services/persona-visuals.ts`
- Modify: `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`
- Modify: `apps/tldw-frontend/e2e/workflows/persona-buddy-interaction.spec.ts`
- Modify: `Docs/Code_Documentation/Persona_Visual_Packs.md`
- Modify: `Docs/Code_Documentation/Persona_Ambient_Companion.md`

**Interfaces:**
- Consumes: Native video manifest/fingerprint, strict sprite exporter, existing portability Jobs, browser adapter, creator, and dsh adapter.
- Produces: `project_chatbook_fallback_pack(pack, assets) -> ChatbookFallbackProjection`; `persona_visual_pack_chatbook_export` Job; `POST /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/exports/chatbook`; golden archive; release qualification record.

- [ ] **Step 1: Write golden projection tests**

```python
def test_chatbook_export_contains_only_reachable_raster_assets(video_pack, tmp_path):
    result = export_chatbook_fallback(video_pack, output_dir=tmp_path)
    with zipfile.ZipFile(result.archive_path) as archive:
        names = set(archive.namelist())
        assert all(not name.endswith(".webm") for name in names)
        pack = json.loads(archive.read("metadata/pack.json"))
        assert pack["pack"]["renderer_type"] == "sprite_frames"
        assert "companion_behavior" not in pack["pack"]
```

Assert exact current Chatbook sprite v1 schema, nine states, static-to-one-frame conversion, reachable raster assets only, rewritten paths, exact checksums, no video/remote/dependencies/unused files, fallback-edition title, preserved Persona identity, omitted-action review, and native fingerprint in outer server envelope/export record but not strict pack body. Compare normalized archive content to the checked-in golden fixture.

- [ ] **Step 2: Run export tests red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_chatbook_export.py -v`

Expected: FAIL because the projection exporter is absent.

- [ ] **Step 3: Implement independent fallback projection and Job route**

Reuse the existing strict sprite exporter after extracting and validating `fallback_manifest`. Collect only its reachable assets. Validate the produced archive through the current server importer before making it downloadable. Add an export record with native fingerprint and omitted states/actions.

```python
def project_chatbook_fallback_pack(
    pack: Mapping[str, Any],
    assets: Sequence[Mapping[str, Any]],
) -> ChatbookFallbackProjection:
    fallback = resolve_video_fallback_manifest(pack["manifest"])
    validate_strict_sprite_frames_v1(fallback, assets)
    reachable = select_assets(assets, collect_visual_manifest_asset_ids(fallback))
    return ChatbookFallbackProjection(
        title=f"{pack['title']} (Fallback Edition)",
        persona_id=pack["persona_id"],
        manifest=fallback,
        assets=reachable,
        native_pack_fingerprint=build_persona_visual_pack_fingerprint(pack, assets),
        omitted_states=video_only_states(pack["manifest"], fallback),
    )
```

The UI button says `Export Chatbook fallback` and displays omissions before starting the Job.

- [ ] **Step 4: Add browser/E2E video failure coverage**

Extend Playwright with one small real transparent-video Chromium smoke. Add fallback-first checks, reduced-motion no video request, unsupported session fallback, Persona/pack switch stale callback, play rejection/stall fallback, same-state idempotence, and transient roaming. Keep WebKit alpha as a mocked contract test, not required real-media CI.

- [ ] **Step 5: Document setup, security, lifecycle, import, and export**

Document FFmpeg/libvpx capability blockers, controls/bounds, default deletion/retain option, publication boundary, review/activation, renderer failure scopes, dsh accepted concepts/ignored fields, archive limits, Chatbook fallback-only semantics, and manual current-Chatbook import qualification. State that Chatbook video remains unsupported and licensing is out of scope.

- [ ] **Step 6: Run the full focused verification matrix**

Backend:

`source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_video.py tldw_Server_API/tests/Persona/test_persona_visual_media.py tldw_Server_API/tests/Persona/test_persona_visual_conversion.py tldw_Server_API/tests/Persona/test_persona_visual_subprocess.py tldw_Server_API/tests/Persona/test_persona_visual_conversion_db.py tldw_Server_API/tests/Persona/test_persona_visual_conversion_worker.py tldw_Server_API/tests/Persona/test_persona_visual_safe_archive.py tldw_Server_API/tests/Persona/test_persona_visual_dsh_pet.py tldw_Server_API/tests/Persona/test_persona_visual_chatbook_export.py tldw_Server_API/tests/Persona/test_persona_visual_portability_worker.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py tldw_Server_API/tests/DB_Management/test_chacha_migration_v53_persona_video.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v53_persona_video.py tldw_Server_API/tests/DB_Management/test_chacha_migration_v54_persona_visual_conversion.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v54_persona_visual_conversion.py -v`

Frontend:

`cd apps/packages/ui && bunx vitest run src/components/Common/PersonaBuddy/__tests__ src/components/PersonaGarden/__tests__/PersonaVideoPackWizard.test.tsx src/components/PersonaGarden/__tests__/PersonaVideoPreview.test.tsx src/components/PersonaGarden/__tests__/personaVideoWizardState.test.ts src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx src/services/__tests__/persona-visual-assets.test.ts`

Lint/typecheck:

`cd apps/tldw-frontend && bunx eslint ../packages/ui/src/components/Common/PersonaBuddy ../packages/ui/src/components/PersonaGarden/PersonaVideoPackWizard.tsx ../packages/ui/src/components/PersonaGarden/PersonaVideoPreview.tsx ../packages/ui/src/components/PersonaGarden/personaVideoWizardState.ts ../packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx ../packages/ui/src/services/persona-visuals.ts ../packages/ui/src/types/persona-visuals.ts`

`cd apps/tldw-frontend && bunx tsc --noEmit`

E2E:

`cd apps/tldw-frontend && bunx playwright test e2e/workflows/persona-buddy-interaction.spec.ts --project=chromium --reporter=line`

Expected: all commands PASS, with established capability/PostgreSQL skips only where the environment lacks them.

- [ ] **Step 7: Run security, fixture, and compatibility gates**

`source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Persona tldw_Server_API/app/core/DB_Management/PersonaVisualConversion_DB.py tldw_Server_API/app/api/v1/endpoints/persona.py -f json -o /tmp/bandit_persona_video_stage2.json`

`find tldw_Server_API/tests/Persona/fixtures/video tldw_Server_API/tests/Persona/fixtures/dsh_pet tldw_Server_API/tests/Persona/fixtures/chatbook -type f -size +2M -print`

`git diff --check`

Expected: Bandit exits 0; fixture-size command prints nothing; diff check exits 0. Import the golden `.tldw-persona-vpack` manually into the current Chatbook release and record pass/fail, Chatbook version, and date in release qualification notes.

- [ ] **Step 8: Commit Stage 2 release proof and documentation**

```bash
git add tldw_Server_API/app/core/Persona/visual_portability/chatbook_exporter.py tldw_Server_API/app/core/Persona/visual_jobs.py tldw_Server_API/app/core/Persona/visual_jobs_worker.py tldw_Server_API/app/api/v1/schemas/persona.py tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/tests/Persona/test_persona_visual_chatbook_export.py tldw_Server_API/tests/Persona/fixtures/chatbook/video-fallback-golden.tldw-persona-vpack apps/packages/ui/src/services/persona-visuals.ts apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx apps/tldw-frontend/e2e/workflows/persona-buddy-interaction.spec.ts Docs/Code_Documentation/Persona_Visual_Packs.md Docs/Code_Documentation/Persona_Ambient_Companion.md
git commit -m "test(persona): verify video pack creation and compatibility"
```

## Stage 2 Completion Gate

- [ ] `video_clips` v1 dispatch, fallback validation, asset traversal/remapping, SQLite, and PostgreSQL constraints pass.
- [ ] The browser shows fallback first, enables video only after a known-alpha probe, scopes failures correctly, and never loads video under reduced motion.
- [ ] Conversion is deterministic, local-only, bounded, cancellable, process-tree-safe, silent, alpha-validated, and idempotent.
- [ ] Publication creates one immutable inactive reviewed revision and cleanup begins only after durable pack and Job completion.
- [ ] Guided controls are limited to key color, tolerance, spill, crop, scale, and baseline; low-confidence proposals require confirmation.
- [ ] dsh ZIP/TGZ input is streaming and non-executable; all approved pools/weights/mirroring/motion ratios map through review.
- [ ] Chatbook export is raster-only, exact current sprite v1, reachable-assets-only, independently validated, and traceable to the native fingerprint.
- [ ] Focused backend, frontend, real Chromium media smoke, E2E, lint, typecheck, Bandit, fixture-size, diff, and manual Chatbook qualification checks pass.
