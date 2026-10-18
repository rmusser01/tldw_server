# SDKs and client packages

This document maps the client-side packages: the voice-assistant SDKs (with their honest maintenance states), the OpenAPI-derived frontend contract package and its drift gate, the shared UI workspace, the personal-context contract library, and the standalone MCP package. There is **no general-purpose generated REST SDK** — the only OpenAPI-derived artifact is a types package.

## Inventory (verified)

| Package | Path | State |
| --- | --- | --- |
| `tldw-voice-assistant` (Python) | `sdks/python/` | v0.1.0 alpha; websockets/aiohttp; audio + wakeword extras. **Dormant** — one commit (2026-01), no CI, not shipped in the `tldw-server` wheel |
| TS voice SDK (reference) | `sdks/typescript/` | **Source-only — no package.json/tsconfig; not installable.** Its README's `npm install @tldw/voice-assistant` does not correspond to any real package. Treat as a hand-maintained reference mirroring the Python SDK |
| `@tldw/voice-assistant-sdk` | `apps/packages/voice-assistant-sdk/` | Workspace-private TS package (client/audio/hooks/types, React hook, wake-word); one unit test; **no consumers in the workspace yet** (WIP) |
| `@tldw/api-types` | `apps/packages/api-types/` | The shared frontend API contract (ADR-065): committed `paths.d.ts` (sorted path union) + `openapi.fingerprint.json` drift gate; generated `schema.d.ts`/`openapi.json` gitignored |
| `@tldw/ui` | `apps/packages/ui/` | The shared-UI core consumed by webui + extension as `workspace:*` |
| `tldw-profile-core` (Python) | `packages/tldw_profile_core/` | v0.1.0; pydantic + rfc8785 (JCS canonical JSON), Python ≥3.11. The canonical **Personal Context contract** (models, canonical bytes, 45 pinned fixtures); shipped inside the server wheel; consumed by Personalization, Sync v2, and the personal-context endpoints |
| `mcp-unified` (Python) | `apps/mcp-unified/` | Standalone MCP runtime/gateway package, **published on PyPI (public alpha)** with its own CLIs; built/tested via Makefile targets |
| App shells | `apps/tldw-frontend` (WebUI, port 8080 host / 3000 dev), `apps/extension` (WXT browser extension), `apps/macos_virtualization_helper` (README-only stub) | |

## Voice-assistant session (dataflow — both language SDKs)

1. Client opens a WebSocket to `/api/v1/voice/assistant` and sends `AUTH`; the server also honors `?token=` on the URL (both SDKs send both — note the query-string form is a log-leak surface).
2. Server auth: a two-dot JWT verifies against the JWT service with a session-blacklist check; anything else is an API key (single-user fixed id + IP allowlist, or multi-user validation with read scope); 10 s auth timeout → close 4401; a non-AUTH first message → close 4400.
3. A voice session is created with per-user ChaChaNotes persistence (client id `voice_assistant`) — **DB init failure is non-fatal**; the session continues without persistence.
4. `CONFIG` (STT model/language, TTS format/voice, sample rate) — the server nulls provider/model hints and re-resolves via its own defaults, so SDK TTS defaults (`kokoro`/`af_heart`) are advisory only.
5. Base64 `AUDIO` frames then `COMMIT`; the server emits `TRANSCRIPTION` → `INTENT` → `ACTION_START`/`ACTION_RESULT` → `TTS_CHUNK*` → `TTS_END`; workflow subscription available. Both SDKs auto-reconnect (5 attempts).

## API contract regeneration (dataflow, ADR-065)

1. A backend change alters the OpenAPI path set → run the generator (requires a Python env that can import the server).
2. The exporter writes a canonical `openapi.json` and refreshes the committed fingerprint; `openapi-typescript` emits the gitignored rich `schema.d.ts`; the committed `paths.d.ts` is re-derived from path keys.
3. CI's fingerprint gate and `make openapi-drift-check` fail the PR until the fingerprint (and `paths.d.ts`) are refreshed in the same change; frontend-only checkouts typecheck against the committed paths with no Python — that split is the deliberate ADR-065 constraint.

## Personal-context contract (dataflow)

Contract changes are made in `packages/tldw_profile_core` **first** ("change the shared canonical object first; change Sync transport separately"); canonical bytes are JCS-normalized with pinned fixture suites; the server consumes the models across Personalization/Sync-v2/endpoints, and the chatbook implements the same contract as an external peer.

## Boundaries

- No generated callable REST SDK exists; `@tldw/api-client` (a shared request core) is ADR-065 future work.
- The voice SDKs wrap exactly one surface (`/api/v1/voice/assistant` WS + `/voice/*` REST) — distinct from the Persona Live voice runtime (ADR-046).
- `tldw_profile_core` is a contract library, not a service; Sync v2 envelopes are a separate contract.
- admin-ui is deliberately **not** a consumer of `@tldw/api-types` (separate stack; its types are hand-written).
- Single-user CORS auto-adds map: `:8080` WebUI host, `:3000` WebUI dev/extension server, `:3001` admin-ui — auto-add happens only in single-user mode with origins unset.

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| WS auth timeout / bad credentials | `AUTH_ERROR` then close 4401 |
| First message not AUTH | Close 4400 |
| Voice DB init failure | Warning only; session continues unpersisted |
| Workflow cancel of a foreign run | 404 / `cancelled:false` (ownership) |
| api-types drift | CI fingerprint gate fails until regenerated |
| profile-core fixture mismatch | The pinned public-contract test fails |

## Verified gotchas

1. `sdks/typescript` is not installable and its README misleads; `Docs/User_Guides/index.md` links both SDK READMEs as if both were products.
2. `sdks/` is dormant and absent from the wheel — alpha reference code.
3. SDK TTS defaults are advisory — the server re-resolves them on CONFIG.
4. `tldw_profile_core` requires Python ≥3.11 while the voice SDK claims ≥3.9 — unrelated packages; don't unify the claims.
5. Regenerating the rich schema needs server-importable Python — the committed-paths-only split exists precisely for frontend-only checkouts.

## Governing decisions and docs

ADR-065 (`Docs/ADR/065-frontend-api-contract-package.md`), ADR-029 (PyPI bundle boundary), ADR-046 (`046-persona-live-conversation-and-voice-runtime.md` — distinct voice runtime; cite to avoid conflation), ADR-052 tangentially. Current references: `sdks/python/README.md`, `apps/packages/api-types/README.md`, `apps/packages/voice-assistant-sdk/README.md`, `apps/mcp-unified/README.md`, `apps/{DEVELOPMENT,Shared_UI_Monorepo,Testing_Guide}.md`, `Docs/Code_Documentation/Personal_Context_Developer_Guide.md`. `Docs/Code_Documentation/VoiceAssistant_Module.md` covers the core pipeline but not the SDKs — this doc closes that gap.

## Related docs

- [admin-ui.md](./admin-ui.md) — the other dedicated client surface
- [audio.md](./audio.md) — the audio stack the voice assistant rides
- [notes-sync.md](./notes-sync.md) — the Sync v2 transport beside the profile contract
