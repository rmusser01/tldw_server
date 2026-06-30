# Watchlists Demo Readiness Runbook

## Environment

- Start the API in single-user mode and keep the startup log visible.
- Start the WebUI against the same API origin used for the demo.
- Use a real browser profile or the extension build only after confirming the API key is configured.
- Record the exact branch, commit, API base URL, WebUI URL, extension build path, and demo source URLs before recording.
- Do not use unverified local-only fixtures as evidence for public demo claims.

## Preflight Sources

Use real remote RSS or site sources that are reachable from the demo machine. Test each source before it appears in the script.

```bash
curl -sS \
  -H "X-API-Key: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -X POST http://127.0.0.1:8000/api/v1/watchlists/sources/test \
  --data '{"url":"https://example.com/rss.xml","source_type":"rss"}'
```

The response must show usable candidate items or explicit diagnostics that match the planned narration. Local loopback feeds such as `http://127.0.0.1`, `http://localhost`, or private LAN-only fixtures are not valid demo sources unless backend policy explicitly allows them for Watchlists source fetching in the environment being recorded.

## Provider And Voice Preflight

Confirm the configured LLM, output template, TTS provider, model, and voices before recording. A text briefing can be demo-ready without audio, but a podcast/audio claim requires both gates below to pass.

Scheduler enqueue gate:

- Create an output with `generate_audio=true`.
- The output metadata must include an audio task id or equivalent queued task reference.
- `GET /api/v1/watchlists/runs/{run_id}/audio` must return a meaningful status such as pending, running, completed, failed, skipped, or no audio requested. A generic 404 or empty response is not enough.

Final playback gate:

- Provider selection completes.
- Model selection completes.
- Voice selection completes for every speaker.
- Script generation completes.
- Per-speaker audio generation completes.
- Final mix completes.
- A playable or downloadable artifact is produced and verified in the browser.

Do not describe audio as playable, produced, or complete when only the Scheduler enqueue gate has passed.

## WebUI Same-Origin Path

1. Open `/watchlists` in the WebUI.
2. Confirm the page loads without a Next.js runtime overlay.
3. Create or verify a source using a preflighted remote URL.
4. Create a monitor that uses the backend template name `briefing_markdown`.
5. Trigger a run and verify Activity shows the run state.
6. Create a report and verify Reports shows the artifact.
7. If `generate_audio=true` is enabled, verify the Scheduler enqueue gate first, then verify final playback separately.
8. If output creation fails, show the in-app error and stop the scripted happy path.

## Extension Path

1. Build the extension and open the options page.
2. Navigate to `#/watchlists`.
3. Confirm the shared route renders Overview, Activity, and Reports.
4. Verify the same API-backed statuses are visible as in the WebUI.
5. Confirm output generation errors render inside the extension page and do not crash the route.

## Demo Script Safe Claims

- Safe after WebUI smoke passes: Watchlists can create feeds and monitors from `/watchlists` in the WebUI.
- Safe for extension only after the extension build and route smoke pass: the extension can render the shared `/watchlists` route and preserve the same Activity/Reports recovery behavior.
- Safe: Briefing monitor payloads use the backend `briefing_markdown` template name.
- Safe: Reports can show pending, failed, skipped, or completed audio states without implying playback exists.
- Safe: Source and run failures can put Watchlists into an attention state.
- Unsafe unless the final playback gate passes: claiming a complete multi-voice podcast or downloadable final mix.
- Unsafe unless source preflight passes: claiming arbitrary localhost or loopback feeds work as demo inputs.

## Hard Stops

- Source preflight fails for the planned source.
- The WebUI or extension route shows a framework runtime overlay.
- Monitor creation uses `briefing_md` instead of `briefing_markdown`.
- Output creation failure crashes the page instead of rendering an in-app error.
- `generate_audio=true` does not produce a task id or meaningful `/runs/{run_id}/audio` status.
- The script claims playable audio before provider, model, voice, script, per-speaker audio, and final mix have completed.

## Known Degradations

- Scheduler enqueue is not final playback. Treat it as queued/in-progress evidence only.
- Audio status may truthfully be pending, failed, or skipped during a demo; narrate that state directly.
- Some remote sources may block scraping or return no usable items. Replace the source or show the diagnostic rather than retrying silently.
- Extension verification depends on a current extension build and seeded server configuration.
- Current verification on 2026-05-21: the Chrome MV3 extension build completed and `tests/e2e/watchlists.spec.ts` passed 14 Watchlists smoke tests. Existing WXT/Rollup font, chunk-size, duplicate-import, and circular chunk warnings remain noisy but did not block the route smoke.

## Verification Snapshot - 2026-05-21

Branch: `codex/watchlists-final-verification`
Base: `origin/dev` at `668ee4929dd2b27a786a1ca519cd22ed936486e4`

Passed gates:

- `bun run test:watchlists:typecheck` from `apps/packages/ui`: 1 file, 3 tests passed.
- `bun run test:watchlists:scale` from `apps/packages/ui`: 7 files, 53 tests passed.
- `bun run test:watchlists:a11y` from `apps/packages/ui`: 12 files, 91 tests passed. Expected mocked error-state stderr appeared in load-error/remediation tests.
- `python -m pytest tldw_Server_API/tests/Watchlists -q`: 498 passed, 9 skipped, 1 xpassed, 147 warnings. The skipped tests are environment-gated integration/E2E cases in the Watchlists suite.
- `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/watchlists.py tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py tldw_Server_API/app/core/Watchlists -f json -o /tmp/bandit_watchlists_remediation_final.json`: `results: []`, `errors: []`.
- WebUI Playwright smoke: `playwright test e2e/workflows/watchlists-demo-readiness.spec.ts --reporter=line` from `apps/tldw-frontend`: 3 passed. The first sandboxed attempt failed to bind `0.0.0.0:8080`; the escalated rerun passed.
- Extension Playwright smoke: `PLAYWRIGHT_JSON_OUTPUT_NAME=.watchlists-e2e-report.json TLDW_E2E_EXTENSION_HEADLESS=0 playwright test tests/e2e/watchlists.spec.ts --reporter=json` from `apps/extension`: 14 passed. `node scripts/assert-playwright-no-skips.mjs .watchlists-e2e-report.json` reported `passed=14 skipped=0 unexpected=0 flaky=0`. A headless CI-mode attempt skipped all 14 tests because Chromium could not keep the MV3 extension context alive in this environment; the escalated headed rerun is the valid result.

Manual live demo dry run:

- Not completed in this verification pass. It still requires the actual demo API environment, reachable real source URLs, configured LLM/TTS providers, and chosen voices.
- The automated WebUI and extension gates verify the route, guided setup, monitor payloads, output error recovery, truthful audio status display, and Reports/Activity surfaces with controlled API responses.
- Before a public demo claims final playable audio, rerun the Provider And Voice Preflight and Final playback gate above against the actual demo environment.
