# Targeted single-user repair verification

Product f83a8ba456; documentation checkpoint bc15c22657. Fresh configuration/database with existing installed dependencies; API18400 PID16030 started before visible provider setup and has not restarted. WebUI18480 PID23418. Browser cycle4-targeted-single-20260916. This is targeted verification, not another full fresh matrix.

## Running outcomes

- Local setup path/privacy acknowledgement, blank-model discovery, exact real Gemma selection/validation/save: passed. Conservative ingest defaults retained; optional audio, RAG/storage and MCP deferred.
- Real first Chat153: HTTP200, Hello!; exact provider/model corroborated in response.
- Setup complete156: HTTP200/success, requires_restart=true in response; no restart performed.
- UAT106 FAILED: client handoff remains First chat with missing/ambiguous catalog target error and visible Finish setup. No model catalog network request occurred. Browser API key is still absent, as expected before the manual Settings connection step.
- Source trace: TldwModels credential guard returns an empty cache without fetching; handoff incorrectly requires a catalog match after the authoritative same-target first-chat response. Existing TASK13260.47 reopened for correction; do not bypass catalog authentication.
- Console:0errors,2missing-browser-key warnings before key configuration; Chromium verbose password-field-outside-form message. No Next overlay.
- Ordinary Chat/model refresh107, fresh canonical reload103/backlink111, ingestion104/105, analysis115/110 and Review060 are pending behind106.

Evidence includes the observed UI, request metadata, actual responses and visually inspected failure screenshot. Runtime secrets remain private and are not retained.

## Repair and attempted mounted recovery

Wizard correction committed8097d672d5 after independent85permanent+2boundary probes, author114cases/6files, ESLint0/0 and90unchanged compiler diagnostics. Existing page survived development hot reload, but its effect cleanup intentionally advanced the captured handoff generation. Finish setup then showed the changed-connection guard; no inference/completion call repeated. Readiness now returns403 after already-completed setup without browser credentials. This HMR-spanning retry is not fresh-run acceptance. Preserve this profile and repeat from a new fresh targeted2 profile on18402/18482. No API restart was performed in the originalprofile.
