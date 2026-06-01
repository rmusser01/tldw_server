# Watchlists

Watchlists manages scheduled monitoring of feeds, sites, and scraped content:
sources, filters, alert rules, WebSub, runs, outputs, reports, templates, audio
briefings, and notification delivery. It is user-visible orchestration that
touches Web Scraping, Collections, Jobs/Scheduler, Notifications, TTS, and
Security egress policy.

## Start Here

- Pipeline: `pipeline.py`, `fetchers.py`, and `filters.py`.
- Templates and reports: `template_store.py`, `template_composer_ast.py`,
  `template_composer_roundtrip.py`, and `report_evidence.py`.
- Alerts and outputs: `alert_rules.py`, `content_alerts.py`,
  `output_enrichment_handler.py`, and `audio_artifact_projection.py`.
- Interop: `opml.py`, `websub.py`, and `audio_briefing_workflow.py`.
- API endpoints and schemas: `app/api/v1/endpoints/watchlists.py`,
  `app/api/v1/endpoints/watchlist_alert_rules.py`, and
  `app/api/v1/schemas/watchlists_schemas.py`.
- Tests: `tests/Watchlists/`.

## Responsibilities

- Fetch and normalize feed/site items under configured egress and selector rules.
- Evaluate filters and include/exclude rules for source items.
- Run scheduled or on-demand jobs and persist run/output metadata.
- Render Markdown/HTML templates and evidence-backed reports.
- Deliver outputs through Notifications and optional audio briefing paths.
- Import/export OPML and support WebSub push item parsing.

## Module Map

- `pipeline.py` orchestrates fetch, filter, output, and run status behavior.
- `fetchers.py` handles RSS/site extraction and selector validation.
- `filters.py` evaluates matching/filter rules.
- `template_store.py` validates stored templates.
- `content_alerts.py` and `alert_rules.py` support alerting workflows.
- `watchlists_telemetry_metrics.py` emits telemetry/metrics for operations.
- `websub.py` and `opml.py` handle external feed formats.

## How It Connects

- `Security.egress` and Web Scraping helpers guard outbound requests.
- Collections stores outputs and reading/list-style artifacts.
- Notifications delivers email/Chatbook outputs.
- TTS and audio modules can create briefings from Watchlist output.
- Jobs/Scheduler service periodic work and admin controls for runs.

## Extension Points

- Add fetchers in `fetchers.py` with explicit egress checks and selector tests.
- Extend filter behavior in `filters.py` and update preview endpoint tests.
- Add output formats by composing template/report helpers and Notification
  delivery rather than embedding side effects in the pipeline.

## Testing

- Pipeline and scheduler: `tests/Watchlists/test_watchlists_pipeline.py`,
  `tests/Watchlists/test_full_pipeline_integration.py`, and
  `tests/Watchlists/test_watchlists_scheduler_integration.py`.
- Preview/filter behavior: `tests/Watchlists/test_preview_endpoint.py`,
  `tests/Watchlists/test_filters_api.py`, and
  `tests/Watchlists/test_filters_matching.py`.
- OPML/WebSub/templates: `tests/Watchlists/test_opml_api.py`,
  `tests/Watchlists/test_websub.py`, and
  `tests/Watchlists/test_watchlists_template_store.py`.
- Delivery/reports/audio: `tests/Watchlists/test_delivery_integrations.py`,
  `tests/Watchlists/test_watchlist_reports_api.py`, and
  `tests/Watchlists/test_audio_briefing_workflow.py`.

## Gotchas

- Network fetches must honor egress policy and should be deterministic in tests
  through fakes or fixture responses.
- Scheduled runs can be expensive. Keep preview paths cheap and avoid mutating
  run state during validation-only flows.
