# First-Class Watchlists Design

Status: Approved design draft
Task: TASK-349
Date: 2026-05-15
Scope: `/watchlists` WebUI and directly required API/data workflows

## Summary

Make Watchlist a first-class project-like container for recurring research and monitoring workflows. The current Watchlists module has mature child concepts, including sources, groups/tags, jobs, runs, scraped items, outputs, templates, OPML import/export, claim-cluster subscriptions, run-health notifications, and topic monitoring dependencies. It does not yet expose a clear user-facing Watchlist object that explains why those child records belong together.

The new model should make a Watchlist the user's durable research object: it owns intent, tracked scope, sources, monitors, alert rules, triage state, reports, evidence, and lifecycle. Existing Sources, Monitors, Activity, Items, Reports, Templates, Settings, and related APIs should be preserved and scoped under this container rather than replaced wholesale.

## Evidence Basis

This design is grounded in the current repository state and prior browser-observed `/watchlists` audit evidence:

- Route and shell: `apps/packages/ui/src/routes/option-watchlists.tsx`
- Current page component: `apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx`
- Frontend types and services: `apps/packages/ui/src/types/watchlists.ts`, `apps/packages/ui/src/services/watchlists.ts`
- Current backend router and schemas: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`, `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
- Persistence: `tldw_Server_API/app/core/DB_Management/Watchlists_DB.py`
- Pipeline: `tldw_Server_API/app/core/Watchlists/pipeline.py`
- Run-stat alert rules: `tldw_Server_API/app/api/v1/endpoints/watchlist_alert_rules.py`, `tldw_Server_API/app/core/Watchlists/alert_rules.py`
- Topic monitoring dependency: `tldw_Server_API/app/api/v1/endpoints/monitoring.py`, `tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py`, `tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py`
- Product/API docs: `Docs/Product/Watchlists/Watchlist_PRD.md`, `Docs/API-related/Watchlists_API.md`, `Docs/Product/Watchlists_IA_Reduced_Navigation_Rollout_Gates_2026_02_24.md`

Observed current UI concepts include Feeds/Sources, Monitors/Jobs, Activity/Runs, Articles/Items, Reports/Outputs, Templates, Settings, guided quick setup, item triage, report generation, and claim-cluster subscriptions. The UI already uses friendlier labels than the API, but the product center remains the pipeline objects rather than a first-class Watchlist.

## Primary Personas

### CTI and OSINT Researcher

Tracks threats, vulnerabilities, CVEs, IOCs, malware families, actors, advisories, campaigns, sectors, regions, and source updates. Needs situational awareness, meaningful alerts, evidence trail, confidence calibration, and defensible reports.

Key jobs:
- Set up a watchlist around a threat, vulnerability, actor, sector, geography, or advisory stream.
- Receive alerts when new collected items match specific descriptors or classifications.
- Review new items by severity, novelty, confidence, source, and evidence value.
- Generate reports that preserve provenance, citations, included/excluded evidence, and uncertainty.

### News-Focused Power User

Tracks developing events, topics, people, organizations, places, sources, and source diversity. Wants synthesized briefings, recency awareness, easy follow-up, and clear separation between actual updates and pipeline failures.

Key jobs:
- Create a topic or event watchlist with source sets and recurring checks.
- Identify what changed since the last review.
- Compare source coverage and avoid repetitive duplicates.
- Generate personal briefings with links and follow-up paths.

## Product Model

Approved direction: Watchlist is a project-like container, not a single monitor/job.

A Watchlist owns:

- Intent: name, description, domain, objective, priority, owner, status.
- Tracked scope: topics, entities, keywords, CVEs, IOCs, people, organizations, events, source groups.
- Collection: sources and one or more monitors.
- Detection: content-match alert rules for user-defined descriptors and classifications.
- Review: item triage, novelty, confidence, severity, source diversity, evidence status.
- Outputs: reports and briefings with provenance and generation history.
- Lifecycle: active, paused, archived, deleted/restorable.

Minimum Watchlist fields:

- `id`
- `name`
- `description`
- `objective`
- `domain`: `cti_osint`, `news`, `general`
- `status`: `active`, `paused`, `archived`
- `priority`: `low`, `medium`, `high`, `critical`
- `tags`
- `created_at`, `updated_at`, `archived_at`
- owner/sharing fields where the current auth model supports them

Existing unscoped data should migrate into a default Watchlist:

- Name: `Imported Watchlist`
- Objective: `Existing feeds, monitors, items, and reports migrated from the previous Watchlists workspace.`
- Domain: `general`
- Status: `active`

## Terminology

Keep Watchlists as the page/module name and make Watchlist the first-class object.

Preferred labels:

- Watchlist: project-like tracking container.
- Feed or Source: input target.
- Monitor: scheduled collection pipeline, currently backed by jobs.
- Activity: run history and health.
- Item or Update: collected result. Prefer Item or Update over Article in scoped CTI contexts.
- Alert: content-match notification.
- Health issue: pipeline/system problem such as run failure, zero items, source failure, or high error rate.
- Report: generated research artifact.
- Template: report format definition.

Important boundary: do not use unqualified "alert" for run failures in the Watchlist UX. Label those as health issues or pipeline health alerts.

## Information Architecture

The `/watchlists` page should pivot from object tabs first to Watchlist first.

Top-level `/watchlists`:

- Watchlists overview/list.
- Create Watchlist.
- Global health summary.
- Recently updated Watchlists.
- Unread content alerts.
- Watchlists needing health attention.

Selected Watchlist view:

- Overview: intent, tracked scope, collection status, unread alerts, recent changes, report readiness.
- Tracking: descriptors, entities, classifications, exclusions.
- Sources: feeds/sites/forums/OPML, source health, tags/groups.
- Monitors: schedules, scope, filters, run now, pause/resume.
- Alerts: content-match rules and alert inbox.
- Items: review queue and report queue.
- Reports: generated outputs, report builder, templates.
- Activity/Evidence: runs, logs, provenance, included/excluded evidence.
- Settings: lifecycle, defaults, sharing, archive/delete/restore.

Existing child tabs can be reused, but they must be scoped by selected Watchlist and should not force the user to infer context from source/job IDs.

## Core Workflow

### 1. Create Watchlist

Users create from a preset or blank state:

- Threat or vulnerability tracking.
- Actor, malware, or campaign tracking.
- Advisory or source monitoring.
- Breaking news event.
- Topic, person, or organization tracking.
- Blank research watchlist.

Required setup fields:

- Name.
- Objective: what the user is trying to learn, detect, or report.
- Domain: CTI/OSINT, news, general.
- Priority.
- Default report style.
- Optional tags/status.

### 2. Define Tracked Scope

The Watchlist should expose structured tracking:

- Plain-language descriptor.
- Entities.
- Keywords/literals/regex.
- Classifications.
- Source constraints.
- Exclusions.

CTI/OSINT fields:

- CVE.
- IOC.
- Threat actor.
- Malware family.
- Campaign.
- Vendor/product.
- Advisory source.
- Sector.
- Geography.
- Severity.
- Confidence.

News fields:

- Event.
- Person.
- Organization.
- Place.
- Beat/topic.
- Source diversity.
- Recency.

### 3. Attach Sources

Sources are associated with a Watchlist:

- RSS feeds.
- Websites.
- Forums where enabled.
- OPML import/export.
- Source groups and tags.
- Health/dedupe state.

Global source reuse can exist, but user-facing source creation should answer: which Watchlist is this source for?

### 4. Configure Monitors

A Watchlist can contain multiple Monitors. Monitor remains the execution unit.

Monitors define:

- Schedule.
- Source scope.
- Job filters.
- Collection settings.
- Output defaults.
- Run now.
- Pause/resume.

### 5. Review Updates

The review queue should prioritize:

- New matches.
- High-severity or high-confidence items.
- Novel items.
- Items from diverse or corroborating sources.
- Unreviewed alerts.
- Report candidates.

### 6. Alert On Content Matches

Content alerts are user-defined match notifications:

- New item mentions a specific CVE.
- New ransomware report affects a specified sector/geography.
- New article links a person to an organization.
- New advisory appears from a specified source.
- Multiple sources report the same event or claim.

Pipeline failures remain visible as health issues.

### 7. Generate Reports

Reports should be defensible artifacts, not only lightweight generated text.

Report metadata should include:

- Watchlist ID.
- Monitor/job ID.
- Run ID or run IDs.
- Item IDs.
- Source IDs and URLs.
- Template/version.
- Generation parameters.
- Evidence inclusion state.
- Created/exported timestamps.

Report content should support:

- Scope and time window.
- Summary or assessment.
- Evidence table.
- Citations/source URLs.
- Confidence and uncertainty notes.
- Included/excluded item trail.
- Export to Markdown/HTML/Chatbook/audio where already supported.

### 8. Manage Lifecycle

Watchlists need clear controls:

- Active.
- Paused.
- Archived.
- Delete with restore window.
- Duplicate.
- Export configuration/report package.

## API and Data Boundary

Add Watchlist-level endpoints while preserving existing child endpoints.

Likely additions:

- `GET /api/v1/watchlists`
- `POST /api/v1/watchlists`
- `GET /api/v1/watchlists/{watchlist_id}`
- `PATCH /api/v1/watchlists/{watchlist_id}`
- `DELETE /api/v1/watchlists/{watchlist_id}`
- `POST /api/v1/watchlists/{watchlist_id}/restore`
- `GET /api/v1/watchlists/{watchlist_id}/overview`
- `GET /api/v1/watchlists/{watchlist_id}/items`
- `GET /api/v1/watchlists/{watchlist_id}/reports`
- `GET /api/v1/watchlists/{watchlist_id}/alerts`

Existing endpoints can first accept `watchlist_id` filters before deeper nested routes are added. This reduces churn and preserves the existing service layer.

Expected child relationships:

- Sources can belong to one or more Watchlists, or start with one Watchlist if many-to-many is too costly for v1.
- Jobs/Monitors belong to a Watchlist.
- Runs inherit Watchlist through their Monitor.
- Items inherit Watchlist through run/job and may also store direct `watchlist_id` for efficient filtering.
- Outputs/Reports carry Watchlist provenance.
- Content alert rules belong to a Watchlist.
- Health rules may be job-scoped but displayed through Watchlist health.

## Alert Boundary

There are currently two relevant alert-like systems:

- `watchlist_alert_rules.py` evaluates completed run statistics such as no items, error rate, items above/below threshold, and run failed.
- Topic Monitoring stores `monitoring_watchlists`, `monitoring_watchlist_rules`, and `topic_alerts`, then scans text for literal or regex patterns.

Product decision:

- User-facing Watchlist Alerts are content-match alerts.
- Run-stat rules become Pipeline Health Rules or Health Issues in the UX.
- Topic Monitoring is a candidate dependency for content-match alert evaluation, but the `/watchlists` page should not expose admin-oriented monitoring concepts directly.

Content alert rule capabilities:

- Plain-language descriptor.
- Exact entity matches.
- Keywords/literals.
- Regex for advanced users.
- Classification constraints.
- Source constraints.
- Severity/priority.
- Dedup/snooze behavior.
- Alert reason and evidence.

Alert review states:

- Unread.
- Read.
- Acknowledged.
- Dismissed.
- Snoozed.

Each content alert should show:

- Watchlist.
- Rule.
- Matched item.
- Matched text/snippet.
- Source URL.
- Timestamp.
- Severity/priority.
- Confidence when available.
- Why it fired.

## Triage Model

The scoped Items/Updates view should support:

- Recency.
- Reviewed/unreviewed.
- Alert match.
- Source.
- Source group/tag.
- Severity/priority.
- Confidence.
- Novelty/change.
- Queued for report.
- Included in report.
- Excluded from report.

Saved views should be per Watchlist.

Useful default views:

- New alerts.
- High priority.
- Needs review.
- Queued for report.
- Recently changed.
- Source diversity.
- Pipeline filtered.

## Report Model

Reports build on current Outputs/Templates, but should be surfaced at Watchlist level.

Report builder flow:

1. Select Watchlist.
2. Choose time window and monitor/run scope.
3. Review candidate evidence.
4. Include/exclude items.
5. Choose report preset/template.
6. Generate preview.
7. Export or save.

Presets:

- CTI assessment.
- Vulnerability watch update.
- Incident/campaign update.
- Advisory digest.
- News briefing.
- Event update.
- Source digest.

Report warnings:

- Weak evidence.
- Single-source claim.
- Stale data.
- Low source diversity.
- No high-confidence matches.
- Collection failures during report window.

## Extension-Sized Management

Full management is in scope for constrained viewports.

Requirements:

- Watchlist switcher must remain reachable.
- Creation/edit flows should be step-based and resumable.
- Tables need list/detail alternatives.
- Source, monitor, alert, item, and report CRUD must remain available.
- Critical copy and controls must not clip or require wide layouts.

## Staged Remediation Plan

### Stage 1: Product Contract and IA Refactor

Goal: introduce Watchlist as the first-class container without breaking existing feeds/jobs/runs/items/output APIs.

Scope:

- Add Watchlist object contract.
- Add parent relationship from child records to Watchlist.
- Migrate existing data into a default Watchlist.
- Update `/watchlists` IA to start with Watchlist selection/overview.
- Keep existing child views.

Dependencies:

- `Watchlists_DB.py` schema path.
- `watchlists_schemas.py` API schema.
- `watchlists.py` router.
- `watchlists.ts` frontend types/service.
- `WatchlistsPlaygroundPage.tsx` and store state.

Tests:

- DB migration/default Watchlist creation.
- API CRUD and restore.
- Existing child endpoint compatibility.
- Frontend route and deep-link behavior.
- Extension viewport smoke test.

Complexity: High.

### Stage 2: Watchlist Creation and Setup Wizard

Goal: replace generic quick setup with Watchlist-first onboarding.

Scope:

- Domain presets: CTI/OSINT, news, general, blank.
- Capture objective and tracked scope before sources.
- Support "start from sources", "start from topic", and "start from report goal".
- Add CTI/news examples in empty states.

Dependencies:

- Stage 1 Watchlist CRUD.
- Current quick setup in `OverviewTab`.
- Current source/job payload builders.

Tests:

- Wizard creates Watchlist, sources, and monitor.
- Wizard works with no sources.
- CTI and news preset copy contract tests.
- Narrow viewport create/edit flow.

Complexity: Medium.

### Stage 3: Content-Match Alerts

Goal: make alerts mean user-defined content matches, separate from pipeline health.

Scope:

- Add Watchlist alert rules for descriptors, classifications, entities, keywords, and source constraints.
- Reuse or adapt Topic Monitoring where appropriate.
- Wrap run-stat alert rules as health issues.
- Add alert inbox and review states.
- Show matched evidence.

Dependencies:

- Topic Monitoring service and DB.
- Existing watchlist run-stat alert rules.
- Notification service.
- Watchlist item persistence.

Tests:

- Content rule creation/update/delete.
- Matching against newly collected items.
- Dedupe behavior.
- Alert evidence display.
- Health issue separation.

Complexity: High.

### Stage 4: Review Queue and Triage Model

Goal: make review efficient for CTI and news.

Scope:

- Rename or contextually scope Articles to Items/Updates.
- Add sorting/filtering by recency, novelty, severity, confidence, source, alert match, reviewed status, and report queue state.
- Add domain metadata display where present.
- Add batch triage and per-Watchlist saved views.

Dependencies:

- Stage 1 scoping.
- Stage 3 content alerts.
- Current `ItemsTab`.
- Existing scraped item schema.

Tests:

- Filter/sort behavior.
- Batch triage.
- Saved views per Watchlist.
- Report queue persistence.

Complexity: Medium to High.

### Stage 5: Defensible Reports

Goal: turn outputs into research artifacts.

Scope:

- Watchlist-scoped report builder.
- Evidence table and source provenance.
- Included/excluded item trail.
- CTI/news report presets.
- Report readiness state.
- Preserve Markdown/HTML/Chatbook/audio paths.

Dependencies:

- Existing outputs/templates APIs.
- Report metadata additions.
- Item review/report queue state.

Tests:

- Report generation with provenance.
- Template/version metadata.
- Export/download.
- Weak-evidence warning states.
- Chatbook/audio compatibility where available.

Complexity: High.

### Stage 6: Extension-Sized Full Management

Goal: make full Watchlist management viable in constrained viewports.

Scope:

- Replace desktop-only tab/table assumptions.
- Add task switcher or drawer navigation.
- Convert dense tables to list/detail patterns.
- Ensure source/monitor/alert/report CRUD works in extension viewport.

Dependencies:

- Current shared WebUI/extension route.
- Existing Ant Design components and layout constraints.

Tests:

- Browser/CDP or Playwright screenshots at extension-sized viewport.
- Keyboard navigation.
- No clipped critical text/controls.
- CRUD smoke flows.

Complexity: Medium.

### Stage 7: Trust, Calibration, and Observability

Goal: improve confidence in automated collection/reporting.

Scope:

- Better empty/loading/error/rate-limit states.
- Source health explanations.
- Last checked and next run state.
- Why item matched.
- Alert dedupe explanations.
- Report evidence-quality warnings.

Dependencies:

- Overview health data.
- Run details/logs.
- Alert reason metadata.
- Report metadata.

Tests:

- No-results state variants.
- Rate-limit and backend error states.
- Source health messaging.
- Report warning generation.

Complexity: Medium.

## Acceptance Criteria

- Watchlist exists as a persisted first-class container.
- Sources, monitors/jobs, runs, items, reports/outputs, and alert rules can be associated with a Watchlist.
- Existing unscoped records are assigned to a default migrated Watchlist.
- Watchlist lifecycle supports active, paused, archived, delete, and restore.
- `/watchlists` opens to a Watchlist-level overview before object-specific tabs.
- Users can create a Watchlist from CTI/OSINT, news, general, or blank presets.
- Inside a selected Watchlist, child views are scoped and labeled clearly.
- The UI distinguishes content alerts from health issues.
- Extension-sized viewport supports full create/edit/manage flows.
- Users can define content-match alerts using descriptors, entities, keywords, classifications, or source constraints.
- Alerts show why they fired.
- Users can mark alerts read, acknowledged, dismissed, or snoozed.
- Users can filter and sort Watchlist items by recency, reviewed state, source, alert match, severity/priority, novelty, and queued-for-report state.
- Reports are scoped to a Watchlist and can include one or more monitor runs.
- Reports include provenance: sources, URLs, item IDs, run IDs, template/version, generated timestamp.
- Reports preserve current export paths.
- Report builder warns when evidence is weak, stale, or too narrow.
- Empty states explain whether setup is missing, no matches were found, or collection failed.
- Loading and rate-limit errors are visible and recoverable.
- Source health and next-run state are visible at Watchlist level.

## Rollout Gates

- Existing source/job/run/item/output flows continue to work under the default migrated Watchlist.
- No route regression for `/watchlists?tab=...` aliases without a migration path.
- Watchlist CRUD and child scoping are covered before the UI hides global object views.
- Content alerts are not shipped in UI until alert reason/evidence is available.
- Report builder does not call outputs "defensible" until provenance is present.
- Extension viewport passes full-management smoke flows.

## Risks

- Naming collision with the existing Topic Monitoring `Watchlist` model.
- Current `/api/v1/watchlists/{watchlist_id}/clusters` path actually treats `watchlist_id` as a job ID. First-class Watchlist IDs will need route compatibility care.
- Many-to-many source reuse may be more complex than one Watchlist parent per source.
- Alert matching may require domain enrichment that scraped items do not currently store.
- Report provenance may need metadata normalization across Collections outputs, Media DB persistence, and Chatbook exports.
- The existing route is already dense; adding Watchlist selection without reducing object-tab complexity could make the page worse.

## Open Questions

- Should a source be owned by one Watchlist initially, or reusable across multiple Watchlists?
- Should Watchlist archive pause monitors automatically?
- Should content alerts be evaluated during ingestion only, or also backfilled against existing items when a rule changes?
- Which confidence/novelty fields are available now versus future enrichment?
- Should CTI entity extraction be manual-entry-first, LLM-assisted, or both?
- Should reports support immutable "published" snapshots distinct from editable generated drafts?

## Next Step

After this design spec is accepted, create an implementation plan that splits work into reviewable tasks. The first implementation slice should focus on Stage 1: Watchlist container contract, migration default, API CRUD, and scoped frontend shell.
