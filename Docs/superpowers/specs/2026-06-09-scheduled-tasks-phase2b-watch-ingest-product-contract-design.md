# Scheduled Tasks Phase 2B Watch/Ingest Product Contract Design

Date: 2026-06-09
Status: Ready for review
Owner: Codex brainstorming session
Backlog: TASK-2324

Related:

- `Docs/superpowers/specs/2026-06-01-scheduled-tasks-automation-workbench-prd-design.md`
- `Docs/superpowers/specs/2026-06-08-scheduled-tasks-automation-workbench-phase2-creation-design.md`
- `backlog/tasks/task-2324 - Design-Scheduled-Tasks-Phase-2B-Watch-Ingest-product-contract.md`

## Summary

Phase 2B defines the product and UX contract required before the `/scheduled-tasks` Watch for new items and Ingest new content templates can move from handoff-only to available.

The core decision is that Watch and Ingest are **source-agnostic user intents**, not vendor-specific task types. GitHub issues, YouTube channels, RSS feeds, websites, forums, package registries, vendor advisories, publications, and future source families are examples inside these intents. The UI should not imply that GitHub or YouTube are the primary product model.

`/scheduled-tasks` should become a safe front door and shared lifecycle surface. Watchlists remains the deep workspace for source collections, monitors, scraping, ingest tuning, filters, outputs, reports, digests, and detailed activity. This spec must not remove, limit, or simplify existing Watchlists functionality.

This document stays at the product and UX layer. Backend work is listed only as product-facing contract dependencies.

## Product Decision

Move Watch and Ingest templates to available only after the product can support a complete traceable loop:

1. User chooses a source-agnostic intent.
2. User enters a visible, editable source.
3. System reports whether the capability is ready.
4. System previews what will be checked or ingested.
5. System detects likely duplicates before creation.
6. User reviews schedule, result destination, and domain ownership.
7. System creates the Watchlists-backed automation and returns exact task/domain IDs.
8. User can open the created task detail from `/scheduled-tasks`.
9. User can inspect runs, outputs, failures, and deeper settings through exact links.

If any part of this loop is missing, the template should remain Handoff only, Needs setup, or Unavailable instead of pretending creation is supported.

## Scope

In scope:

- Product contract for making Watch for new items actionable from `/scheduled-tasks`.
- Product contract for making Ingest new content actionable from `/scheduled-tasks`.
- First-time and power-user flows for Watch/Ingest.
- Runtime capability, source preview, duplicate detection, creation response, deep-link, failure, and result-destination requirements.
- UX copy and state guidance for available, degraded, blocked, duplicate, running, success, and failure states.
- Browser extension expectations when the extension opens the same Create route.

Out of scope:

- Backend schema design, migrations, queue implementation, handler registration, or storage decisions.
- Replacing Watchlists with a simplified Scheduled Tasks editor.
- Vendor-specific GitHub or YouTube templates as the primary IA.
- Recurring RAG and Agent Task availability.
- Home implementation, beyond the metadata contract required for future surfacing.
- Bulk source import, saved views, dry run, run-now, and cross-task Results/Runs tabs unless already supported by a safe domain contract.

## Terms

| Term | Meaning |
| --- | --- |
| Watch | User wants to know when a source has new or changed matching items. The expected output is alerting, review, or awareness. |
| Ingest | User wants new source content added to library, media, search, or knowledge surfaces. The expected output is indexed/searchable content. |
| Source | A user-visible input such as a URL, feed, site, repository, channel, publication, advisory page, registry, or other supported location. |
| Source family | Optional system classification, such as feed, video channel, repo issues, website, publication, advisory, or unknown. This is not the primary IA. |
| Preview | A non-destructive explanation of recent items or ingest candidates, with limits and confidence stated plainly. |
| Duplicate | An existing task, monitor, source, or ingested item that appears to cover the same source and intent. |
| Domain entity | The Watchlists-owned monitor, job, output, source, or run that backs the scheduled automation. |
| Scheduled task | The normalized row and detail object visible from `/scheduled-tasks`. |

## Source-Agnostic Intent Model

The Create tab should present Watch and Ingest as user goals:

| Intent | User wording | Example sources | Result promise |
| --- | --- | --- | --- |
| Watch for new items | "Tell me when something new appears." | Issues, feeds, advisories, publications, forums, site pages, package releases | New matching items appear as task results, notifications, Watchlists outputs, or Home cards when supported. |
| Ingest new content | "Keep this source searchable." | Channels, feeds, pages, publications, documents, media sources, site sections | New content is processed into media, search, RAG, or other configured destinations when supported. |

Source examples should appear as examples, helper text, or preview classifications. They should not become top-level navigation unless a future source marketplace exists.

## Ownership Boundary

| Concern | `/scheduled-tasks` owns | Watchlists owns |
| --- | --- | --- |
| Intent selection | Primary entry point and copy | Optional domain entry point |
| Lightweight source entry | Visible, editable, sanitized source field | Source collection and curation |
| Capability status | Summary and recovery link | Detailed service/source health |
| Preview summary | Recent items/candidates and limitations | Full preview, scraping, filtering, and source diagnostics |
| Duplicate warning | Existing task/source warning and action routing | Source/job-level duplicate details |
| Creation | Safe front-door create only after contracts exist | Domain creation, validation, and deep config |
| Schedule summary | Cadence, timezone, next-run preview | Advanced scheduling where applicable |
| Results summary | Latest counts, status, and deep links | Outputs, reports, digests, item detail, activity |
| Pause/resume | Only when safe support is reported | Full job controls |
| Editing | Name, schedule, notification basics only when supported | Source rules, ingest behavior, filters, outputs, reports |

Rule: if a setting changes how a source is collected, scraped, filtered, transformed, reported, or delivered in Watchlists, the deep editor stays in Watchlists. `/scheduled-tasks` can link to that setting but should not rebuild it.

## Availability Gates

Watch and Ingest can be marked Available only when all gates pass for the current environment and current user:

| Gate | Required behavior | If missing |
| --- | --- | --- |
| Capability health | UI can ask whether Watch/Ingest creation is supported, disabled, degraded, or blocked. | Show Needs setup or Handoff only with recovery link. |
| Source preview | UI can validate source input and show non-destructive recent items or ingest candidates. | Keep handoff-only or create with an explicit "Preview unavailable" warning only if product accepts that risk. |
| Duplicate detection | UI can warn about same-intent and likely-same-source tasks before create. | Keep handoff-only for recurring automations that may spam users or waste ingest work. |
| Created entity response | Create returns normalized scheduled task ID, domain entity ID, and exact links. | Do not show "created" from `/scheduled-tasks`. |
| Task visibility | Created automation appears in `/scheduled-tasks` Tasks without requiring a page reload workaround. | Open Watchlists only and explain that central visibility is not available yet. |
| Run/result links | Task detail can link to latest run, outputs, and domain activity when available. | Mark domain visibility incomplete and keep deep inspection in Watchlists. |
| Failure contract | Failures have a stable reason, failed step, and recovery action. | Do not create from `/scheduled-tasks`; hand off to Watchlists. |
| Result destination | Review and success states can say where results will appear. | Do not promise Home or task results. |
| Safe source handling | Source values are visible, editable, and sanitized; secrets are not retained. | Drop unsafe prefill and require manual setup in Watchlists. |
| Watchlists preservation | Existing Watchlists routes, controls, and workflows continue unchanged. | Block Phase 2B release. |

## Product-Facing Contracts

These are contracts the UX needs. They are not implementation prescriptions.

| Contract | Minimum fields or behavior | UX reason |
| --- | --- | --- |
| Capability health | `template`, `state`, `reason`, `setup_url`, `supported_source_families`, `limits`, `can_preview`, `can_create`, `can_pause_resume` | Prevents false availability and tells users what to fix. |
| Source preview | Display source, normalized source key, detected source family, auth requirement, preview items/candidates, preview timestamp, known limitations | Lets users confirm they pasted the right source before scheduling. |
| Duplicate detection | Exact duplicates, likely duplicates, matching reason, existing task URL, existing Watchlists URL, last run, status, allowed actions | Prevents duplicate alerts, duplicate ingests, and confusion about existing automations. |
| Create response | Scheduled task ID, domain entity ID, task detail URL, Watchlists manage URL, next run, initial status, result destinations | Enables exact success navigation and trust that the task exists. |
| Run/result link contract | Latest run URL, output URL, item/result URLs, count summaries, failure URL, domain activity URL | Makes `/scheduled-tasks` a monitoring surface without cloning Watchlists. |
| Failure reason | Stable code, plain-language message, failed step, recoverability, suggested action, recovery URL, redacted diagnostic detail | Supports recovery without exposing raw logs or secrets. |
| Result destination metadata | Whether results appear in task detail, Watchlists outputs, media/search, notifications, Home, or another domain | Allows review and success copy to be accurate. |
| Return path | Watchlists can return users to exact `/scheduled-tasks` task detail after domain setup when a normalized task exists | Keeps the cross-surface workflow coherent. |

## First-Time User Flow: Watch For New Items

Example user goal: "Tell me when a source has new relevant items." A GitHub issues monitor is one possible example, not the template name.

1. User opens `/scheduled-tasks?tab=create` and chooses Watch for new items.
2. The template explains: "Use this when you want alerts or review items when a source changes."
3. User enters a source in a visible field. Placeholder: "Paste a source URL, feed, repo, site, advisory, or publication."
4. System checks capability health and source preview.
5. Preview shows recent items, detected source family, and limitations. If the source resembles issues, preview may show author type and bot/system hints where available.
6. Duplicate check runs before scheduling and shows exact or likely existing watches.
7. User chooses schedule, notification behavior, and basic matching preset when supported.
8. Review screen answers:
   - what source will be checked;
   - what counts as a new item;
   - when it runs;
   - where results appear;
   - what Watchlists still owns.
9. User creates the automation.
10. Success opens the created task detail in `/scheduled-tasks`, with secondary actions to open Watchlists or create another.

Failure fallback:

- If source preview or creation is not supported, show "Setup continues in Watchlists. No scheduled task has been created yet."
- Preserve only safe user-visible source text in a copyable setup summary.
- Link to Watchlists setup with the closest available route.

## First-Time User Flow: Ingest New Content

Example user goal: "Keep this source searchable." A YouTube channel ingest is one possible example, not the template name.

1. User chooses Ingest new content.
2. The template explains: "Use this when you want new source content added to your library and search surfaces."
3. User enters a source. Placeholder: "Paste a channel, feed, site, publication, document source, or media source."
4. System resolves the source and previews recent ingest candidates.
5. Preview labels candidates as new, already ingested, unsupported, needs auth, or preview unavailable when known.
6. User selects basic destination expectations only if the current product can honor them: media library, search index, RAG/knowledge, or Watchlists output.
7. Duplicate detection warns about existing ingest tasks and already-ingested items.
8. User chooses schedule and reviews processing expectations such as transcript/download/indexing limitations when known.
9. User creates the automation.
10. Success opens the task detail, showing next run, Watchlists manage link, and where ingested content will be visible.

Failure fallback:

- If the source cannot be resolved, keep the input visible and explain whether the issue is invalid source, auth, unsupported family, rate limit, or preview timeout.
- If ingest is possible only through Watchlists, use the handoff flow and avoid "scheduled" language.

## Power-User Flow

Power users need direct, dense, reversible paths:

| Need | UX requirement |
| --- | --- |
| Jump to template | `/scheduled-tasks?tab=create&template=watch` and `template=ingest` open the selected template. |
| Paste and inspect quickly | Source field receives focus; Enter or primary action starts preview when safe. |
| Avoid duplicates | Exact duplicate warning offers "Open existing task" as primary. Likely duplicate offers "Review existing" and, if policy allows, "Create anyway". |
| Create repeatedly | Success offers "Create another" without losing template context. |
| Inspect output | Created task detail shows latest run, latest result/output, and Watchlists deep links. |
| Edit deep settings | "Open in Watchlists" routes to the specific monitor/job/source, not only the Watchlists landing page. |
| Operate many tasks | Task table filters can distinguish Watch, Ingest, Managed in Watchlists, Needs attention, Paused, Running, and Waiting. |
| Debug quickly | Failure summary shows failed step, last successful run, retry/recovery action, and domain activity link. |

Do not add bulk management in Phase 2B unless existing safe actions and state refresh semantics are available. A slow but exact flow is better than fast bulk actions that can desynchronize Watchlists.

## Handoff And Return Behavior

Handoff remains required whenever the contract cannot create safely.

Handoff panel requirements:

- State the owner: "Setup continues in Watchlists."
- State the side effect: "No scheduled task has been created yet."
- Show the safe setup summary in editable text.
- Explain why handoff is needed: missing capability, auth setup, source family unsupported, preview unavailable, or Watchlists owns deep configuration.
- Link to the closest Watchlists destination.
- Offer "Copy setup summary" when deep prefill is not supported.

Return behavior after Watchlists setup:

- If a normalized scheduled task exists, return to `/scheduled-tasks?tab=tasks&task_id=<id>`.
- If only a Watchlists entity exists, return to the exact Watchlists monitor/job/run/output.
- If no central task visibility exists yet, the return copy should say: "This automation is managed in Watchlists. It will appear in Scheduled Tasks after central visibility is supported."

## State Model

| State | User-facing label | Required message |
| --- | --- | --- |
| Available | Available | "Create this scheduled task here." |
| Needs setup | Setup required | Explain missing auth, connector, provider, or configuration with direct recovery. |
| Handoff only | Continue in Watchlists | "No scheduled task has been created yet." |
| Managed in Watchlists | Managed in Watchlists | "Scheduled Tasks shows status and links. Watchlists owns source and output settings." |
| Preview unavailable | Preview unavailable | Explain whether creation is blocked or allowed with limited confidence. |
| Duplicate exact | Already exists | Primary action opens existing task. |
| Duplicate likely | Similar automation found | User reviews existing task before creating another. |
| Waiting | Waiting for next run | Show next run and timezone. |
| Running | Running now | Show current step, elapsed time, and last update if available. |
| Found results | Found results | Show count and exact result/output link. |
| Completed no results | No new results | Show last checked time. |
| Needs attention | Needs attention | Show failed step, reason, and recovery action. |
| Paused | Paused | Clarify user-controlled pause. |
| Blocked | Blocked | Clarify dependency-controlled block. |

## UX Copy Recommendations

Template labels:

| Context | Copy |
| --- | --- |
| Watch title | Watch for new items |
| Watch description | Get notified when a source has new matching items. |
| Ingest title | Ingest new content |
| Ingest description | Add new source content to your library and search surfaces. |
| Source field label | Source |
| Source helper | Paste a source URL, feed, site, advisory, publication, channel, or other supported source. |
| Preview CTA | Preview source |
| Watch create CTA | Create watch |
| Ingest create CTA | Create ingest |
| Domain CTA | Open in Watchlists |
| Duplicate primary | Open existing task |
| Duplicate secondary | Review similar automation |
| Handoff primary | Continue in Watchlists |
| Handoff secondary | Copy setup summary |

Review copy:

- "Scheduled Tasks will show status, next run, and recent results. Watchlists owns source rules, ingest behavior, outputs, and reports."
- "Preview shows recent items only. Future runs may find different items."
- "Results destination: Scheduled Tasks detail, Watchlists outputs, and Home when automation results are supported there."
- "No scheduled task has been created yet."

Success copy:

- "Watch scheduled. Opening task detail."
- "Ingest scheduled. Opening task detail."
- "This automation is also managed in Watchlists."
- "Next run: [date/time/timezone]."

Error copy:

| Condition | Message |
| --- | --- |
| Invalid source | "This source could not be recognized. Check the URL or continue setup in Watchlists." |
| Auth required | "This source needs authorization before it can be checked." |
| Rate limited | "The source could not be previewed because the provider is rate limited. Try again later or continue in Watchlists." |
| Duplicate exact | "This source is already being watched or ingested by an existing task." |
| Create failed | "The automation was not created. Your inputs are still here." |
| Partial visibility | "Created in Watchlists, but central Scheduled Tasks visibility is not available yet." |

Avoid:

- "GitHub monitor" as a primary template label.
- "YouTube ingest" as a primary template label.
- "Task created" after handoff only.
- "External" without naming the owner.
- Raw implementation terms such as cron, APScheduler, job backend, handler, queue, or worker.

## Home Surfacing Model Dependency

Phase 2B does not implement Home, but Watch/Ingest creation must return result-destination metadata so future Home cards can be trustworthy.

Home should eventually show:

| Card type | Trigger | Primary action |
| --- | --- | --- |
| New watched items | Watch run finds reviewable items | Open exact task result or Watchlists output |
| New ingested content | Ingest run adds searchable items | Open ingested media/search result |
| Needs attention | Watch/Ingest run fails or is blocked | Open task failure detail |
| Running now | Long-running ingest is active | Open run detail |

Home should dedupe notifications and cards for the same run/result. A user should not have to dismiss the same automation event in multiple places.

## Browser Extension Expectations

The extension may open the same Create route with current-page context, but it must follow the same safety rules:

- Prefill only visible, editable source text.
- Do not silently preserve fragments, credential-like query parameters, invite codes, session tokens, or auth values.
- Do not infer schedule, filters, notification policy, destination, or credentials from page context.
- Show detected source family as a hint, not a decision.
- If the current page is not recognized, leave the generic Source field empty.
- In constrained widths, keep source, preview, duplicate warning, schedule, and review steps readable without horizontal scrolling.

Extension entry copy:

- "Create scheduled task from this page" when a safe source is available.
- "Review source before scheduling" on the first step.
- "Some setup may continue in Watchlists."

## Accessibility And Usability Requirements

- Template cards, tabs, wizard steps, duplicate warnings, and result links must be keyboard operable.
- Status must use text plus icon, never color alone.
- Preview loading and running updates should use polite live regions.
- Focus should move to the first invalid field on validation failure.
- After create success, focus should move to the opened task detail heading.
- Duplicate warnings should be announced as status or alert depending on severity.
- Source preview tables need column headers and responsive stacked rows at extension widths.
- Date/time previews must include timezone and be announced with validation messages.
- Icon-only actions need accessible names and tooltips.
- Copyable setup summaries need visible confirmation when copied.
- Reduced-motion preferences should be respected for progress and transition states.

## Risks And Mitigations

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Watchlists is accidentally reduced to a modal | Existing Watchlists users lose deep workflows | Keep source rules, ingest behavior, outputs, reports, and activity in Watchlists. |
| GitHub/YouTube examples distort IA | Users with other sources think they are unsupported | Use source-agnostic template names and broad source examples. |
| Creation succeeds but task is hard to find | Users lose trust and cannot inspect results | Require scheduled task ID and exact detail URL in create response. |
| Duplicate watches create alert noise | Users get spammed and may disable automation | Require duplicate detection before making templates available. |
| Duplicate ingests waste resources | Processing and storage costs increase | Preview candidate status and explain duplicate/skipped policy. |
| Preview feels authoritative when it is partial | Users overtrust incomplete source checks | Label preview timestamp, limits, and unsupported fields. |
| Handoff creates hidden side effects | Users cannot tell whether anything was scheduled | Handoff copy must say no task was created. |
| Unsafe source prefill leaks secrets | URLs can contain tokens or private fragments | Sanitize and omit unsafe values; keep user confirmation visible. |
| `/scheduled-tasks` and Watchlists states diverge | Users see conflicting status | Use domain links and explicit "Managed in Watchlists" ownership copy. |
| Backend failures surface as raw errors | Users cannot recover | Stable failure reasons and recovery links are required. |

## Acceptance Checklist

Before Phase 2B implementation starts:

- Watch/Ingest availability gates are accepted by product/design.
- Watchlists ownership boundary is accepted by maintainers.
- Source examples are source-agnostic and do not privilege GitHub/YouTube.
- Handoff copy remains available for unsupported or degraded environments.
- Success states require exact task and domain links.
- Duplicate policy is agreed for exact and likely duplicates.
- Result destination language is accurate for current and future surfaces.
- Browser extension source-prefill safety rules are accepted.

Before Watch/Ingest templates move to Available:

- Capability health, preview, duplicate, create, and failure contracts exist.
- Created automations appear in `/scheduled-tasks` Tasks.
- Created automation success opens exact task detail.
- Task detail links to exact Watchlists monitor/job/run/output when available.
- Existing Watchlists workflows are unchanged.
- Handoff-only fallback still works when capability is unavailable.
- Accessibility requirements are covered in interaction tests or manual QA.

## Open Product Questions

| Question | Why it matters |
| --- | --- |
| Which Watchlists setup routes can safely accept prefilled source text? | Determines whether handoff is a deep link or copy-summary flow. |
| Which source families can provide reliable preview in the first available slice? | Determines whether availability is global or family-specific. |
| Should "Create anyway" be allowed for likely duplicates? | Affects power-user flexibility and duplicate-noise risk. |
| Which destinations can Ingest honestly promise at create time? | Prevents overpromising searchable or RAG-ready results. |
| Should pause/resume be exposed from `/scheduled-tasks` for Watchlists-backed tasks? | Requires safe state synchronization and clear ownership copy. |
| What is the minimum Home metadata contract for future result cards? | Avoids creating automations that cannot later surface useful outcomes. |

## Recommended Delivery Slices

| Slice | Outcome | Backend dependency posture |
| --- | --- | --- |
| 2B.1 Product contract | This spec accepted and linked from Backlog | None |
| 2B.2 Capability-aware frontend shell | Watch/Ingest can display runtime Available/Needs setup/Handoff states when contracts exist | Depends on capability contract only |
| 2B.3 Watchlists creation handoff adapter | Source preview, duplicate warnings, and safe create for the first supported source families | Depends on Watchlists contracts |
| 2B.4 Task detail/result links | Created Watch/Ingest tasks open exact details and domain links | Depends on created entity and run/result link contracts |
| 2B.5 Home surfacing follow-up | Home can show latest Watch/Ingest results and failures | Depends on normalized result metadata |
