# Scheduled Tasks Automation Workbench Phase 2 Creation Design

Date: 2026-06-08
Status: Ready for spec review
Owner: Codex brainstorming session
Backlog: TASK-2320

Related:

- `Docs/superpowers/specs/2026-06-01-scheduled-tasks-automation-workbench-prd-design.md`
- `Docs/superpowers/plans/2026-06-01-scheduled-tasks-automation-workbench-phase1-implementation-plan.md`
- `backlog/tasks/task-498 - Implement-Scheduled-Tasks-Automation-Workbench-Phase-1.md`

## Summary

Phase 2 should make `/scheduled-tasks` feel like a creation workbench without overclaiming capabilities or collapsing domain workspaces.

The approved direction is a two-part product plan:

1. **Phase 2A: Creation framework.** Add an intent-based Create experience, URL-addressable tabs, deterministic template finder, guided wizard model, one fully working Reminder template, and honest capability-aware handoffs for Watchlists, RAG, ACP/Agent Tasks, and future automation families.
2. **Phase 2B: Watch/Ingest product contract.** Define the product and backend dependencies required to make "Watch for new items" and "Ingest new content" templates fully actionable from `/scheduled-tasks` while preserving the existing Watchlists UX and persona fit.

GitHub issue monitoring and YouTube channel ingest are examples of possible sources, not primary product assumptions. The product should organize creation by user intent, not by vendor/source type.

This spec stays mostly at the product and UX layer. Backend work appears only as dependency contracts and acceptance criteria for later implementation.

## Product Decision

Use an **Intent-Based Creation Workbench** model.

`/scheduled-tasks` should help users answer:

- "What can I automate?"
- "Which product area owns the deeper setup?"
- "Can I create this now, or is it planned?"
- "Where will results show up?"
- "How do I inspect or fix this after it runs?"

It should not become the deep editor for every domain. Watchlists remains the full workspace for source collection, source curation, scraping, ingest tuning, filters, reports, digests, outputs, and detailed run activity.

## Current Context

Phase 1 established these baseline capabilities:

- `/scheduled-tasks` lists reminders and Watchlists jobs in a unified table.
- User-facing task status, overview metrics, and task detail inspection exist.
- Reminder creation/editing uses safer schedule controls than raw ISO/cron-first inputs.
- Watchlists jobs are visible as externally managed monitors with deep links to Watchlists.
- Watchlists functionality remains intact and separate.

Remaining creation problems:

- First-time users still need help understanding what scheduled tasks can automate.
- The Create path is not yet a general automation entry point.
- Source-specific examples can mislead users into thinking GitHub or YouTube are the main product model.
- Recurring RAG and ACP/Agent schedules are not yet visibly actionable from this surface.
- Watch/Ingest creation can only be promised if existing Watchlists or other APIs support a safe creation path.
- Users need clearer confirmation of where created tasks, runs, failures, and results will appear.

## Goals

1. Make `/scheduled-tasks` creation understandable from intent rather than implementation.
2. Ship a narrow Phase 2A that can improve the UI without requiring new backend contracts.
3. Provide at least one fully working creation path: Reminder.
4. Show Watch, Ingest, Recurring Question, Agent Task, and Advanced Task paths with accurate capability states.
5. Preserve Watchlists as a separate, first-class workspace and avoid limiting existing Watchlists UX.
6. Avoid treating GitHub, YouTube, or any other source as the primary product structure.
7. Provide deterministic prompt-style template finding without pretending to parse or configure a full task from natural language.
8. Establish Phase 2B product contracts for future actionable Watch/Ingest templates.
9. Improve power-user speed through deep links, URL-addressable tabs/templates, and clear handoffs.
10. Make status, result destination, and recovery expectations explicit before a task is created.

## Non-Goals

- Do not remove, limit, or simplify existing Watchlists workflows.
- Do not build a generic workflow builder.
- Do not implement new backend task families in Phase 2A.
- Do not infer schedule, source, filter, credentials, or notification settings from free text in Phase 2A.
- Do not make recurring RAG or Agent Task scheduling look available until product contracts exist.
- Do not add run-now, dry-run, bulk actions, export, saved views, or cross-task Results/Runs tabs in Phase 2A.
- Do not redesign Home in Phase 2A, although creation copy must accurately describe result destinations supported today and planned later.

## Users And Jobs

| User | Job | UX requirement |
| --- | --- | --- |
| First-time automation user | Understand what can be automated and create something safe | Intent cards, plain-language status, previews, validation, honest unavailable states |
| Researcher / analyst | Watch changing sources or keep checking for an unanswered question | Clear distinction between Watch, Ingest, and planned Recurring Question |
| Watchlists power user | Keep using Watchlists for source/report workflows while seeing schedules centrally | Deep links, no duplicate simplified Watchlists editor in `/scheduled-tasks` |
| Automation power user | Jump quickly to a task type, inspect state, create another task | Direct URLs, template preselection, post-create "Open task" and "Create another" |
| Operator / debugger | Know what is scheduled, blocked, running, failed, or externally managed | Capability states, dependency messages, domain handoffs, status explanations |

## Issues Addressed By This Spec

| Severity | Issue | Affected workflow | Why it matters | Phase 2 direction |
| --- | --- | --- | --- | --- |
| P0 | Creation implies more capabilities than are visible or supported | All non-reminder templates | Users lose trust if a task appears created but cannot run or surface results | Capability states must prevent fake creation |
| P0 | Watchlists could be accidentally collapsed into `/scheduled-tasks` | Watch and Ingest | Existing Watchlists users have a separate persona/job and need deep controls | `/scheduled-tasks` summarizes and starts/hands off; Watchlists owns deep setup |
| P1 | Source-specific examples distort the product IA | GitHub, YouTube, RSS, feeds, sites | Users may think the workbench is only for a few vendors | Organize templates by intent, then accept examples inside the intent |
| P1 | First-time users do not know where results appear | All creation flows | Result discovery is central to trust in automation | Review and success states must explain Home, task detail, and domain destinations accurately |
| P1 | Planned RAG/Agent capabilities can look broken if mixed with available templates | Recurring Question, Agent Task | Users need to distinguish unavailable, planned, and setup-required work | Planned cards are visible but non-actionable except for domain docs/handoff |
| P1 | Free-text prompt could overpromise AI setup | Prompt-style creation | Users may expect complete natural language automation setup | Deterministic template finder only; no config inference in Phase 2A |
| P2 | Power users need fast paths | Create and manage many tasks | Slow repeated navigation hurts frequent users | URL-addressable tabs/templates, post-create actions, advanced handoff chooser |
| P2 | Duplicate Watch/Ingest setup could create noisy jobs later | Watch/Ingest | Duplicate monitors or ingests cause notification spam and wasted processing | Phase 2B requires duplicate detection or warning before actionable creation |
| P2 | Extension-sized layouts can break creation flows | Browser extension | The extension shares the route and has constrained width | Responsive cards, stepper, and validation states are required |

## Phase 2A Scope

Phase 2A should be frontend-first and use existing APIs only.

In scope:

- Convert `/scheduled-tasks` into URL-addressable tabs: Overview, Tasks, Create.
- Add a Create tab with intent templates and lightweight filters.
- Add a deterministic prompt-style "Find a template" input.
- Add a guided wizard shell and copy model for supported templates.
- Make Reminder the fully working creation template.
- Represent Watch for new items, Ingest new content, Recurring Question, Agent Task, and Advanced Task with accurate states.
- Add handoff flows to Watchlists, RAG/Knowledge, and ACP/Agent domains where appropriate.
- Include copyable setup summaries when deep prefill is not available.
- Preserve Phase 1 table/detail/overview behavior.
- Preserve existing Watchlists links and ownership language.

Out of scope:

- New normalized result inbox APIs.
- New Home automation inbox cards.
- New Watchlists creation adapters.
- New RAG schedule primitive.
- New ACP/Agent schedule integration.
- Bulk actions, saved views, dry run, run now, export.

## Phase 2B Scope

Phase 2B should be a dependency and product-contract phase for actionable Watch/Ingest creation.

It should define, then later implement, the minimum contracts needed for:

- Creating a Watchlists-backed monitor from `/scheduled-tasks` without losing access to Watchlists deep configuration.
- Creating or handing off an Ingest job with clear duplicate, indexing, transcript/download, and searchable-destination behavior.
- Reporting capability health before users commit.
- Warning about duplicates when a source is already watched or ingested.
- Returning created task IDs and domain IDs so success states can open exact task details.
- Surfacing run/result links through current `/scheduled-tasks` details and later Home/results inbox work.

Phase 2B should not replace Watchlists. It should make `/scheduled-tasks` a safe front door to a subset of Watchlists-backed automations.

## Information Architecture

Phase 2A should keep the workbench compact:

| Tab | URL | Purpose |
| --- | --- | --- |
| Overview | `/scheduled-tasks` | Operational summary and latest attention states from Phase 1 |
| Tasks | `/scheduled-tasks?tab=tasks` | Unified task list and detail drawer from Phase 1 |
| Create | `/scheduled-tasks?tab=create` | Intent template gallery, prompt-style finder, and creation/handoff flows |

Do not add Runs or Results tabs in Phase 2A. Those remain target IA from the broader PRD and should wait for normalized run/result contracts.

Deep link behavior:

| URL | Behavior |
| --- | --- |
| `/scheduled-tasks` | Opens Overview |
| `/scheduled-tasks?tab=tasks` | Opens Tasks |
| `/scheduled-tasks?tab=create` | Opens Create |
| `/scheduled-tasks?tab=create&template=reminder` | Opens Create with Reminder selected |
| `/scheduled-tasks?tab=create&template=watch` | Opens Create with Watch for new items selected |
| Invalid template | Opens Create with a non-blocking "Template not available" message |
| Created reminder success | Opens created task detail by default, with "Create another" as secondary |

## Create Tab Model

The Create tab should answer "What do you want to automate?" before asking for schedule details.

Recommended structure:

1. **Intent search.** Label: "Find a template". Placeholder: "Try reminder, watch changes, ingest content, question, or agent".
2. **Lightweight filters.** Suggested filters: All, Available now, Watch, Ingest, Research, Agent, Advanced.
3. **Template grid.** Cards grouped by intent, not source.
4. **Capability messages.** Each card must show whether it can be created here, needs setup elsewhere, is managed by Watchlists, is planned, or is unavailable.
5. **Selected template panel or wizard.** Use the same underlying model for available templates and handoff templates.

### Template Catalog

| Template | User intent | Phase 2A state | Primary CTA | Secondary action |
| --- | --- | --- | --- | --- |
| Reminder | "Remind me later or repeatedly" | Available | Create reminder | Create another after success |
| Watch for new items | "Tell me when something new appears" | Managed in Watchlists or Needs setup | Continue in Watchlists | Copy setup summary |
| Ingest new content | "Add new content to my library/search" | Managed in Watchlists or Needs setup | Continue in Watchlists | Copy setup summary |
| Recurring question | "Keep asking this question as new data arrives" | Planned | View planned capability | Open Knowledge/RAG area when available |
| Agent task | "Send a prompt/message to an agent later" | Planned | View planned capability | Open ACP/Agent area when available |
| Advanced task | "I know the domain I need" | Available as handoff chooser | Choose destination | Copy setup summary |

### Capability States

| State | Meaning | UX behavior |
| --- | --- | --- |
| Available | Can be created from `/scheduled-tasks` now | Enabled CTA, full wizard, success opens task |
| Needs setup | Could work after configuration or credentials | CTA opens setup destination or health guidance |
| Managed in Watchlists | Domain already owns the required deep setup | CTA opens Watchlists with explanation |
| Planned | Product direction exists but this template cannot create a task yet | Disabled create CTA; show honest future state and domain link if useful |
| Unavailable | Dependency is down or unsupported in current environment | Disabled create CTA with recovery action when available |

Card copy must never imply a task was created when the user was only handed off to another surface.

## Deterministic Template Finder

Phase 2A should include a prompt-style input, but it must behave as deterministic template matching.

Rules:

| Keywords | Suggested template |
| --- | --- |
| `remind`, `reminder`, `later`, `daily`, `weekly`, `monthly` | Reminder |
| `watch`, `monitor`, `new`, `changes`, `alert`, `notify` | Watch for new items |
| `ingest`, `scrape`, `download`, `index`, `searchable`, `library` | Ingest new content |
| `question`, `answer`, `rag`, `search again`, `keep looking` | Recurring question |
| `agent`, `prompt`, `message`, `assistant`, `acp` | Agent task |
| `advanced`, `workflow`, `custom` | Advanced task |

Constraints:

- Do not infer source URL, schedule, filters, credentials, notification settings, or destination from free text.
- Do not generate raw JSON or backend config.
- Show matched templates with clear confidence copy, such as "Best match: Watch for new items".
- If no match is found, show a neutral fallback: "No exact template match. Choose a template below or start from Advanced task."
- Keep the matching logic testable as a pure helper.
- Structure the model so a later AI draft assistant can replace or augment matching without changing the card/wizard IA.

## Creation Flow

All Phase 2A templates should use the same conceptual flow, even when some steps collapse for handoff cards.

| Step | Purpose | Phase 2A behavior |
| --- | --- | --- |
| 1. Intent | Choose what the user wants to automate | Template card or matched prompt result |
| 2. Configure | Capture task-specific inputs | Reminder fields are editable; handoff templates collect only safe summary details if supported |
| 3. Preview | Explain what will happen before save | Reminder shows schedule/result destination; handoff templates show setup summary and owner |
| 4. Schedule | Pick one-time or recurring cadence | Reminder supports current schedule controls; handoff templates explain schedule happens in domain workspace |
| 5. Review | Confirm name, schedule, owner, results destination | Required before create or handoff |
| 6. Create / Continue | Save or open domain workspace | Reminder creates task; Watch/Ingest continues in Watchlists; planned templates do not create |

After successful reminder creation:

- Default action: open created task detail in Tasks.
- Secondary action: create another.
- Confirmation copy: "Reminder scheduled. Results and status appear in Tasks. Notifications follow your current reminder settings."

After Watch/Ingest handoff:

- Do not show "Task created".
- Show "Setup continues in Watchlists".
- Include copyable setup summary when a prefilled deep link is not supported.
- Link to Watchlists in a new or current route according to existing navigation patterns.

## Watchlists Boundary

Watchlists remains the deep workspace. `/scheduled-tasks` should not rebuild Watchlists inside a simpler modal.

| Concern | Owned by `/scheduled-tasks` | Owned by Watchlists |
| --- | --- | --- |
| Intent selection | Yes | Optional entry points |
| Basic explanation of Watch vs Ingest | Yes | Yes |
| Source collection management | No | Yes |
| Source curation and bulk import | No | Yes |
| Scraping and ingest tuning | No | Yes |
| Filters and extraction rules | No except future safe presets | Yes |
| Reports, digests, outputs | No | Yes |
| Detailed run activity | Summary/deep link only | Yes |
| Created task visibility | Yes, after product contract exists | Yes |
| Pause/resume | Only if current APIs safely support it | Yes |
| Failure diagnosis | Summary/deep link only in Phase 2A | Yes |

### Watch vs Ingest Decision Guide

Use this distinction throughout copy and card grouping:

| Intent | User language | Result expectation |
| --- | --- | --- |
| Watch for new items | "Tell me when something new appears" | Alerts, review items, reports, awareness |
| Ingest new content | "Add new content to my library/search" | Searchable media, indexed content, knowledge base updates |

Examples can include GitHub issues, YouTube channels, RSS feeds, site pages, forums, vendor advisories, publications, or any future source type. The UI should not privilege GitHub or YouTube as the only important sources.

### Handoff Requirements

When Phase 2A cannot safely create a Watch/Ingest task:

- Explain why setup continues elsewhere.
- Preserve the user's chosen intent and any safe source text in a copyable summary.
- Deep-link to the closest Watchlists creation/setup destination if available.
- Avoid hidden side effects.
- Avoid "created", "scheduled", or "active" success language.
- Provide a return path to `/scheduled-tasks`.

Suggested copy:

- "Setup continues in Watchlists, where source rules, ingest behavior, outputs, and reports are managed."
- "No scheduled task has been created yet."
- "Copy setup summary"
- "Open Watchlists setup"

### Phase 2B Watch/Ingest Acceptance Outputs

Before Watch/Ingest templates become "Available", the product must support:

- The created automation appears in `/scheduled-tasks` Tasks.
- Success opens the exact task detail, not only `/watchlists`.
- The review screen explains what will be checked and where results appear.
- Duplicate warnings are understandable and actionable.
- Failures are recoverable through a specific domain link.
- Watchlists deep links open the correct monitor, run, output, or setup surface.
- Existing Watchlists workflows remain unchanged.

## Workflow Walkthroughs

These walkthroughs use concrete examples, but the product should present them as examples inside broader intent templates.

### Example 1: Watch New Repository Issues

First-time path in Phase 2A:

1. User opens Create and searches "watch new issues".
2. The finder suggests Watch for new items.
3. The card explains: "Use this when you want alerts or review items when a source changes."
4. If Watchlists prefill is not available, the user sees "Setup continues in Watchlists" and a copyable summary:
   - Intent: Watch for new items
   - Example source: repository issues URL
   - Suggested rule: ignore bot/system users where supported
   - Desired result: notify when matching items appear
5. The CTA opens Watchlists setup. No task is claimed as created.

Power-user path in Phase 2A:

1. User opens `/scheduled-tasks?tab=create&template=watch`.
2. User chooses "Continue in Watchlists" or copies the summary.
3. User returns to `/scheduled-tasks?tab=tasks` after setup to verify the Watchlists-backed job appears, if current integration supports it.

Phase 2B actionable target:

- The user can paste a source URL, preview recent items, choose notification behavior, detect duplicate monitors, create the Watchlists-backed monitor, and open the resulting task detail.

### Example 2: Ingest New Channel Or Feed Content

First-time path in Phase 2A:

1. User searches "keep this channel searchable" or chooses Ingest new content.
2. The card explains: "Use this when you want new content added to your library and search surfaces."
3. The UI does not assume YouTube is the only or primary source.
4. If no safe creation API exists, setup continues in Watchlists with a copyable summary:
   - Intent: Ingest new content
   - Source URL: user-provided URL if collected
   - Destination: media library/searchable knowledge, if supported by domain setup
   - Schedule: configure in Watchlists

Phase 2B actionable target:

- The user can preview source resolution, understand duplicate/skipped items, choose ingest/searchable destinations, create the automation, and later see ingested item counts, failures, and links to media/search surfaces.

### Example 3: Recurring Question

Phase 2A:

1. User searches "keep looking for an answer".
2. The finder suggests Recurring question.
3. The card is marked Planned.
4. Copy explains: "This will run a saved question against newly available data and surface promising matches. It is not yet available from Scheduled Tasks."
5. If a current Knowledge/RAG route is appropriate, provide an "Open Knowledge" style handoff, but do not create a task.

Future target:

- User enters a question, chooses source scope, chooses a threshold/review behavior, schedules recurrence, and sees results with citations, review state, dismiss/keep watching, and mark solved.

### Example 4: Scheduled Agent Message

Phase 2A:

1. User searches "send this prompt to an agent tomorrow".
2. The finder suggests Agent task.
3. The card is marked Planned unless a safe ACP/Agent schedule UI exists.
4. Copy explains that future setup will choose agent, workspace/context, message, governance/sandbox, schedule, and result destination.
5. Provide an ACP/Agent domain link only as a handoff.

Future target:

- User selects an agent, composes a message, previews action scope, chooses schedule, confirms governance and result destination, then inspects transcripts/artifacts/failures from task detail and ACP deep links.

## Advanced Task Handoff

Advanced task is not a raw config editor.

It should be a domain chooser for users who already know where the deeper setup belongs:

| Destination | Copy | CTA |
| --- | --- | --- |
| Watchlists | "Sources, monitors, reports, digests, and ingest rules" | Open Watchlists |
| Workflows | "Multi-step or internal orchestration workflows" | Open Workflows when available |
| ACP / Agent Tasks | "Agent prompts, sessions, artifacts, and governance" | Open Agent Tasks / ACP |
| RAG / Knowledge | "Retrieval scope, saved questions, sources, and citations" | Open Knowledge / RAG |

The Advanced path should allow a copyable setup summary, but should not expose raw JSON, cron-only inputs, or backend implementation vocabulary in Phase 2A.

## Status, Feedback, And Trust

Phase 2A must make status and ownership visible before the user commits.

### Review Screen Questions

Every Review screen or handoff screen should answer:

- What will run?
- Who manages it: Scheduled Tasks, Watchlists, RAG/Knowledge, ACP/Agent Tasks, or another domain?
- When will it run or where will the schedule be configured?
- Where will results appear?
- What can fail?
- Where does the user go to inspect or change deeper settings?

### Required States

| State | UX requirement | Example copy |
| --- | --- | --- |
| Empty Create | Explain automations by intent and show template cards | "Choose what you want to automate." |
| Loading templates | Use skeleton cards or lightweight loading text | "Loading automation templates..." |
| Available template | Enable full wizard | "Create reminder" |
| Needs setup | Explain missing dependency and action | "Connect or configure this in Watchlists before scheduling." |
| Managed elsewhere | Explain owner and handoff | "Managed in Watchlists" |
| Planned | Prevent create while preserving roadmap clarity | "Planned capability" |
| Handoff success | Confirm no task was created | "Setup continues in Watchlists." |
| Create success | Confirm exact created entity | "Reminder scheduled." |
| Create failure | Explain cause and preserve inputs | "Could not schedule reminder. Check the highlighted fields and try again." |
| Invalid deep link | Keep user in Create without blocking | "That template is not available. Choose another template." |

## Results And Home Surfacing Model

Phase 2A should not implement a new Home automation inbox. It must still set correct expectations.

Recommended copy policy:

- For Reminder: use only result/notification language supported by the existing reminder behavior.
- For Watchlists-managed tasks: say results and detailed activity are managed in Watchlists; `/scheduled-tasks` shows the task row and available deep links when the job is visible through the control plane.
- For planned RAG/Agent tasks: do not promise Home surfacing yet.

Future Home surfacing model:

| Home module | Shows | Action |
| --- | --- | --- |
| Needs attention | Failed or blocked scheduled automations | Open task, open domain setup, retry when safe |
| Latest automation results | New watched items, ingested content, RAG matches, agent outputs, reminders | Review, dismiss, open exact result |
| Running now | Active runs with current step and elapsed time | Open run, cancel when safe |
| Upcoming | Important next runs | Open task, pause when safe |

Home cards should eventually deep-link to exact task/run/result IDs, not only broad domain pages.

## Management And Monitoring Expectations

Phase 2A should preserve the Phase 1 management model and make creation outcomes flow into it.

After creating or returning from setup, users should be able to use the existing `/scheduled-tasks` Tasks tab to understand:

- whether the automation is waiting, running, completed, failed, paused, disabled, blocked, or managed elsewhere;
- last run and next run when the current control-plane contract exposes them;
- whether deeper run activity, outputs, or settings are available through Watchlists or another domain link;
- whether a failure is recoverable in `/scheduled-tasks` or requires the domain workspace;
- which task detail drawer or domain URL to open next.

Phase 2A should not invent run history or result details that are not available through existing contracts. If a created or handed-off automation cannot be observed in `/scheduled-tasks`, the success or handoff state must say so directly and point to the owning workspace.

## Browser Extension Behavior

The browser extension shares the scheduled-tasks route, so the Phase 2A Create experience must work at constrained extension widths.

Phase 2A extension expectations:

- The extension can open `/scheduled-tasks?tab=create` and direct template URLs.
- The same intent cards, capability states, and handoff copy should render in extension-sized layouts.
- If the extension has a current-page URL available to pass safely, it may prefill only a visible source field or setup summary. It must not silently infer filters, schedule, credentials, notification policy, or task ownership.
- If current-page context detection is not implemented, the UI should behave exactly like the WebUI Create tab.
- Context-aware creation from current pages remains a later phase unless existing extension and route contracts already support it safely.

Future extension target:

- Recognize broad page contexts such as source pages, feeds, articles, research pages, and agent/workspace pages.
- Suggest an intent template with confidence.
- Hand off to the same Create tab and keep deep configuration in the owning domain workspace.

## UX Copy Recommendations

Preferred labels:

| Avoid | Use |
| --- | --- |
| Create Reminder Task | Create scheduled task |
| Native | Managed here |
| External managed | Managed in Watchlists |
| Cron | Custom schedule |
| Run at | Run once at |
| Enabled | Task is active |
| scheduled | Waiting for next run |
| disabled | Disabled |
| failed | Needs attention |
| GitHub monitor | Watch for new items |
| YouTube ingest | Ingest new content |
| Advanced config | Advanced task |

Template card copy:

- Reminder: "Schedule a one-time or recurring reminder."
- Watch for new items: "Get notified when a source has new matching items."
- Ingest new content: "Add new source content to your library and search surfaces."
- Recurring question: "Keep checking for an answer as new data arrives."
- Agent task: "Send a message to an agent at a scheduled time."
- Advanced task: "Choose the workspace that owns deeper automation setup."

Button copy:

- "Create reminder"
- "Continue in Watchlists"
- "Copy setup summary"
- "Open task"
- "Create another"
- "Choose destination"
- "View planned capability"

Trust copy:

- "No scheduled task has been created yet."
- "Setup continues in Watchlists."
- "This task is managed in Watchlists. Scheduled Tasks shows status and links when available."
- "Preview explains what will happen. It does not create results."
- "Results destination depends on the task type and connected workspace."

## Accessibility And Responsive Requirements

- Tabs must be keyboard operable and expose selected state.
- Template cards must be reachable by keyboard and use buttons/links with accessible names.
- Status must use text and icon, not color alone.
- Prompt suggestions must use a polite live region.
- Wizard step changes must move focus to the step heading.
- Validation messages must be associated with their fields.
- The extension-sized layout must avoid table-only or wide-card-only interactions.
- Template cards should stack cleanly below narrow widths.
- The Review step must remain readable without horizontal scrolling.
- Loading states should not trap focus.
- Planned/unavailable cards should not be focus-dead ends; users need a meaningful next action or explanation.
- Reduced-motion preferences should be respected for step transitions.

## Power-User Requirements

Phase 2A should provide speed without adding heavy management features:

- Direct Create tab URL.
- Direct template URLs.
- Keep filters and prompt matcher lightweight and predictable.
- Success defaults to the created task detail.
- Secondary "Create another" supports repeated entry.
- Advanced task opens a domain chooser rather than forcing users through beginner copy.
- Tasks tab remains the inspection and management surface after creation.

Deferred power-user features:

- Saved views.
- Bulk pause/resume.
- Run now.
- Dry run.
- Duplicate task.
- Export.
- Cross-task Runs tab.
- Cross-task Results tab.

## Backend And Product Dependencies For Phase 2B

Phase 2B should not start from implementation details. It should define product-facing contracts first.

| Dependency | Needed for | Product requirement |
| --- | --- | --- |
| Capability reporting | Accurate template states | UI can tell whether Watch, Ingest, RAG, Agent, and reminders are available, blocked, setup-required, or planned |
| Watchlists create/prefill contract | Actionable Watch/Ingest templates | `/scheduled-tasks` can pass intent/source/safe defaults without duplicating Watchlists controls |
| Duplicate detection | Watch/Ingest safety | User sees existing monitors/ingest jobs for the same source before creating another |
| Created entity response | Post-create success | API returns scheduled task ID plus domain IDs so UI can open exact detail |
| Run/result link contract | Trust and discovery | Task detail can link to latest run/output/result and eventually Home cards |
| Failure reason contract | Recovery | User sees missing auth, invalid source, rate limit, duplicate, parser, transcript, or indexing failure with next action |
| Pause/resume support flag | Safe controls | UI only shows controls when the domain supports them safely |
| Result destination metadata | Review step and Home | UI can accurately say where results will appear and how to inspect them |

## Acceptance Criteria

### Phase 2A Product Acceptance

- `/scheduled-tasks` supports URL-addressable Overview, Tasks, and Create tabs.
- Create tab presents templates by intent, not by source vendor.
- Reminder is the only fully available creation template unless an existing API safely supports more.
- Watch for new items and Ingest new content do not claim creation success unless a real task is created.
- GitHub and YouTube appear only as examples inside broader Watch/Ingest copy, not as top-level IA assumptions.
- Deterministic template finder suggests templates from keyword matching and does not infer config.
- Planned Recurring Question and Agent Task cards clearly communicate that task creation is not available yet.
- Advanced task is a domain handoff chooser, not a raw JSON/config builder.
- Handoff screens include owner, next step, and "No scheduled task has been created yet" where applicable.
- Reminder success opens task detail by default and offers Create another.
- Existing Watchlists UX remains reachable and unchanged.
- Keyboard, screen-reader, and extension-sized responsive flows are covered by focused tests or documented verification.

### Phase 2B Product Acceptance

- Watch/Ingest can become Available only after the product can create or prefill safely through a real domain contract.
- Created Watch/Ingest automations appear in `/scheduled-tasks` Tasks.
- Success opens exact task detail.
- Duplicate warnings are shown before save where possible.
- Review explains schedule, source, result destination, owner, and failure/recovery path.
- Watchlists remains the deep editor for source collections, filters, ingest tuning, reports, digests, outputs, and detailed activity.
- Failure states are recoverable and domain-specific.
- Home/result surfacing has exact links before UI promises exact Home review cards.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Users expect prompt input to create a full automation | Name it "Find a template"; show matched templates, not generated config |
| Watch/Ingest cards feel broken if they only hand off | Use clear state labels and copyable setup summaries; make Reminder the working example |
| Product over-indexes on GitHub/YouTube | Keep top-level taxonomy intent-based; examples live inside help text |
| Power users find the wizard slow | Support direct template URLs and post-create Create another |
| Planned cards create frustration | Provide honest "Planned capability" state and useful domain links only where they help |
| Future backend implementation leaks into Phase 2A | Keep Phase 2B as dependency contracts; do not add fake adapters |

## Open Questions

- Which current Watchlists route is the best setup destination for Watch and Ingest handoffs?
- Does the current Watchlists UI support safe prefilled setup through query parameters, or should Phase 2A use copyable summaries only?
- Can current reminder notification behavior be described precisely enough in Review copy, or should copy stay generic?
- Should Advanced task include Workflows in Phase 2A if the route is not stable, or mark it as planned?
- What telemetry, if any, is acceptable in this local/self-hosted product for measuring creation success?

## Implementation Notes For Later Planning

- Create a focused implementation plan before editing product code.
- Keep Phase 2A tests mostly in pure helper and component coverage:
  - template catalog state;
  - deterministic matcher;
  - tab/query-param routing;
  - reminder template success path;
  - handoff copy contract;
  - invalid template deep link;
  - keyboard/accessibility assertions where feasible.
- Use existing Phase 1 ScheduledTasks helpers and components where possible.
- Avoid touching Watchlists implementation unless a route link or copy contract requires a small, reviewed change.
- Bandit is not required for a frontend-only implementation slice unless backend Python changes are introduced.
