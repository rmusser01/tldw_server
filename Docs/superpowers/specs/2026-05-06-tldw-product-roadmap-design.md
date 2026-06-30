# tldw Product Roadmap Design

Date: 2026-05-06
Owner: Codex collaboration session
Status: Reviewed with user, pending implementation planning
Backlog: TASK-97

## Summary

Create one aligned product roadmap for `tldw_server`, the WebUI, the browser
extension, SaaS packaging, enterprise seat licensing, and the OSS self-hosted
base. The roadmap uses the existing workspace UI paradigm as the product spine:
users enter a persistent workspace, bring in workplace inputs, understand them
with source-grounded reasoning, produce living work products, delegate bounded
work to agents, and review outputs before accepting or exporting them.

The roadmap has three nested horizons:

1. A 6-8 week horizon focused on SaaS-ready first value for individual and
   small-team users.
2. A 6-month horizon focused on enterprise pilot readiness.
3. A 12-month horizon focused on category leadership for workplace productivity
   and analysis.

The roadmap is not a new product surface invented beside the current app. It is
a consolidation and packaging plan for the product capabilities already present
or already planned in the repo.

## Commercial Frame

The product is commercially aimed at general workplace productivity for
white-collar work. It should not be framed narrowly as a RAG tool, a media
analysis tool, or a study app.

The category is still forming. The useful frame is:

- Enterprise: an AI workbench plus research and analysis operating system,
  sold through seat licensing.
- SaaS: a private and local-first productivity agent platform for individuals,
  consultants, and small teams.
- OSS: the same core product direction for technical users who can self-host
  and manage the stack themselves.

The primary market position is department-agnostic, with early proof points
leaning toward strategy, operations, product, engineering, and technical
documentation teams.

## Product Spine

The central loop is:

1. Ingest or connect workplace inputs.
2. Understand them with source-grounded, cited reasoning.
3. Produce a living work product.
4. Delegate bounded work to task agents.
5. Review, revise, accept, export, or assign the result.

The workspace is the container for this loop. Work products are living workspace
artifacts first and polished exports second.

The flagship work product templates are:

1. Executive brief.
2. Research dossier.
3. Competitive or market memo.
4. Technical or project spec.

The first-value experience should be template-led, while still allowing users to
start from whichever path feels natural:

- create from sources
- connect or import a workplace source
- start in chat and progressively add sources, tasks, and outputs

## Differentiators

The roadmap should protect these differentiators in this order:

1. Source-grounded traceability: every serious answer and work product can show
   the sources, citations, decisions, and artifacts behind it.
2. Private and local-first deployment options: SaaS for convenience, enterprise
   for compliance, OSS for self-hosted control.
3. Workspace continuity: sources, chats, tasks, outputs, agents, history, and
   decisions stay together.
4. Task-agent execution: agents do bounded real work over workspace context and
   return reviewable outputs.
5. Reviewable outputs: automation is visible, auditable, and reversible.

## Product Pillars

### Workspace System

The workspace is the commercial product unit. It should eventually contain
sources, chats, notes, generated artifacts, tasks, decisions, agents, review
state, and history.

The first roadmap milestone is workspace consolidation discovery. The roadmap
leans toward `WorkspacePlayground` as the canonical shell, but this remains an
explicit decision to validate with current route, state, and user-flow evidence.

### Source-Grounded Intelligence

Every serious answer or output should show what it used, why it was trusted, and
where claims came from. This includes ingestion, connectors, RAG, citations,
source comparison, traceability, confidence states, and review states.

### Work Product Builder

The work product builder turns selected sources, chat context, and template
intent into living workspace artifacts. Executive briefs, research dossiers,
market memos, and technical specs are templates over the same workspace model,
not separate products.

### Agentic Execution And Review

Agents should run bounded tasks inside a workspace, use selected sources and
tools, and return outputs with logs, citations, approvals, rejection reasons,
and retry or triage state.

This pillar connects ACP, MCP, workflows, jobs, scheduled tasks, and output
review.

### Commercial Packaging And Deployment

OSS, SaaS, and enterprise share the same product spine. They differ by
onboarding, support, managed infrastructure, auth and admin depth, compliance,
quotas, deployment mode, and integration promises.

### Trust, Admin, And Reliability

Setup, health, auth, permissions, billing and usage, audit, backup/export,
observability, and security are product features. They are especially important
for enterprise seat licensing and team adoption.

## Current Repo Anchors

The roadmap should build from existing surfaces and docs:

- `apps/extension`
- `apps/packages/ui/src/components/Option/WorkspacePlayground`
- `apps/packages/ui/src/components/Option/ChatWorkspace`
- `apps/packages/ui/src/components/DocumentWorkspace`
- `apps/packages/ui/src/store/workspace.ts`
- `apps/packages/ui/src/types/workspace.ts`
- `Docs/Product/WebUI/Workspace_Playground_Redesign.md`
- `Docs/Design/Workspace_Persistence_Architecture.md`
- `Docs/Product/WebUI/WebUI_UX_Strategic_Roadmap_2026_02.md`
- `Docs/Design/tldw_web_design_system_contract.md`
- `Docs/Product/ACP_Agent_Orchestration_PRD.md`
- `Docs/Product/RAG-Upgrades-PRD.md`

The current workspace model already includes sources, selected source state,
generated artifacts, quick notes, chat sessions, saved workspaces, persistence,
artifact payload offload, and source transfer. The roadmap should not create a
parallel "new roadmap workspace."

## 6-8 Week Horizon: SaaS-Ready First Value

Success bar: a new individual or small-team user can create a workspace, bring
in useful inputs, generate one flagship work product, inspect citations and
source lineage, and understand what to do next without learning the internals.

Scope cut line: this horizon should define all four flagship templates, but it
does not need to fully implement all four end-to-end. The implementation plan
should pick one golden-path template for complete first-value execution, then
leave the other three as well-specified templates or thin pilots if capacity
allows. The horizon should also avoid full route consolidation; it should decide
the canonical workspace model first and implement only the smallest UX changes
needed to prove the golden path.

First implementation slice: canonical workspace decision record, typed
server/local workspace bridge, executive brief template, and generated artifact
review contract.

### Milestone 1: Workspace Consolidation Discovery

Outcome: a decision record for the canonical workspace model.

Deliverables:

- Inventory `WorkspacePlayground`, `ChatWorkspace`, and `DocumentWorkspace`.
- Map route naming, entry points, handoffs, and user intent.
- Map shared and divergent state: sources, selected sources, staged sources,
  chat sessions, notes, artifacts, persistence, and viewport behavior.
- Identify the boundary between current browser-local workspace persistence and
  the server-backed workspace record needed for cross-device, SaaS team, and
  enterprise use.
- Decide whether `WorkspacePlayground` becomes the canonical shell, or whether
  the routes remain separate under one shared workspace model.
- Record how chat-first and document-focused workflows fit the canonical
  workspace direction.

Decision bias: `WorkspacePlayground` likely becomes the canonical shell, with
chat-first and document-focused experiences becoming modes or specialized
entry points if discovery supports that direction.

### Milestone 2: Workspace Template System V1

Outcome: templates become product primitives.

Deliverables:

- Define template metadata for executive brief, research dossier,
  competitive/market memo, and technical/project spec.
- Define source requirements, template prompts, output sections, review
  checklists, citation expectations, and export behavior.
- Connect templates to workspace artifacts rather than one-off generated text.
- Choose one golden-path template for full implementation planning, including
  artifact schema, source-lineage requirements, review states, and export target.
- Identify which parts can be implemented through existing outputs, slides,
  data tables, prompt studio, chatbooks, and workspace artifact types.

### Milestone 3: First-Value Onboarding Flow

Outcome: users can choose a template and enter through sources, connector/import,
or chat.

Deliverables:

- A template chooser or equivalent first-value entry point.
- Handoffs from upload, URL, paste, existing media, connector/import, and chat
  into the same workspace context.
- Clear setup, unavailable, degraded, empty, loading, retrying, permission
  denied, and review-needed states using shared design-system language.
- Package-scoped onboarding instrumentation for template selection, source
  addition, first output, citation inspection, and export or accept actions:
  managed SaaS events, enterprise policy-controlled audit and usage events, and
  OSS local-only or opt-in diagnostics.

### Milestone 4: Traceable Work Product Artifacts

Outcome: generated work products are reviewable workspace artifacts.

Deliverables:

- Source lineage and citation panel for generated artifacts.
- Artifact states such as draft, reviewing, accepted, needs revision, exported,
  and assigned where supported.
- Revise, accept, export, and assign affordances.
- Versioning direction for generated work products.
- Regression coverage for citation visibility and artifact state transitions.

### Milestone 5: Commercial Packaging Baseline

Outcome: the same product spine is mapped to OSS, SaaS, and enterprise.

Deliverables:

- Packaging matrix for OSS, SaaS individual/team, and enterprise.
- Feature-gating decisions for managed services, connectors, admin, usage,
  compliance, and deployment mode.
- Minimum SaaS setup decisions for individual versus team workspace ownership,
  invite and seat entitlement stubs, billing and usage gates, and which team
  collaboration features are explicitly deferred to the 6-month enterprise
  horizon.
- A clear first-horizon team cut line: either one owner with no real-time shared
  editing, or a narrow invite/read-only collaboration stub. Full shared team
  workspace semantics belong to the 6-month horizon.
- Public docs narrative for the workspace-first productivity workbench.
- Setup and deployment promises for each package.

## 6-Month Horizon: Enterprise Pilot Readiness

Success bar: the product can support enterprise pilot teams with shared work,
admin visibility, traceability, and reviewable agent execution.

Milestones:

1. Server-backed team workspaces with sharing, roles, workspace-level
   permissions, workspace membership, and migration or sync boundaries from the
   current browser-local workspace store.
2. Mature workplace connectors for a prioritized subset of Drive, Notion,
   GitHub, email, Slack, or comparable workplace inputs. The list is a candidate
   set, not a commitment to mature every connector in one horizon.
3. Agent task runs inside workspaces with logs, approval/rejection, retry, and
   triage states.
4. Template library expansion for strategy, operations, product, engineering,
   documentation, market intelligence, and recurring internal analysis.
5. Admin controls for usage, audit, retention, quotas, source governance,
   provider/model policy, and workspace lifecycle.
6. Reliability gates for workspace journeys, citation quality, generated
   artifact regressions, onboarding conversion, and connector health.

## 12-Month Horizon: Category Platform

Success bar: `tldw_server` defines a durable workplace productivity and
analysis platform around private, source-grounded, workspace-native AI work.

Milestones:

1. Workspace operating system: sources, outputs, tasks, agents, decisions,
   memory, and review state form one durable system.
2. Organization intelligence layer: reusable institutional memory with
   permissions, lineage, and traceability.
3. Advanced agent orchestration: multi-agent work plans, reviewer gates,
   recurring workflows, tool governance, and policy-aware execution.
4. Enterprise deployment flexibility: SaaS, VPC/private cloud, self-host,
   hybrid, local model, and compliance-sensitive modes.
5. Marketplace or library for templates, connectors, skills, agent packs, and
   evaluation packs.
6. Compliance and lifecycle controls for audit trails, retention, exports,
   legal holds, admin policy, observability, and recovery.

## Architecture Mapping

### Workspace Consolidation

Primary anchors:

- `WorkspacePlayground` for the current three-pane sources/chat/studio pattern.
- `ChatWorkspace` for the chat-first staged-source console.
- `DocumentWorkspace` for document-focused work.
- `workspace.ts` and `Workspace_Persistence_Architecture.md` for local
  persistence, split-key storage, IndexedDB offload, and payload bounds.

Architecture principle: consolidate the product model before consolidating
routes. Route consolidation should follow a decision record, not happen as a
premature rewrite.

Important gap: the current workspace persistence path is browser-local. That is
acceptable for proving individual first value, but it is not enough for
cross-device continuity, SaaS team workspaces, enterprise sharing, audit, or
workspace lifecycle governance. The roadmap should explicitly plan the
server-backed workspace record and its relationship to local cache before
enterprise pilot work starts.

### Work Product Builder

Primary anchors:

- `GeneratedArtifact` and `OUTPUT_TYPES` in `apps/packages/ui/src/types/workspace.ts`
- `WorkspacePlayground/StudioPane`
- backend outputs, slides, data tables, chatbooks, prompt management, and RAG
  search

Architecture principle: templates produce workspace artifacts with source
lineage and review state. Exports are snapshots from living artifacts.

Implementation planning should not treat "template" as prompt text only. A work
product template needs a minimal artifact contract: input requirements, output
sections, source-lineage fields, review states, revision behavior, and export
targets.

### Source-Grounded Reasoning

Primary anchors:

- media ingestion and Media DB
- RAG unified pipeline
- citations and post-generation verification
- evaluations
- connectors and sources routes

Architecture principle: source traceability is a product contract. It should be
visible in UI, returned in APIs where applicable, and tested through journey and
artifact checks.

### Agent Execution

Primary anchors:

- ACP and MCP
- workflows
- jobs
- scheduled tasks
- agent registry and agent task pages

Architecture principle: agent execution should become workspace-scoped and
reviewable. Agents should not feel like detached power-user tools when they are
being used for workplace productivity.

### Packaging

Primary anchors:

- setup and onboarding
- deployment profiles
- admin pages
- usage and billing surfaces
- AuthNZ, orgs, RBAC, audit, and provider policy
- browser extension and shared UI package

Architecture principle: packaging should not fork the product. It should expose
different managed guarantees over the same workspace-first model.

### Browser Extension

Primary anchors:

- `apps/extension`
- `apps/packages/ui/src/entries`
- shared route and sidepanel components under `apps/packages/ui/src/routes`
- web clipper and sidepanel flows

Architecture principle: the extension is a thin capture, quick-ingest, and
quick-chat launcher into the canonical workspace model. It should use the shared
UI package and shared route contracts. It should not become a separate workspace
surface with a divergent product model.

### Design System

Primary anchor: `Docs/Design/tldw_web_design_system_contract.md`.

Architecture principle: workspace, setup, artifact, and agent states should use
shared product state language: setup required, unavailable, degraded, empty,
loading, retrying, permission denied, blocked, error, ready, and review needed.

## Packaging Matrix

| Package | Primary buyer/user | Product promise | Emphasis |
| --- | --- | --- | --- |
| OSS | Technical individuals and teams | Self-hosted workplace AI workbench with full control | Transparency, extensibility, local/private deployment |
| SaaS | Individuals, consultants, small businesses, small teams | Fast first value with private/local-first positioning | Onboarding, templates, connectors, workspace continuity |
| Enterprise | Corporate teams buying seats | Governed workplace productivity and analysis OS | Admin, compliance, sharing, audit, source governance, deployment flexibility |

All packages should share the same roadmap spine. Differences belong in
deployment, entitlement, support, scale, governance, and managed-service depth.

Instrumentation must follow the package boundary:

- OSS: no managed telemetry by default; local diagnostics may be available and
  opt-in.
- SaaS: managed product events may measure activation and first value, but they
  should avoid capturing source content.
- Enterprise: audit, usage, and telemetry behavior must be policy-controlled
  and admin-visible.

## Validation Plan

Because this is a roadmap/design artifact, validation is document-level:

1. Verify the spec is grounded in current repo surfaces.
2. Verify the roadmap preserves the approved commercial framing.
3. Verify the roadmap does not invent a parallel workspace model.
4. Verify Backlog task acceptance criteria are checked after the spec is
   reviewed.
5. Run spelling/format sanity checks and `git diff --check`.
6. Skip Bandit with a recorded reason because this change is documentation-only.

## Non-Goals

This roadmap does not:

1. Implement workspace consolidation.
2. Decide final route consolidation without discovery evidence.
3. Redesign every WebUI screen.
4. Replace existing RAG, MCP, ACP, jobs, workflows, or connector plans.
5. Create a separate Chatbook or Textual roadmap.
6. Define pricing.
7. Define sales collateral beyond the product roadmap and packaging baseline.

## Open Decisions For Implementation Planning

1. Is `WorkspacePlayground` the canonical shell, or should it remain one route
   within a shared workspace model?
2. Which first template should be implemented as the golden path?
3. Which connector or input path should be the first SaaS-ready non-file import?
4. What is the minimum server-backed workspace record for SaaS/team and
   enterprise use, and how does it synchronize with the current local workspace
   cache?
5. What artifact review states already exist, and which need backend support?
6. Which admin and packaging gates are docs/config-only first, and which require
   product code changes?
7. What telemetry is acceptable for OSS, SaaS, and enterprise without violating
   the privacy posture?
