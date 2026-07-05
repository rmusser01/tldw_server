# Research Workspace NotebookLM Pro And Ultra Review Design

Date: 2026-07-04
Status: Reviewer-approved
Backlog: TASK-12133

## Summary

Research Workspace already exposes most of the shape a NotebookLM user expects:
sources, selected-source chat, citations and retrieval diagnostics, notes,
generated study outputs, sharing, import/export, Deep Research handoff, ACP task
handoff, and sandbox diagnostics.

The main gap is not raw capability. The gap is expectation management. A
NotebookLM Pro or Ultra subscriber expects the product to make import types,
grounded chat settings, source discovery, output formats, video/infographic
artifacts, and agentic work visibly obvious. tldw can meet the useful parts of
that expectation without cloning Google-specific sync, quotas, or media polish.

The recommended direction is:

1. Make NotebookLM-core parity obvious where tldw already has the plumbing.
2. Add cheap, honest versions of missing high-value outputs before full media
   generation.
3. Connect discovery results, Deep Research bundles, and browser capture into
   one workspace loop.
4. Reframe NotebookLM Ultra's agentic chat as tldw-native workspace agent tasks
   with visible run activity, tool calls, approvals, and produced files.

## Review Scope

This document compares the currently exposed Research Workspace page in the
WebUI and browser extension options route against current NotebookLM features
for Google AI Pro and Google AI Ultra subscribers. It is a competitive review
and remediation spec, not an implementation plan.

The implementation plan that follows this spec should stay narrower than this
comparison. The first build should prioritize visible parity and tldw
differentiation, not every NotebookLM feature.

## Sources And Freshness

Official Google NotebookLM docs checked on 2026-07-04:

- [Learn about NotebookLM](https://support.google.com/notebooklm/answer/16164461)
- [Add or discover new sources for your notebook](https://support.google.com/notebooklm/answer/16215270)
- [Use chat in NotebookLM](https://support.google.com/notebooklm/answer/16179559)
- [Upgrade NotebookLM](https://support.google.com/notebooklm/answer/16213268)

NotebookLM limits and feature availability are explicitly marked by Google as
subject to change, so downstream implementation work should refresh this table
before hard-coding any comparison copy.

## Current tldw Surface

### Route And Extension Placement

The canonical WebUI page is `/research-workspace`, rendered through
`apps/tldw-frontend/pages/research-workspace.tsx` and
`apps/packages/ui/src/routes/option-research-workspace.tsx`.

The browser extension also registers `/research-workspace` as an options/full
page route through
`apps/tldw-frontend/extension/routes/option-research-workspace.tsx` and the
extension route registry.

Research Workspace is not currently a browser extension sidepanel route.
`apps/packages/ui/src/routes/sidepanel-route-registry.tsx` exposes chat, agent,
companion, clipper, persona, flashcards, and settings, but not the full
Research Workspace.

### Capability Gates

Research Workspace advertises these capability ids in
`research-workspace-capabilities.ts`:

- `source_browse`
- `chat`
- `artifact_text_generation`
- `slides_generation`
- `audio_summary`
- `export_download`
- `sync_share`

This is a useful foundation for NotebookLM-style visible availability. New
discovery, media artifact, and agent task entrypoints should plug into this
model instead of adding one-off disabled states.

### Source Intake

The Add Source modal exposes these user-facing paths:

- Upload local files.
- Add existing media from the tldw media library.
- Add URLs.
- Paste text.
- Search server-backed web providers.

The visible media filters cover PDF, video, audio, website, document, text, and
email. The upload copy emphasizes PDF, DOCX, text, audio, and video. The
underlying project has broader ingestion ambitions, but this page does not make
NotebookLM-like image, CSV, PPTX, ePub, Google Drive, Google Docs, Google
Slides, Google Sheets, or YouTube transcript import feel first-class.

### Source Management

The Sources pane already has strong research-workbench affordances:

- Folder organization.
- Selected source scoping.
- Filters and sorting.
- Virtualized source lists.
- Source readiness and status drilldown.
- Source preview.
- Local annotations.
- Move, copy, and batch removal.

This is already competitive with NotebookLM's source panel, and more explicit
than NotebookLM for ingestion readiness.

### Chat

The Chat pane exposes:

- RAG mode and general mode.
- Selected-source scoping.
- Model selection.
- Full-source text mode.
- Top-k, similarity threshold, and reranking settings.
- Retrieval diagnostics, including chunks, sources used, relevance score,
  faithfulness, token use, and cost.
- Stop, retry, clear, undo, sharing, and lorebook activity export.

Compared with NotebookLM, tldw is stronger on transparent retrieval controls
and weaker on obvious beginner-facing chat settings such as conversational
style, answer length, save-to-note behavior, and citation-to-source navigation.

### Studio Outputs

The Studio pane exposes:

- Audio Summary.
- Summary.
- Mind Map.
- Report.
- Compare Sources.
- Flashcards.
- Quiz.
- Timeline.
- Slides.
- Data Table.

It also exposes higher-value literature work products, including Literature
Matrix, Corpus Gap Finder, Evidence-Bound Hypotheses, Research Proposal Pack,
and Executive Brief templates. These are differentiators, but they should be
more discoverable to users arriving with NotebookLM expectations.

Artifacts can be viewed, downloaded, regenerated, discussed in chat, saved to
notes, deleted, and in some cases edited. Slides and audio have settings.

### Notes, Sharing, And Workspace Controls

Quick notes support search, keywords, workspace-tagged notes, save, download,
and preview. Workspace header controls include workspace switching, share,
import/export, BibTeX export, templates, telemetry, banner customization,
archive/delete, agent tasks, ACP run history, and sandbox diagnostics.

Sharing supports team/org entries, share links, active shares, access levels,
cloning, password, max uses, expiry, and revoke. This is a credible
self-hosted answer to NotebookLM paid advanced sharing once the UX is polished.

## NotebookLM Feature Baseline

### Core Product Model

NotebookLM positions itself as an AI research assistant for uploading or
discovering sources, chatting over those sources with inline citations, and
transforming sources into study guides, briefings, audio overviews, mind maps,
and related outputs.

### Source Types And Discovery

NotebookLM currently lists these source paths:

- Audio files.
- Pasted text.
- Google Drive files, including Docs, Slides, and Sheets.
- Images.
- DOCX, TXT, Markdown, PDF, CSV, and PPTX files.
- Web URLs.
- ePub files.
- Public YouTube URLs with captions.
- Gemini Chats context.
- Fast Research over web or Drive.
- Gemini Deep Research from inside NotebookLM.

Important source limitations:

- Web URL import scrapes text content only.
- YouTube import uses the transcript only and requires public videos with
  captions.
- Google Drive sources auto-sync every few minutes, but Drive audio import is
  not supported.
- Google file comments and footnotes are not imported.
- Source discovery can import selected Fast Research or Deep Research results.
- With 5+ sources, NotebookLM can auto-label and categorize sources.

### Chat And Notes

NotebookLM chat offers:

- Grounded answers over uploaded sources.
- Inline citations with hover and source navigation.
- Checkbox-based source inclusion and exclusion.
- Chat settings for conversational style and response length.
- Save to note while preserving formatting, tables, and citations.
- Private chat history and a clear-history action.
- Chat entrypoints for notes, Audio Overview, and Mind Maps.

NotebookLM Ultra adds desktop-only agentic chat:

- Search the web.
- Run code.
- Create downloadable files, charts, images, structured data, spreadsheets, and
  slide decks.
- Modify uploaded charts.
- Version artifacts.
- Expand a visible activity/details view of what NotebookLM is doing.
- Complete research from chat and import reports and sources into the notebook.

Google describes these Ultra functions as experimental and requiring
supervision.

### Google AI Plan Limits

| Feature | Standard | Plus | Pro | Ultra 20 TB | Ultra 30 TB |
| --- | --- | --- | --- | --- | --- |
| Notebooks | 100/user | 200/user | 500/user | 500/user | 500/user |
| Sources | 50/notebook | 100/notebook | 300/notebook | 500/notebook | 600/notebook |
| Chats | 50/day | 200/day | 500/day | 2.5K/day | 5K/day |
| Audio Overviews | 3/day | 6/day | 20/day | 100/day | 200/day |
| Video Overviews | 3/day | 6/day | 20/day, cinematic 2/day | 100/day, cinematic 10/day | 200/day, cinematic 20/day |
| Reports | 10/day | 20/day | 100/day | 500/day | 1K/day |
| Flashcards | 10/day | 20/day | 100/day | 500/day | 1K/day |
| Quizzes | 10/day | 20/day | 100/day | 500/day | 1K/day |
| Mind Maps | 10/day | 20/day | 100/day | 500/day | 1K/day |
| Deep Research | 10/month | 3/day | 20/day | 75/day | 200/day |
| Data Tables | Limited | More limits | High limits | Higher limits | Highest limits |
| Infographics | Limited | More limits | High limits | Higher limits | Highest limits |
| Slide Decks and Revisions | Limited | More limits | High limits | Higher limits | Highest limits |
| Premium features | Advanced sharing paid; Custom Chat and Analytics available to everyone | Same | Same | Same | Same |
| Gemini model access | Access | Access | Higher access | Highest access | Highest access |
| Earlier access | Standard | Early | Priority | Priority | Priority |
| Watermark removal | No | No | No | Yes | Yes, for infographics and slide decks |

Do not copy NotebookLM's quota model into tldw. tldw should expose local
capacity, provider availability, and service health instead.

## Competitive Fit And Gap Matrix

Priority meanings:

- P0: must be handled before claiming NotebookLM-core parity.
- P1: valuable follow-up for Pro/Ultra expectation matching.
- Differentiator: keep and make clearer.
- Skip: do not implement for this effort.

| Area | Current tldw state | NotebookLM Pro/Ultra expectation | Recommendation |
| --- | --- | --- | --- |
| Source intake basics | Upload, existing media, URL, paste, server web search | Upload common docs, web, audio, YouTube, Drive, images, CSV, PPTX, ePub | P0: make supported import types and limitations explicit. P1: add visible YouTube transcript, image, CSV, PPTX, and ePub affordances where backend support exists or can be routed through existing ingestion. |
| Google Drive autosync | Not exposed | Drive Docs/Slides/Sheets auto-sync | Skip for this effort. Offer local file import and future connector language instead. |
| Source discovery | Server web search and Deep Research-related import paths exist | Fast Research result review, web/Drive source import, Deep Research import | P0: make web search results importable as workspace sources with status. P1: make Deep Research return/import state obvious inside Research Workspace. |
| Auto labels | Folders and manual organization exist | Auto-label and categorize after 5+ sources | P1: add "suggest labels" or "organize sources" as a generated source-management action, not a background surprise. |
| Grounded chat | Strong RAG controls and diagnostics | Source-scoped chat with citations, simple chat settings, save to note | P0: add beginner-facing chat style/length presets and clearer save-to-note. P1: improve citation hover/source navigation. |
| Retrieval transparency | Chunks, relevance, faithfulness, tokens, cost | Less exposed in NotebookLM | Differentiator: keep, but collapse by default for beginners. |
| Studio outputs | Audio summary, summary, mind map, report, compare, flashcards, quiz, timeline, slides, data table, literature products | Reports, flashcards, quizzes, mind maps, data tables, infographics, slide decks and revisions, audio/video overviews | P0: relabel/organize outputs so NotebookLM-equivalent artifacts are obvious. P1: add cheap video overview storyboard/script and infographic artifact before full media generation. |
| Literature workflows | Literature Matrix, Gap Finder, Evidence-Bound Hypotheses, Proposal Pack | Not a core NotebookLM consumer feature | Differentiator: keep and surface as "Research-grade templates." |
| Notes | Quick notes, save/download/preview | Noteboard, save chat responses preserving citations | P0: make save-to-note from chat/artifact explicit and preserve citation metadata. |
| Sharing | Rich share links, access levels, clone, expiry, revoke | Advanced sharing for paid users | Differentiator after polish. Do not copy pricing-gated language. |
| Ultra agentic chat | ACP task handoff, run history, sandbox diagnostics, lorebook activity export | Agentic chat can search web, run code, create files/charts/images, version artifacts, complete research | P1: unify these as "Workspace Agent Tasks" from chat and Studio. Use visible run activity, approvals, tool calls, files, and provenance. |
| Thinking steps | ACP/run diagnostics can expose activity | NotebookLM exposes "thinking steps" wording | Skip hidden-reasoning parity. Show observable work: plan summary, tool calls, approvals, retrieval, files, and warnings. |
| Browser extension | Research Workspace exists as options/full route; sidepanel has clipper/chat/agent | NotebookLM browser app, not necessarily extension-first | P0: use extension advantage for capture, open-in-workspace, and handoff to chat/agent. Do not build a full sidepanel clone. |
| Limits | Local/self-hosted capacity and provider state | Published plan quotas | Skip quota parity. Expose capability health and local limits. |

## Recommended Work Packages

### WP1: Make NotebookLM-Core Parity Obvious

Goal: A NotebookLM migrant should understand in the first minute how to add
sources, ask grounded questions, scope sources, save useful answers, and
generate common outputs.

Scope:

- Add a concise import capability summary in Add Source that lists supported
  local types, URL behavior, audio/video transcript behavior, and unsupported
  Google-specific sync.
- Normalize source-type labels so URL, YouTube-like URL, local audio/video,
  document, table, image, and pasted text have clear status and limitations.
- Add beginner-facing chat settings for style and answer length, mapped onto
  existing prompt/model behavior.
- Make "save to note" explicit for chat answers and artifacts, including
  citation metadata when available.
- Reorganize Studio output grouping so NotebookLM-equivalent outputs are easy
  to find before advanced literature products.

Acceptance:

- The Research Workspace page can truthfully claim source-scoped chat,
  notes, summaries, audio summaries, mind maps, reports, flashcards, quizzes,
  slides, and data tables are available or capability-gated.
- Unsupported NotebookLM-specific imports are named as unavailable or planned,
  not hidden.
- Beginner controls do not remove advanced RAG diagnostics.

### WP2: Close High-Value Output Gaps Cheaply

Goal: Address the visible Pro/Ultra output gap without committing to expensive
media generation in the first pass.

Scope:

- Add "Video Overview" as a script plus slide/storyboard artifact that can be
  exported or used to generate slides later.
- Add "Infographic" as a structured visual brief or HTML/SVG artifact, with
  source-backed claims and citation metadata.
- Add artifact revision/version labels for slides, infographic briefs, data
  tables, and generated files.
- Add export affordance copy for Markdown, text, CSV/JSON, PPTX where existing
  export paths can support it.

Non-scope:

- Full narrated video rendering.
- Cinematic video generation.
- Arbitrary image generation from NotebookLM-style prompts.
- Pixel-perfect Google artifact templates.

Acceptance:

- Users see honest video and infographic work products that do not overpromise
  rendered media.
- Artifact revisions are visible enough to support Ultra-style expectations.

### WP3: Connect Discovery Loops

Goal: Make source discovery feel like the beginning of a workspace, not a
separate search tool.

Scope:

- Let server web search results be reviewed, selected, imported, and tracked
  as workspace sources.
- Surface Deep Research return/import state inside the workspace, including
  report provenance, selected imported sources, skipped sources, and failures.
- Capability-gate discovery by provider/web-search readiness.
- Add extension handoff actions: capture current page to workspace, open
  workspace, ask chat about captured page, and start agent task with captured
  context.

Non-scope:

- Google Drive search.
- Google account or Workspace integration.
- Full Research Workspace inside the extension sidepanel.

Acceptance:

- A user can go from browser page or web search result to workspace source to
  grounded chat without losing context.
- Discovery import failures are visible and recoverable.

### WP4: Turn Ultra Agentic Chat Into Workspace Agent Tasks

Goal: Match the useful Ultra mental model while staying honest about tldw's
architecture.

Scope:

- Add a chat and Studio entrypoint for "Start workspace task" that launches
  existing ACP/sandbox flows with selected sources and user instructions.
- Show task run activity: plan summary, tool calls, approvals, files produced,
  retrieval used, warnings, and final artifacts.
- Allow generated files or reports to be saved back as workspace artifacts or
  notes.
- Expose version history for agent-produced artifacts where storage already
  supports it.

Non-scope:

- Exposing hidden chain-of-thought.
- Running arbitrary code without sandbox/capability checks.
- Pretending agent actions are safe without approvals.

Acceptance:

- Ultra-style actions are discoverable from chat, but governed by existing ACP,
  sandbox, and capability health.
- The UI explains what happened through observable evidence, not hidden
  reasoning.

## Positioning

The product should not say "NotebookLM clone." The stronger position is:

"A local/self-hosted research workspace with NotebookLM-style source chat and
outputs, plus provider choice, browser capture, transparent retrieval, durable
exports, literature-grade work products, and governed agent tasks."

This framing turns tldw's existing strengths into the reason to use the
workspace:

- Local/self-hosted data control.
- Bring-your-own-model/provider choice.
- Extension capture loop.
- RAG and citation diagnostics.
- Exportable artifacts and notes.
- Literature work products for serious research.
- MCP/ACP/sandbox governance instead of opaque agent behavior.

## Non-Goals

- Do not implement Google Drive autosync for this effort.
- Do not add Google account, Gemini Chat import, or Workspace integration.
- Do not copy NotebookLM quotas or plan-badge language.
- Do not expose hidden chain-of-thought or call it "thinking steps."
- Do not build full video/cinematic generation before a cheaper storyboard
  artifact proves demand.
- Do not build a full Research Workspace sidepanel clone.
- Do not hide unsupported imports behind vague "coming soon" copy.

## Risks And Mitigations

Risk: The comparison creates a sprawling implementation plan.
Mitigation: Treat this document as the broad audit. The first implementation
plan should choose WP1 plus one thin slice of WP2, WP3, or WP4.

Risk: Video and infographic labels overpromise.
Mitigation: Name first-pass outputs as storyboard/script and infographic brief
unless the artifact truly renders media.

Risk: Ultra agentic parity encourages unsafe code execution.
Mitigation: Route through existing ACP, sandbox, capability checks, approvals,
and observable run logs.

Risk: Beginner-facing controls bury power-user diagnostics.
Mitigation: Add presets and clearer grouping, not removal of RAG settings.

Risk: Google documentation changes.
Mitigation: Refresh official docs before implementation copy, tests, or docs
claim current NotebookLM quotas.

## First Implementation Recommendation

Start with a narrow WP1 plan:

1. Import/source expectation copy and visible capability status.
2. Chat style/length presets plus explicit save-to-note.
3. Studio grouping/copy so NotebookLM-equivalent outputs are obvious.
4. A thin extension handoff affordance if existing clipper/open-workspace
   routing can support it without sidepanel rebuild.

Defer full video/infographic rendering, Google Drive parity, quota UI, and
major agent orchestration redesign.

## Verification For This Spec

This spec is complete when:

- The current Research Workspace exposed features are documented.
- NotebookLM Standard, Plus, Pro, Ultra 20 TB, and Ultra 30 TB limits and
  features are summarized from official Google sources.
- Parity gaps, tldw differentiators, skips, risks, and four work packages are
  separated.
- A spec review pass is completed before implementation planning begins.

Spec review result:

- Status: Approved.
- Blocking issues: none.
- Advisory follow-ups: refresh official Google docs again before planning if
  feature names or quotas will be copied into implementation, and keep the
  extension slice conditional on existing capture/handoff routing.
