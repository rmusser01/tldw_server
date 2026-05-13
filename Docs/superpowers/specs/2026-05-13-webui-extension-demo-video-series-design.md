# WebUI And Extension Demo Video Series Design

Date: 2026-05-13
Owner: Codex collaboration session
Status: Draft for user review
Backlog: TASK-320

## Summary

This design defines a repeatable demo-video campaign for the tldw WebUI and
browser extension. The campaign should market one product through several
audience-specific lenses without fragmenting the product story.

The approved production model is:

- Record one dense 20-30 minute full product walkthrough using the real local
  app, real WebUI, real extension, and seeded non-sensitive demo data.
- Structure that walkthrough into reusable chapters.
- Cut the master footage into persona-based marketing videos.
- Cut shorter clips from both the master walkthrough and persona videos for
  README, website, social, release, and documentation surfaces.

The intended long-form style is practical and feature-forward: closer to a
hands-on feature showcase than a cinematic brand ad. The walkthrough should
prove breadth and credibility. The persona cuts should translate that breadth
into "this is for me" stories.

Detailed feature inventory, exact scripts, shot lists, recording runbooks, and
asset checklists are deliberately deferred to the next planning phase.

## Goals

1. Create a reusable video campaign architecture for demonstrating the WebUI
   and browser extension.
2. Support broad public advertising first, individual self-hosters second, and
   team or organizational buyers third.
3. Show the real product instead of relying on mockups or disconnected
   marketing animation.
4. Preserve a single product narrative while allowing different audiences to
   see different entry points.
5. Make future re-recording practical as the product changes.
6. Keep this design scoped to campaign structure and production approach, not
   script-level feature coverage.

## Non-Goals

- Do not enumerate every feature that must appear in the final scripts.
- Do not write narration scripts in this design.
- Do not create recording automation, demo data, seed scripts, or screenshots
  in this design.
- Do not add WebUI or extension demo-mode behavior as part of this design.
- Do not choose final publishing copy, thumbnails, or channel-specific metadata
  in this design.

## Audience Strategy

The same product should be marketed through different audience frames.

Priority order:

1. Broad public audience.
2. Individual self-hosters and privacy-conscious users.
3. Teams and organizations.

The broad public tone should combine:

- Practical value: save web pages, videos, notes, and documents, then ask useful
  questions over them.
- Technical credibility: self-hostable WebUI and browser extension backed by a
  local or configured server.
- Privacy awareness: users control their data and configured providers.

The campaign should avoid generic "AI changes everything" claims. It should
lead with visible product proof.

## Campaign Architecture

The video set has one master asset and several derived assets.

### Master Asset

The master asset is a 20-30 minute full product walkthrough.

It should:

- Use a real local server.
- Show the WebUI and extension connected to the same server.
- Use seeded, public, redistributable demo sources.
- Be chaptered for YouTube and reuse.
- Cover setup and configuration only to the depth needed to build trust.
- Trim or time-compress slow media and model-processing steps.
- Keep narration tied to what is visible on screen.

This is the durable proof asset for technical viewers and curious evaluators.

### Derived Assets

The master footage should feed:

- Self-hoster and privacy professional persona cut.
- Student and academic researcher persona cut.
- Journalist, analyst, and OSINT researcher persona cut.
- Team knowledge manager and internal research ops persona cut.
- Writer, note-taker, and knowledge worker persona cut.
- Short clips for social posts, README embeds, landing pages, release notes, and
  docs.

Each derived asset should link back to the full walkthrough where appropriate.

## Full Walkthrough Chapter Model

The master walkthrough should use chapter-level structure. Full feature
coverage will be decided during script planning.

### 1. What tldw Is

Introduce tldw as a private or self-hostable AI workspace for capturing,
researching, transforming, and organizing personal knowledge.

Show the relationship between:

- Local or configured server.
- WebUI.
- Browser extension.
- User-controlled sources and outputs.

### 2. Setup And Connection

Show enough setup to make the product trustworthy:

- Local server is running.
- WebUI is connected.
- Extension is connected to the same server.
- Model or provider selection exists.

This should not become a full installation tutorial. Detailed install videos can
be separate follow-up assets.

### 3. Browser Capture Loop

Demonstrate the extension as the in-browser entry point:

- Open a web page.
- Use the sidepanel.
- Clip or ingest page content.
- Show that the content becomes available in the WebUI knowledge or media
  workspace.

This is the strongest bridge between the extension and WebUI story.

### 4. Knowledge QA

Demonstrate asking questions over captured material:

- Ask a question over selected or searchable sources.
- Show source-grounded answers and citations when available.
- Show enough retrieval/search controls to prove the workflow is inspectable.

The goal is to show that the assistant works over the user's own archive rather
than behaving like a generic chatbot.

### 5. Media Workflow

Show a media-oriented workflow:

- Ingest or open video, audio, or an existing processed media item.
- Show transcript, summary, and chat over the media.
- Trim waiting time or use preprocessed seeded data for long-running steps.

### 6. Workspace Output Generation

Show that tldw turns sources into useful work products:

- Notes.
- Summaries.
- Flashcards or quizzes.
- Study or work artifacts.
- Other output types selected during script planning.

This section should communicate that the workflow does not end at search or
chat. It produces reusable work.

### 7. Power-User Workspace

Show breadth without lingering on every setting:

- Notes.
- Prompts.
- Characters or personas.
- Reusable workflows.
- Model and provider settings.

This chapter should feel like a fast tour of depth for people who want to grow
into the product.

### 8. Privacy, Self-Hosting, And Admin Proof

Make the trust claims concrete:

- No telemetry positioning.
- Local or configured model providers.
- User-owned server and data.
- Enough admin or configuration visibility to support the claim.

Avoid overpromising security posture. Show what the current product actually
does and where users control their own deployment.

### 9. Closing Recap

End with the product promise:

- Capture sources.
- Ask your archive.
- Transform sources into useful work.
- Keep control of your data.

Point viewers to the persona cuts, setup docs, or project repository depending
on the publication surface.

## Persona Series Structure

Each persona video should be a focused edit from the master walkthrough with a
short custom opening, targeted bridge narration, and a concise recap.

Recommended format:

1. Problem open, 10-20 seconds.
2. Workflow proof, 2-5 minutes.
3. Capability recap, 20-40 seconds.

Persona cuts should not imply separate products. They are different doors into
the same product.

### Self-Hoster And Privacy Professional

Primary promise:

Use AI over personal or professional knowledge while retaining control over the
server, data, and configured model providers.

Likely proof points:

- Local server and WebUI connection.
- Extension connected to the user's server.
- Model or provider flexibility.
- No-telemetry and self-hosting posture.
- User-owned data and admin/config surfaces.

### Student And Academic Researcher

Primary promise:

Turn reading, papers, notes, videos, and documents into searchable, cited, and
study-ready knowledge.

Likely proof points:

- Capture academic or learning sources.
- Ask source-grounded questions.
- Generate notes, summaries, flashcards, or quizzes.
- Reuse captured material across study and writing workflows.

### Journalist, Analyst, And OSINT Researcher

Primary promise:

Capture, compare, and interrogate source material while preserving enough
source context to stay grounded.

Likely proof points:

- Browser capture from multiple sources.
- Source tracking and citations.
- Media transcript and summary workflows.
- Cross-source questioning and comparison.
- Notes or outputs that preserve provenance.

### Team Knowledge Manager And Internal Research Ops

Primary promise:

Convert scattered internal or research material into repeatable knowledge
workflows and reusable outputs.

Likely proof points:

- Organized knowledge workspace.
- Repeatable capture and review loop.
- Shared or team-oriented workflows where currently supported.
- Output generation for internal briefs, summaries, or study material.
- Admin/config proof for deployment confidence.

### Writer, Note-Taker, And Knowledge Worker

Primary promise:

Move from collected sources to usable writing, notes, prompts, and reusable
knowledge structures.

Likely proof points:

- Notes workspace.
- Prompt library.
- Characters or personas where useful.
- Source-backed drafting or synthesis.
- Reusable workspaces and generated artifacts.

## Demo Environment Requirements

The campaign should use a real local environment with seeded data.

The seed set should be curated before recording and should include:

- At least one web article or documentation page.
- At least one video or audio item, ideally preprocessed before the final take.
- At least one PDF or document.
- Example notes.
- Example prompts.
- Example flashcards or quizzes.
- Example source-backed chat history if useful.

All demo data must be public, non-sensitive, and safe to redistribute in
screenshots or recordings. Do not use private API keys, personal notes, private
browser history, confidential documents, private chats, or unreleasable media.

The environment should use a clean browser profile with:

- No personal bookmarks visible.
- No personal accounts visible unless intentionally part of the demo.
- Notifications disabled.
- Readable zoom and font size.
- Enlarged or high-visibility cursor.
- Stable viewport and recording resolution.

## Production Workflow

### 1. Prepare Demo Data

Choose sources that demonstrate breadth without adding legal or privacy risk.
Preprocess slow media items before recording when needed.

### 2. Build A Recording Runbook

Each walkthrough chapter should have a checklist containing:

- Starting URL or route.
- Required server state.
- Required account or auth state.
- Source item to select.
- Query or action to perform.
- Expected visible result.
- Fallback if model or media processing is slow.

The runbook is a follow-up artifact, not part of this design.

### 3. Record Master Footage

Record each chapter as a separate take, even if the final video is edited as one
continuous walkthrough. This makes re-recording and persona cuts easier.

Recommended capture posture:

- 1440p or 4K source capture, exported at 1080p or higher.
- Consistent browser zoom.
- Clean desktop and browser chrome.
- No private data in frame.
- Narration can be recorded live or added after editing, but the final script
  should stay synchronized with what is visible.

### 4. Edit The Full Walkthrough

The master edit should:

- Keep waiting time short.
- Use chapter labels.
- Use light zooms or callouts only when they clarify product behavior.
- Avoid hiding the UI behind heavy motion graphics.
- Show enough configuration to build trust but not so much that the video turns
  into documentation.

### 5. Cut Persona Videos

Persona videos should reuse master footage as much as possible. Add custom
openings, bridge narration, and closing calls to action for each audience.

### 6. Cut Short Clips

Short clips should be derived after the persona videos so they inherit the same
claims and visual proof.

Likely clip types:

- Web page to knowledge question.
- Video to transcript to summary.
- Source-backed answer with citations.
- Flashcards or quizzes from sources.
- Private/self-hosted setup proof.
- Extension sidepanel to WebUI handoff.

### 7. Publish And Maintain

Publish the campaign as a set:

- Full walkthrough with chapters.
- Persona playlist.
- Short clip library.
- README or docs embeds where useful.
- Release-post embeds for major product updates.

Treat demo videos like docs. Refresh the master walkthrough when major UI,
extension, setup, or workflow changes make the old footage misleading.

## Follow-Up Artifacts

The next phase should create a dedicated demo-video documentation area, likely:

- `Docs/Product/DemoVideos/README.md`
- `Docs/Product/DemoVideos/full-walkthrough-runbook.md`
- `Docs/Product/DemoVideos/persona-self-hoster-privacy.md`
- `Docs/Product/DemoVideos/persona-student-researcher.md`
- `Docs/Product/DemoVideos/persona-journalist-osint.md`
- `Docs/Product/DemoVideos/persona-team-knowledge.md`
- `Docs/Product/DemoVideos/persona-writer-knowledge-worker.md`
- `Docs/Product/DemoVideos/asset-checklist.md`

These should be created during script and runbook planning, after full feature
coverage is enumerated.

## Risks And Mitigations

### Risk: One Walkthrough Becomes A Feature Dump

Mitigation:

Keep the master walkthrough chaptered and workflow-based. Full feature coverage
can be handled in script planning, but each chapter should still have one clear
viewer promise.

### Risk: Persona Cuts Feel Like Separate Products

Mitigation:

Use consistent product language, the same visual environment, and links back to
the full walkthrough. Persona videos should change the framing, not the product
truth.

### Risk: Real-App Recording Is Flaky

Mitigation:

Use seeded data, preprocessed media, chapter-level takes, and fallback states.
Do not mock the core product, but avoid relying on long-running processing in a
live take when the output can be prepared beforehand.

### Risk: Marketing Claims Drift From Product Reality

Mitigation:

Tie scripts to visible UI evidence. Re-record or retire clips when routes,
capabilities, or workflows materially change.

### Risk: Demo Data Leaks Sensitive Information

Mitigation:

Use only public, redistributable, non-sensitive sources. Keep the browser
profile and local server state dedicated to demo recording.

## Verification For This Design

This design is complete when:

- The campaign architecture is documented.
- The full walkthrough chapter model is documented.
- The persona series structure and persona mappings are documented.
- The real-app production workflow is documented.
- Follow-up runbook and script artifacts are identified without being
  implemented here.
