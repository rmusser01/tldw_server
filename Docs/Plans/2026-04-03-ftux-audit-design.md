# FTUX Audit: tldw_server2 WebUI, Extension & Server

## Context

This is a full audit of the first-time user experience across three surfaces (Next.js webui, browser extension, FastAPI server) for two personas: **parent/family safety user** and **researcher/knowledge worker**. The audit covers UX friction, information architecture, accessibility, error handling, and performance perception.

The goal is to produce a prioritized issues list and a recommended improvement approach.

---

## Comprehensive Issues List

### A. Onboarding & Setup Flow

| # | Issue | Severity | Persona | Location |
|---|-------|----------|---------|----------|
| A1 | **No intent/persona selection during onboarding** - Both parent and researcher see identical setup flow. After connecting, 4 generic buttons (Ingest, Media, Chat, Settings) give no persona-specific guidance. The gap between "connected" and "productive" is wide. | High | Both | `OnboardingConnectForm.tsx` success screen (~line 899+) |
| A2 | **Demo mode doesn't showcase moderation** - Demo mode focuses entirely on chat/ingest. A parent exploring would never discover moderation capabilities exist. | Medium | Family | `OnboardingConnectForm.tsx`, demo mode flow |
| A3 | **Server setup requires CLI knowledge** - `pip install`, `python -m uvicorn`, `.env` file editing, `cp` commands. No installer, no GUI server manager. A non-technical parent cannot self-serve setup. | High | Family | `CLAUDE.md` setup instructions, `Config_Files/` |
| A4 | **Placeholder API key detection is passive** - Server detects placeholders (`your_api_key_here`) but only logs warnings. The setup UI highlights them but doesn't explain which keys are needed for which features. | Medium | Both | `setup_manager.py` (line 31-41), `setup.py` |
| A5 | **Audio bundle provisioning is expert-level** - The setup wizard offers STT/TTS audio bundle installation, but the UI gives no guidance on what bundles are, why you'd want them, or which one to pick. | Low | Both | `/setup/audio/recommendations` endpoint |
| A6 | **No "what you'll need" checklist before setup** - User doesn't know upfront they need: a running server, an API key, optionally ffmpeg, optionally CUDA. They discover requirements as they fail. | Medium | Both | Onboarding flow start |

### B. Navigation & Information Architecture

| # | Issue | Severity | Persona | Location |
|---|-------|----------|---------|----------|
| B1 | **40+ navigation items visible immediately** - Header shortcuts show 8 groups with 40+ items. No progressive disclosure. Cognitive overload for any new user. | High | Both | `header-shortcut-items.ts` (lines 58-411) |
| B2 | **Moderation buried in "Tools" group** - A parent's primary feature is in the 7th of 8 nav groups, alongside "Chunking Playground" and "Chatbooks." | High | Family | `header-shortcut-items.ts` (line 338-360) |
| B3 | **Workspace only accessible via header shortcuts** - "Research Studio" (workspace-playground) is in the "Research" group but not in the sidebar or primary navigation. Easy to miss entirely. | Medium | Researcher | `header-shortcut-items.ts` (line 134-139) |
| B4 | **Jargon-heavy labels with no descriptions** - "RAG", "STT", "TTS", "ACP Playground", "Chunking Playground", "Chatbooks", "World Books" are meaningless to non-technical users. No tooltips or plain-language subtitles. | High | Family | `header-shortcut-items.ts` labels |
| B5 | **"Moderation Playground" name is confusing** - "Playground" implies experimental/toy. "Moderation" is ambiguous. A parent looking for parental controls wouldn't click this. "Family Safety" or "Content Controls" would be clearer. | Medium | Family | `header-shortcut-items.ts` line 354-358 |
| B6 | **Family Guardrails Wizard hidden in Settings** - The production-ready 8-step wizard at `/settings/family-guardrails` (with household setup, guardian/dependent management, safety templates, invite tracking) is only discoverable through the Settings nav, marked as beta. No link from moderation onboarding or setup flow. | High | Family | `settings-nav-config.ts` (line 208-214), `FamilyGuardrailsWizard.tsx` |
| B7 | **No relationship shown between related features** - Moderation, Family Guardrails, and Guardian are three separate nav items with identical icons (ShieldCheck) and no explanation of how they relate. | Medium | Both | `settings-nav-config.ts` (lines 201-222) |
| B8 | **Header shortcut groups don't match user mental models** - "Chat & Persona" and "Research" are developer categories. A researcher thinks "I want to search my papers" not "I need Knowledge QA in the Research group." | Medium | Researcher | `header-shortcut-items.ts` group structure |

### C. Chat Page (/chat) FTUX

| # | Issue | Severity | Persona | Location |
|---|-------|----------|---------|----------|
| C1 | **5 starter modes may overwhelm first-time users** - General, Compare, Character, Knowledge Q&A, Deep Research are all shown equally. No indication of which is simplest or recommended for beginners. | Medium | Both | `PlaygroundEmpty.tsx` (lines 79-133) |
| C2 | **"Compare models" starter assumes multi-model setup** - A first-time user likely has one provider configured. Clicking "Compare models" will fail or confuse if only one model is available. | Low | Both | `PlaygroundEmpty.tsx` (line 91-99) |
| C3 | **Layout guide text is abstract** - "History (left), timeline (center), composer (bottom), Search & Context (right)" means nothing before using the page. Regions aren't labeled in the actual UI. | Low | Both | `PlaygroundEmpty.tsx` (lines 188-221) |
| C4 | **Disconnected state message is passive** - "Connect to a tldw server to start chatting. Go to Settings to configure your connection." No button to navigate directly; user must find Settings themselves. | Medium | Both | `PlaygroundEmpty.tsx` (line 148-152) |
| C5 | **"Quick Ingest" secondary CTA assumes knowledge of ingestion** - A parent doesn't know what "ingest" means. A researcher might expect "upload" or "import." | Low | Family | `PlaygroundEmpty.tsx` (line 162) |
| C6 | **"Take a quick tour" link is easy to miss** - Placed at the very bottom, small text, no visual prominence. The tour is actually useful but hard to discover. | Low | Both | `PlaygroundEmpty.tsx` (lines 223-234) |

### D. Moderation Page (/moderation-playground) FTUX

| # | Issue | Severity | Persona | Location |
|---|-------|----------|---------|----------|
| D1 | **Onboarding is a single dismissible banner** - "Welcome to Moderation Playground. Configure content safety rules, test them live, and manage per-user overrides." + "Got it, let's start." No guided walkthrough, no link to Family Guardrails Wizard (which is a full 8-step production wizard with household setup, templates like "default-child-safe"/"teen-balanced"/"school-research", and invite tracking). | High | Family | `ModerationPlaygroundShell.tsx` (lines 249-263) |
| D2 | **5 tabs shown immediately with no guidance on order** - Policy & Settings, Blocklist Studio, User Overrides, Test Sandbox, Advanced. A parent doesn't know where to start. No "recommended first step" indicator. | High | Family | `ModerationPlaygroundShell.tsx` (lines 23-29) |
| D3 | **"Advanced" tab visible to all users** - Power-user configuration shown alongside basic controls. Increases perceived complexity. | Low | Family | `ModerationPlaygroundShell.tsx` (line 28) |
| D4 | **No tutorial/Joyride for moderation** - Workspace has auto-start Joyride (via `startTutorial("workspace-playground-basics")` and tutorial registry), Chat has "Take a tour" link. Moderation has neither. The tutorial system is straightforward to extend -- just needs a definition file, i18n strings, and `data-testid` targets. | Medium | Family | Missing from `tutorials/definitions/` |
| D5 | **Permission error message is technical** - "Moderation controls require an admin account with SYSTEM_CONFIGURE permission." A non-technical user doesn't know what SYSTEM_CONFIGURE means or how to get it. | Medium | Family | `ModerationPlaygroundShell.tsx` (lines 238-245) |
| D6 | **Offline warnings chain multiple conditions without clear resolution** - 5 different offline states (auth error, unconfigured, unreachable, generic offline, permission error) each show different warnings. The hierarchy of what to fix first isn't clear. | Medium | Both | `ModerationPlaygroundShell.tsx` (lines 176-231) |
| D7 | **Ctrl+S save shortcut not discoverable** - No tooltip, no keyboard shortcut hint in the UI. User must guess or read code. | Low | Both | `ModerationPlaygroundShell.tsx` (lines 78-96) |
| D8 | **Hero section takes significant vertical space** - The gradient hero with "Moderation Playground" title and server status badge pushes actual controls below the fold on smaller screens. | Low | Both | `ModerationPlaygroundShell.tsx` (lines 266-295) |

### E. Workspace Page (/workspace-playground) FTUX

| # | Issue | Severity | Persona | Location |
|---|-------|----------|---------|----------|
| E1 | **Auto-starts Joyride without user consent** - First visit immediately launches guided tour. If user dismisses accidentally, localStorage flag is set and tour never shows again. No way to re-trigger from the UI. | Medium | Both | `WorkspacePlayground/index.tsx` (lines 1666-1687) |
| E2 | **Complex multi-pane layout with no preview** - Three panes (Sources, Chat, Studio) with resizable dividers. No screenshot or diagram showing what this page does before entering it. | Medium | Researcher | `WorkspacePlayground/index.tsx` |
| E3 | **Keyboard shortcuts not discoverable** - Cmd+1/2/3 for panes, Cmd+K search, Cmd+Z undo, Cmd+Shift+N new workspace. No shortcut legend visible in the UI. | Low | Both | `WorkspacePlayground/index.tsx` (lines 1689-1780) |
| E4 | **"Research Studio" label doesn't match page content** - Nav calls it "Research Studio" but the page itself is about workspaces, sources, notes, and chat. The name doesn't communicate the value. | Low | Researcher | `header-shortcut-items.ts` (line 138) vs page content |
| E5 | **Storage quota handling is invisible until failure** - Workspace has a 5MB localStorage budget with complex shard/split logic. Users get no warning as they approach the limit. | Low | Researcher | `WorkspacePlayground/index.tsx` (lines 82-88, 264-327) |

### F. Extension FTUX

| # | Issue | Severity | Persona | Location |
|---|-------|----------|---------|----------|
| F1 | **Auto-probe only checks localhost:8000** - If server is on a different port or host, user must manually enter URL. No explanation of where to find the server URL. | Medium | Both | `OnboardingConnectForm.tsx`, auto-probe logic |
| F2 | **Sidepanel step counter is text-only** - "Step 1 of 3" with no visual progress bar. However, the `EmptySidePanel` component is otherwise excellent (contextual headings per error type, specific remediation instructions, suggestion cards when connected). Low priority. | Low | Both | Extension sidepanel `EmptySidePanel` (`empty.tsx` lines 198-394) |
| F3 | **CORS error message requires developer knowledge** - "Browser can't reach server due to security" with fix: "Add ALLOWED_ORIGINS in .env." A non-technical user cannot act on this. | Medium | Family | `validation.ts` error categories |
| F4 | **No indication the extension exists** - The webui doesn't mention or link to the browser extension. A user who set up via webui might never discover it. | Medium | Both | No cross-promotion in webui |
| F5 | **Magic link auth only in onboarding form** - If a multi-user user misses it during setup, there's no way to use magic link login later from extended options. | Low | Both | Agent exploration finding |

### G. Cross-Cutting Issues

| # | Issue | Severity | Persona | Location |
|---|-------|----------|---------|----------|
| G1 | **No unified "getting started" guide in the UI** - Documentation exists in `/Docs/` and various READMEs but no in-app help center or getting-started flow that connects setup to first productive use. Note: `CompanionHomeShell` already has quick actions (Chat, Knowledge, Analysis cards), LLM provider alerts, and per-card empty states with setup prompts -- so the infrastructure exists but isn't leveraged for a cohesive first-run path. | High | Both | `CompanionHomePage.tsx`, across all surfaces |
| G2 | **Inconsistent empty states** - Chat uses `FeatureEmptyState` component (icon + title + description + 2 actions). Moderation uses inline yellow/blue banners. Workspace uses Joyride. Note: `FeatureEmptyState` only supports 1 error state at a time with 1 icon and 2 actions, so moderation's 5 simultaneous error conditions can't be directly replaced with it -- would need a wrapper or prioritized single-error display. | Medium | Both | Various page components, `FeatureEmptyState.tsx` |
| G3 | **Tutorial infrastructure under-leveraged** - Joyride tutorial system exists with `startTutorial()`, a `TUTORIAL_REGISTRY` array, and `getTutorialsForRoute()` matching. Only workspace auto-triggers it. Chat has a manual "tour" link. Moderation, settings, and most other pages have no tutorials. Adding a new tutorial is straightforward: definition file + i18n strings + `data-testid` targets + registry import. | Medium | Both | `tutorials/` directory, `store/tutorials.ts`, `tutorials/registry.ts` |
| G4 | **No "what's new" or changelog in UI** - Active development means features change frequently. Returning users get no notification of new capabilities. | Low | Both | Not implemented |
| G5 | **Performance perception: lazy-load "Loading..." text** - Multiple panels show bare "Loading..." text during Suspense fallback. No skeleton loader or spinner, making the app feel broken during load. | Low | Both | `ModerationPlaygroundShell.tsx` (lines 140-172), various |
| G6 | **Three safety-related nav items with same icon** - Moderation Playground, Family Guardrails, and Guardian all use `ShieldCheck` icon. Visually indistinguishable in nav. | Low | Family | `settings-nav-config.ts` |
| G7 | **Onboarding test coverage is minimal** - Only 2-3 guard tests exist for `OnboardingConnectForm` (ingest CTA routing, error classification). No tests for the full connection flow, auth mode switching, or success screen. Any refactoring needs guard tests first. | Medium | Both | `OnboardingConnectForm/__tests__/` |

---

## Recommended Approach: A then B (Targeted Fixes, then Persona-Routed Onboarding)

### Phase 1: Targeted Friction Fixes (2-3 weeks)

These changes remove the worst friction without architectural changes:

**1. Add intent selector on onboarding success screen** (addresses A1)
   - After connection succeeds, show 2-3 cards: "Chat with AI" / "Set up family safety" / "Research my documents"
   - "Family safety" card routes directly to `/settings/family-guardrails` (the production 8-step wizard)
   - "Research" card routes to Quick Ingest, then `/chat`
   - "Chat" card routes to `/chat` (current default)
   - This leverages existing `CompanionHomeShell` quick actions rather than duplicating guidance
   - File: `OnboardingConnectForm.tsx` success screen

**2. Elevate moderation + family guardrails in navigation** (addresses B2, B5, B6, B7, G6)
   - Create a new "Safety" nav group containing: Family Guardrails, Content Controls (renamed from "Moderation Playground"), Guardian
   - Give each a distinct icon (e.g., Users for Guardrails, ShieldCheck for Content Controls, Eye for Guardian)
   - Remove beta tag from Family Guardrails -- it's a production-ready 8-step wizard
   - File: `header-shortcut-items.ts` (type: `HeaderShortcutItem` already supports all needed fields)
   - File: `settings-nav-config.ts` (for settings sidebar grouping)

**3. Improve moderation onboarding** (addresses D1, D2)
   - Replace dismissible banner with a structured onboarding card that:
     - Links to Family Guardrails Wizard as primary CTA: "New here? Set up family profiles first"
     - Shows recommended tab order with visual indicators
     - Persists until user completes at least one action (not just clicks "dismiss")
   - Note: Don't try to replace inline error banners with `FeatureEmptyState` -- moderation's multi-error display is more nuanced. Instead, add priority ordering to the existing banner chain (show most actionable error first, collapse others)
   - File: `ModerationPlaygroundShell.tsx`

**4. Add plain-language descriptions to nav items** (addresses B4)
   - Add `descriptionKey` + `descriptionDefault` optional fields to `HeaderShortcutItem` type
   - Add descriptions to the ~15 most jargon-heavy items (not all 40+)
   - Update renderer in `HeaderShortcuts.tsx` to display subtitle text below label
   - Key translations: "RAG" -> "Search & Retrieve", "STT" -> "Speech to Text", "TTS" -> "Text to Speech", "ACP" -> "Agent Protocol", "Chunking" -> "Document Splitting"
   - Scope: type definition (1 file) + data entries (~15 items) + renderer (1 file) = 3 levels of change
   - File: `header-shortcut-items.ts`, `HeaderShortcuts.tsx`

**5. Fix disconnected state on Chat page** (addresses C4)
   - Add a direct "Open Settings" button to the disconnected description in `PlaygroundEmpty`
   - Use `useNavigate()` (already imported) to route to `/settings/tldw`
   - File: `PlaygroundEmpty.tsx` (lines 148-152)

**6. Add Joyride tutorial for moderation** (addresses D4, G3)
   - Create `tutorials/definitions/moderation.ts` with 4-5 steps targeting: hero/status, Policy tab, Blocklist tab, Test Sandbox, and the "Family Guardrails" link
   - Add i18n strings to `assets/locale/en/tutorials.json`
   - Import and register in `tutorials/registry.ts`
   - Auto-trigger on first visit from `ModerationPlaygroundShell`
   - Add `data-testid` attributes to target elements
   - File: new `tutorials/definitions/moderation.ts`, `tutorials/registry.ts`, `ModerationPlaygroundShell.tsx`

**7. Replace "Loading..." with skeleton loaders** (addresses G5)
   - Use Ant Design `Skeleton` component (already a project dependency) in Suspense fallbacks
   - Apply to moderation tab panels and other high-visibility lazy-loaded panels
   - File: `ModerationPlaygroundShell.tsx` panel fallbacks

**8. Add regression tests before refactoring onboarding** (addresses G7)
   - Before touching `OnboardingConnectForm`, write guard tests for: single-user auth flow, multi-user auth flow, connection progress states, success screen rendering
   - Follow pattern from `WorkspacePlayground.stage1.onboarding.test.tsx`
   - File: new tests in `OnboardingConnectForm/__tests__/`

### Phase 2: Persona-Routed Onboarding (3-4 weeks, after Phase 1)

These changes introduce persona awareness to reduce cognitive load:

**1. Persona selection during onboarding** (addresses A1, B1)
   - Card grid before server URL: "Parent/Family", "Researcher", "Power User"
   - Store in connection store as `userPersona: "family" | "researcher" | "explorer" | null`
   - This does NOT gate features -- it changes defaults and highlighting
   - File: `OnboardingConnectForm.tsx`, `store/connection.ts`

**2. Persona-aware navigation defaults** (addresses B1, B8)
   - The `HEADER_SHORTCUT_SELECTION_SETTING` in `ui-settings.ts` already controls which shortcuts are visible
   - Family: default selection ~6 items (Chat, Media, Content Controls, Family Guardrails, Settings, Help)
   - Researcher: default selection ~9 items (Chat, Media, Knowledge, Workspace, Deep Research, Collections, Notes, Settings)
   - Explorer: all items visible (current behavior)
   - Add "Show all features" link at bottom of filtered nav
   - File: `header-shortcut-items.ts`, `HeaderShortcuts.tsx`, `ui-settings.ts`

**3. Persona-specific post-connection guided flow** (addresses A1, G1)
   - Family: success screen shows "Step 1: Set up family profiles" -> links to Family Guardrails Wizard. "Step 2: Review content rules" -> links to Content Controls. "Step 3: Test with a sample message" -> links to Test Sandbox tab
   - Researcher: success screen shows "Step 1: Import your first document" -> Quick Ingest. "Step 2: Find it in Media" -> Media. "Step 3: Ask about it" -> Chat with Knowledge Q&A starter
   - Explorer: current success screen (4 buttons) unchanged
   - Leverage existing `CompanionHomeShell` quick actions for the "after onboarding" experience
   - File: `OnboardingConnectForm.tsx` success screen

**4. Cross-promote extension in webui** (addresses F4)
   - Add "Get the browser extension" card in post-setup flow or Settings/About page
   - File: Settings page or onboarding success screen

---

## Out of Scope (noted but not addressed)

- **A3: Server setup requiring CLI** - This is a deployment/packaging problem, not a UI problem. Would need an installer or Docker Compose one-click solution. Separate initiative.
- **E5: Storage quota warnings** - Low severity, complex implementation. Track as a future enhancement.
- **Full Approach C (Mission Control home, progressive unlock)** - High effort, high risk. The groundwork from Phase 1+2 (persona in store, filtered nav) makes this easier to build later if desired.

---

## Verification Plan

After implementing each phase:

1. **Manual walkthrough** as each persona:
   - Fresh browser (cleared localStorage), no server running -> verify error guidance is clear and actionable
   - Fresh browser, server running -> walk through onboarding to first productive action
   - Count clicks from "first visit" to "first useful result" for each persona
   - Verify moderation is discoverable within 2 clicks from home page

2. **Test coverage**:
   - Existing tests in `WorkspacePlayground.stage1.onboarding.test.tsx` show the pattern
   - Add similar tests for: moderation onboarding, intent selector cards, tutorial auto-start, persona-filtered nav
   - Guard tests for `OnboardingConnectForm` BEFORE refactoring (Phase 1, item 8)

3. **Accessibility check**:
   - Tab navigation through onboarding flow and intent selector
   - Screen reader announces intent cards and progress states
   - Color contrast on new onboarding elements and nav descriptions
   - Keyboard navigation through moderation tutorial steps

---

## Key Files to Modify

| File | Phase | Changes |
|------|-------|---------|
| `apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx` | 1+2 | Intent selector on success screen, persona selection, guided post-connect flow |
| `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts` | 1+2 | New "Safety" group, regroup moderation, add description fields, persona filtering |
| `apps/packages/ui/src/components/Layouts/HeaderShortcuts.tsx` | 1+2 | Render description subtitles, persona filter UI, "Show all" link |
| `apps/packages/ui/src/components/Option/ModerationPlayground/ModerationPlaygroundShell.tsx` | 1 | Replace onboarding banner with structured card, add tutorial trigger, skeleton loaders, error priority |
| `apps/packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx` | 1 | Add "Open Settings" button for disconnected state |
| `apps/packages/ui/src/components/Layouts/settings-nav-config.ts` | 1 | New "Safety" group, remove beta flag from Family Guardrails, distinct icons |
| `apps/packages/ui/src/tutorials/definitions/moderation.ts` (new) | 1 | Moderation Joyride tutorial definition |
| `apps/packages/ui/src/tutorials/registry.ts` | 1 | Register moderation tutorial |
| `apps/packages/ui/src/store/connection.ts` | 2 | Add `userPersona` field |
| `apps/packages/ui/src/services/settings/ui-settings.ts` | 2 | Persona-aware shortcut defaults |
| `apps/packages/ui/src/components/Option/Onboarding/__tests__/` (new) | 1 | Guard tests before refactoring |
