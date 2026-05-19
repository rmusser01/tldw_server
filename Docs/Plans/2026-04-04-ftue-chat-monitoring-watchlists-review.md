# FTUE Review: /chat, /monitoring, /watchlists

**Date:** 2026-04-04
**Scope:** First-time user experience for /chat (WebUI PlaygroundChat + extension sidepanel), /admin/monitoring, and /watchlists
**Personas:** Competent non-technical user (A), Competent technical user (B)
**Builds on:** Existing v1 audit (33 issues), v2 audit (13 issues), ingest/knowledge/workspace/extension audit (60 issues), blocker plan (4 fixes)

---

## Category: Chat — WebUI PlaygroundChat (CHAT)

| ID | P | Title | Persona | Fix Direction |
|----|---|-------|---------|---------------|
| CHAT-001 | P1 | NoProviderBanner says "Open Settings" but gives no guidance on WHAT to do there — no link to API key signup, no explanation of "LLM provider" concept | A | Add plain-language explanation: "You need an API key from a service like OpenAI or Anthropic. [Get an OpenAI key](link) / [Get an Anthropic key](link). Then paste it in Settings." Link to `/settings/tldw` (blocker fix pending) |
| CHAT-002 | P1 | Model selector appears after setup but gives no guidance on model differences — no speed/quality/cost indicators, no "recommended for beginners" tag | A | Add capability badges (vision, fast, reasoning) and a "(Recommended)" tag on a sensible default. Show brief tooltip: "GPT-4o: versatile, good for most tasks" |
| CHAT-003 | P2 | PlaygroundEmpty welcome shows 5 starter cards (General chat, Compare models, Character chat, Knowledge Q&A, Deep Research) but labels assume prior knowledge — "Knowledge-grounded Q&A" is jargon | A | Rename to plain language: "Search your documents", "Chat with a character", "Compare AI models side-by-side", "Deep research" |
| CHAT-004 | P2 | "Page regions" guide at bottom of PlaygroundEmpty explains left/center/bottom/right layout but uses developer terminology ("composer", "Search & Context panel") | A | Use plain labels: "Chat history (left)", "Conversation (center)", "Message input (bottom)", "Sources & tools (right)" |
| CHAT-005 | P2 | System prompt feature is completely undiscoverable on first visit — no default prompt, no hint that prompts exist until user opens tutorial | Both | Show brief hint near model selector: "Tip: Set a system prompt to customize how the AI responds" with link to prompt selector |
| CHAT-006 | P2 | Slash commands only discoverable by typing "/" or via tutorial — no hint in the input placeholder beyond "(/ for commands)" which is easily missed | A | On first message focus, show a brief tooltip: "Type / for commands like /search or /web" that dismisses after first use |
| CHAT-007 | P2 | No conversation export in WebUI PlaygroundChat — sidepanel has JSON/Markdown export but playground view does not | B | Add export option to playground chat header or message context menu |
| CHAT-008 | P2 | API error messages are generic — rate limits, auth failures, model-not-found all show as toast notifications without actionable guidance | Both | Categorize errors: rate limit -> "This model has a rate limit. Wait a moment or switch models." Auth -> "Your API key may have expired. Check Settings." Model not found -> "This model is no longer available. Try another." |
| CHAT-009 | P2 | Character/persona selection is hidden — no mention in empty state, no visual cue that character cards exist | A | Add "Chat as a character" card in PlaygroundEmpty that links to character selector, or show a small "Persona" toggle in chat header |
| CHAT-010 | P3 | Keyboard shortcuts not discoverable — no help text, no shortcut reference in empty state or settings | B | Add "Keyboard shortcuts" link in PlaygroundEmpty or show "Press ? for shortcuts" hint |
| CHAT-011 | P3 | Demo mode exists but not offered prominently on first visit — user must find it in settings | A | In NoProviderBanner, add secondary CTA: "Or try demo mode to explore without an API key" |
| CHAT-012 | P3 | Compare mode (side-by-side model comparison) is not discoverable until explicitly enabled — no mention in empty state | B | Add brief mention in PlaygroundEmpty: "You can also compare models side-by-side" with link to enable |
| CHAT-013 | P2 | "Take a quick tour" button in PlaygroundEmpty is easy to miss — positioned at bottom, no visual emphasis | Both | Move tour CTA higher or make it a dismissible banner on first visit |

## Category: Chat — Extension Sidepanel (SIDE)

| ID | P | Title | Persona | Fix Direction |
|----|---|-------|---------|---------------|
| SIDE-001 | P1 | ConnectionBanner "Finish setup" button opens Options page but user must navigate to the correct settings tab — no deep link to the setup form | A | Deep link to `/options.html#/setup` or `/options.html#/settings/tldw` directly |
| SIDE-002 | P1 | Auth error state shows inline API key form (good) but only for single-user mode — multi-user mode just says "Fix the key in Settings" with no inline login form | A | Add inline username/password form for multi-user mode, or at minimum deep-link to the auth page |
| SIDE-003 | P2 | Three example prompts ("Summarize this page", "What are the key points?", "Explain this in simple terms") are all page-analysis prompts — no variety for general chat, knowledge search, or other capabilities | Both | Add variety: keep 1 page prompt, add 1 general ("What can you help me with?"), add 1 knowledge ("Search my documents for...") |
| SIDE-004 | P2 | Tab management (multiple conversation tabs) exists but is not explained — users don't know they can pin, rename, or set status on tabs | A | Show brief tooltip on first tab creation: "You can right-click tabs to pin, rename, or organize conversations" |
| SIDE-005 | P2 | Voice input options (server STT, browser speech, voice chat) are available but distinction is unclear — three different voice modes with no explanation | A | Consolidate to a single "Voice" button with submenu, or add brief labels: "Dictate (type what you say)", "Voice chat (conversation)" |
| SIDE-006 | P2 | CORS mismatch error detection exists but the guidance message is highly technical: "Set ALLOWED_ORIGINS to include [origin]" | A | Simplify to: "Your browser can't connect to the server due to a security setting. Ask your server admin to allow connections from your browser, or check the setup guide." |
| SIDE-007 | P3 | "Connected" status shows green dot but no server name/URL — user can't confirm WHICH server they're connected to | B | Show truncated server hostname next to status: "Connected to localhost:8000" |
| SIDE-008 | P3 | Sidepanel settings accessible via gear icon but opens full Options page — no quick in-panel settings for common tweaks (model, text size) | B | Add a lightweight settings drawer in sidepanel for model selection, text size, and temporary chat toggle |

## Category: Monitoring Dashboard (MON)

| ID | P | Title | Persona | Fix Direction |
|----|---|-------|---------|---------------|
| MON-001 | P0 | Alert assign action hardcodes `user_id: 1` instead of current user — "assign to you" assigns to wrong user in multi-user setups | Both | Use actual current user ID from auth context. Also: backend expects `assigned_to_user_id` but frontend sends `user_id` — payload field name mismatch |
| MON-002 | P1 | No explanation of what monitoring does — page title is "Monitoring & Alerting" with no description, no "getting started" guidance, no link to docs | A | Add introductory text: "Monitor your tldw server's health and set up alerts for important metrics. Create rules below to get notified when something needs attention." |
| MON-003 | P1 | Alert rule creation requires typing metric names freehand — no dropdown, no autocomplete, no list of available metrics | Both | Add a metric picker dropdown populated from `/admin/stats` response keys, or at minimum show example metrics as placeholder text: "e.g., cpu_usage, memory_mb, request_count" |
| MON-004 | P1 | System Overview shows raw key-value dumps with JSON.stringify'd objects — no formatting, no units, no labels explaining what metrics mean | Both | Format values with units (bytes -> MB/GB, ms, %, count). Add brief labels or tooltips explaining each metric. Group related metrics (CPU, Memory, Disk, Network). |
| MON-005 | P1 | Empty alert rules table shows nothing — no "Create your first alert rule" guidance, no suggested starter rules | Both | Add empty state: "No alert rules configured. Create your first rule below, or try a starter: [CPU > 90% for 5 min] [Memory > 85% for 10 min] [Disk > 95%]" with one-click creation |
| MON-006 | P2 | Duration field in alert rule form is unexplained — "Duration (min)" with no context for what it means | A | Add helper text: "Alert triggers when the metric stays above/below the threshold for this many minutes continuously" |
| MON-007 | P2 | Threshold field has no guidance on typical ranges — user must guess what values are normal | Both | Show dynamic hint based on selected metric: "Current value: 45%, typical range: 20-60%" (requires API enhancement) or static guidance per metric type |
| MON-008 | P2 | Severity levels (low/medium/high/critical) have color coding but no explanation of what each level means or triggers | A | Add tooltip: "Critical: immediate attention needed, High: investigate soon, Medium: monitor closely, Low: informational" |
| MON-009 | P2 | Snooze action is hardcoded to 1 hour — no option to choose duration | Both | Add dropdown: "Snooze for: 30 min / 1 hour / 4 hours / 24 hours / Custom" |
| MON-010 | P2 | Recent Activity section is collapsed by default and empty on first visit — users may never discover it | A | Show expanded on first visit with explanatory text: "Recent dashboard activity will appear here as you and other admins interact with the monitoring system" |
| MON-011 | P2 | Security Alert Status shows raw key-value dump identical to System Overview — same formatting/labeling problems | Both | Same fix as MON-004: format values, add labels, group related items |
| MON-012 | P1 | No refresh indicator or auto-refresh — user must manually click refresh to see updated metrics, no indication of data staleness | Both | Add "Last updated: X ago" timestamp and optional auto-refresh toggle (30s/60s/5m) |
| MON-013 | P3 | Alert history "Alert" column falls back to `record.metric` or `record.id` — naming is ambiguous | Both | Rename column to "Rule / Metric" and show both the rule name (if exists) and metric name |
| MON-014 | P3 | No way to edit existing alert rules — only create and delete | B | Add edit action to rules table, or show rule details in a drawer with editable fields |
| MON-015 | P3 | No way to test an alert rule — user creates rule and waits, no "test now" or "dry run" | B | Add "Test rule" button that evaluates the rule against current metrics and shows whether it would trigger |
| MON-016 | P1 | Backend `enabled` field exists on alert rules (admin_schemas.py) but frontend doesn't expose it in create form — can't disable a rule without deleting it | B | Add "Enabled" toggle to alert rule form and table |
| MON-017 | P3 | When monitoring APIs return 404/501 ("Not Available"), there's no guidance on how to enable monitoring on the server | Both | Add help text: "Monitoring features require server configuration. Check your server's config.txt or documentation to enable the monitoring module." |
| MON-BUG-002 | P0 | `duration_minutes` and `severity` are required by backend (`admin_schemas.py:1026-1027`) but typed as optional in frontend (`admin.ts:206`). Form doesn't mark them required. Users who omit these fields get a cryptic 422 error. | Both | Add `rules: [{ required: true }]` to Duration and Severity form items. Fix `admin.ts` type signatures. Add asterisk required indicators. |
| MON-BUG-003 | P2 | Alert history `rowKey` uses `Math.random()` fallback (~line 473), breaking React reconciliation — rows unmount/remount on every re-render when records lack `id` | B | Replace `Math.random()` with index-based fallback key |
| MON-A11Y-001 | P2 | Zero accessibility markup in MonitoringDashboardPage — no `aria-label`, `role`, `aria-live`, or `sr-only` elements. Screen reader users cannot use the page. | Both | Add `aria-label` to sections, `aria-live="polite"` to dynamic metric displays, `role="alert"` to error states |
| MON-I18N-001 | P2 | No `useTranslation` import in MonitoringDashboardPage — 50+ hardcoded English strings. Page is untranslatable. | Both | Move all user-visible strings to locale files and use `t()` calls |
| MON-RESPONSIVE-001 | P2 | Alert rule form uses `layout="inline"` (~line 398), placing all 5 fields + button on one line. Breaks below ~900px width with unpredictable wrapping. | A | Replace with responsive grid layout (Ant Design Row/Col or CSS grid) |

## Category: Watchlists (WATCH)

| ID | P | Title | Persona | Fix Direction |
|----|---|-------|---------|---------------|
| WATCH-001 | P2 | Beginner vs Advanced path choice is shown but "Beginner (guided)" vs "Advanced (direct forms)" labels assume users know what "guided" means in this context — some users won't self-identify | A | Rephrase as: "Step-by-step setup (recommended for new users)" vs "I know what I'm doing (show me the forms)" |
| WATCH-002 | P2 | Quick Setup Step 0 asks for feed URL but non-technical users may not know what an RSS feed URL is or where to find one | A | Add "Where do I find a feed URL?" collapsible hint with examples: "Most blogs and news sites have RSS feeds. Try adding /feed or /rss to the site URL, or search for 'site name RSS feed'" |
| WATCH-003 | P2 | Feed type selector shows "Forum (coming soon)" as disabled — this is confusing for new users who may think the feature is broken | A | Either hide the disabled option entirely or add clear "Not yet available" badge with no interaction |
| WATCH-004 | P2 | Quick Setup Step 1 "Setup goal" offers "Briefing" vs "Triage" — "Triage" is jargon that non-technical users won't understand | A | Rename to: "Full reports (AI-generated summaries)" vs "Just collect articles (I'll read them myself)" |
| WATCH-005 | P2 | Template syntax (Jinja2) teach-point triggers on first Templates tab visit but gives no in-app examples — links to external Jinja2 docs which are dense and intimidating | A | Add 2-3 inline template examples with "Copy this template" buttons: a simple list, a summary format, a newsletter format |
| WATCH-006 | P2 | Feed test preview shows "X ingestable, Y filtered from Z sample items" — "ingestable" and "filtered" are technical terms | A | Rephrase: "Found X articles (Y skipped as duplicates or irrelevant) out of Z total" |
| WATCH-007 | P2 | Health bar with zero data shows minimal information — no explanation of what it will show once data exists | A | Add placeholder text: "Feed health and monitor status will appear here once you add feeds and create monitors" |
| WATCH-008 | P2 | Monitor creation "Advanced mode" exposes raw cron editor with link to crontab.guru — helpful for technical users but overwhelming if landed on by mistake | A | Gate behind explicit "Show cron editor" toggle, don't show by default even in advanced mode. Preset buttons should always be visible. |
| WATCH-009 | P3 | Orientation hints are permanently dismissible per tab — once dismissed, no way to re-trigger them | A | Add "Show tips again" option in Watchlists Settings tab |
| WATCH-010 | P3 | OPML import is a secondary CTA in empty state but not mentioned in Quick Setup wizard — users who have OPML files from other feed readers miss this path | Both | Add "Import existing feeds from OPML" as optional branch in Quick Setup Step 0 |
| WATCH-011 | P3 | Audio briefing toggle in Quick Setup defaults to enabled — non-technical users may not understand what "audio briefing" means or that it requires TTS to be configured on the server | A | Add brief explanation: "Generate an audio version of your report (requires text-to-speech to be set up on your server)" |
| WATCH-012 | P2 | Articles tab empty state says "No feed items found" (generic Antd Empty) instead of using the rich WatchlistsEmptyState component | Both | Replace with WatchlistsEmptyState for "articles" entity type which has contextual messaging and CTAs |
| WATCH-013 | P3 | No indication of which monitors are running or have run recently in the Overview — health bar shows counts but not recency | B | Add "Last run: X ago" per monitor in Overview summary, or show a "stale monitors" warning if no runs in 48h |

## Category: Cross-Cutting (XC)

| ID | P | Title | Persona | Fix Direction |
|----|---|-------|---------|---------------|
| XC-001 | P1 | No navigation breadcrumbs or "You are here" indicator on any of the three pages — users arriving via direct link or header shortcut don't know where they are in the app hierarchy | A | Add breadcrumb or page title with brief description beneath the header on each page |
| XC-002 | P1 | Header shortcut for Monitoring has no description (descriptionKey/descriptionDefault missing) — other shortcuts have descriptions explaining what the page does | Both | Add `descriptionDefault: "Server health alerts, security status, and system metrics"` (already in blocker plan Task 2 but verify it shipped) |
| XC-003 | P2 | No cross-page discovery — chat doesn't mention watchlists, watchlists doesn't mention chat, monitoring doesn't mention either. Each page is an island. | A | Add contextual "Did you know?" hints: Chat -> "Set up Watchlists to auto-collect content for your conversations", Watchlists -> "Chat about your collected articles in /chat", Monitoring -> "Monitor your watchlist health from here" |
| XC-004 | P2 | Tutorial system has tutorials for playground-basics, playground-tools, playground-voice but none for /watchlists or /monitoring | Both | Create watchlists-basics and monitoring-basics tutorials (3-5 steps each) using existing TutorialRunner infrastructure |
| XC-005 | P2 | Error messages across all three pages use different styles — chat uses toast notifications, monitoring uses inline Alert components, watchlists uses notification + modal mix | Both | Standardize error presentation: inline Alert for page-level errors, toast notification for action-level errors, modal for destructive confirmations |
| XC-006 | P3 | No "recently visited" or "quick actions" on the home/companion page for these three features — users must navigate via header shortcuts each time | A | Add recent pages section or pinnable shortcuts on CompanionHome |

---

## Review Corrections

The following corrections were identified during plan review:

1. **MON-BUG-002 added (P0)**: `duration_minutes`/`severity` required by backend but optional in frontend form. Users who omit these get cryptic 422 errors.
2. **MON-BUG-003 added (P2)**: `rowKey` uses `Math.random()` fallback, breaking React reconciliation.
3. **MON-A11Y-001 added (P2)**: Zero accessibility markup in entire monitoring page.
4. **MON-I18N-001 added (P2)**: No i18n -- 50+ hardcoded English strings.
5. **MON-RESPONSIVE-001 added (P2)**: Alert rule form `layout="inline"` breaks below ~900px.
6. **CHAT-002 downgraded P1 -> P2**: Model capabilities data already flows through API. Enhancement, not blocker.
7. **XC-001 downgraded P1 -> P2**: Pages already have titles. Breadcrumbs are wayfinding improvement.
8. **WATCH-012 upgraded P2 -> P1**: Articles tab is primary post-feed destination. Trivial fix, high impact.
9. **MON-016 upgraded P2 -> P1**: Can't disable rules without deleting. Core monitoring operational need.
10. **MON-012 upgraded P2 -> P1**: Monitoring dashboard without staleness indicator is misleading.
11. **SIDE-002 descoped**: Inline multi-user JWT login is complex. Sprint 2 = deep-link only; inline form deferred.
12. **XC-* ID namespace collision**: These should be referenced as XC2-001 through XC2-006 to avoid collision with existing audit's XC-001 through XC-006.

---

## Priority Summary

### P0 (Bug — fix immediately)
- **MON-001**: Alert assign hardcodes user_id=1 + payload field mismatch with backend
- **MON-BUG-002**: `duration_minutes`/`severity` required by backend but optional in frontend form

### P1 (Significant friction — fix in 1-2 sprints)
- **CHAT-001**: NoProviderBanner gives no actionable guidance on getting API keys
- **SIDE-001**: "Finish setup" doesn't deep-link to setup form (Chrome-specific code path)
- **SIDE-002**: No inline auth for multi-user mode in sidepanel (descoped to deep-link only)
- **MON-002**: No explanation of what monitoring does
- **MON-003**: Alert rule metric names must be typed freehand with no discovery
- **MON-004**: System Overview raw key-value dump with no formatting
- **MON-005**: Empty alert rules table has no guidance or starter rules
- **MON-012**: No staleness indicator or auto-refresh on monitoring dashboard
- **MON-016**: Can't disable alert rules without deleting them
- **WATCH-012**: Articles tab uses generic Empty instead of WatchlistsEmptyState
- **XC2-002**: Monitoring header shortcut missing description

### P2 (Suboptimal but user can succeed — fix in 1-2 months)
- CHAT-002, CHAT-003, CHAT-004, CHAT-005, CHAT-006, CHAT-007, CHAT-008, CHAT-009, CHAT-013
- SIDE-003, SIDE-004, SIDE-005, SIDE-006
- MON-006, MON-007, MON-008, MON-009, MON-010, MON-011, MON-BUG-003, MON-A11Y-001, MON-I18N-001, MON-RESPONSIVE-001
- WATCH-001, WATCH-002, WATCH-003, WATCH-004, WATCH-005, WATCH-006, WATCH-007, WATCH-008
- XC2-001, XC2-003, XC2-004, XC2-005

### P3 (Polish — backlog)
- CHAT-010, CHAT-011, CHAT-012
- SIDE-007, SIDE-008
- MON-013, MON-014, MON-015, MON-017
- WATCH-009, WATCH-010, WATCH-011, WATCH-013
- XC2-006

---

## Totals

**59 issues total** (53 original + 6 from review):
- P0: 2 (MON-001, MON-BUG-002)
- P1: 11
- P2: 32
- P3: 14

**By page:**
- Chat WebUI (CHAT): 13
- Chat Sidepanel (SIDE): 8
- Monitoring (MON): 22
- Watchlists (WATCH): 13
- Cross-cutting (XC2): 6

---

## Relationship to Existing Audits

| This Review | Existing Audit | Action |
|-------------|---------------|--------|
| CHAT-001 (NoProviderBanner guidance) | Blocker #4.1 & #4.8 (link + copy fix) | Extends -- blocker fix changes link target and removes .env jargon, but doesn't add API key signup links or plain-language explanation |
| XC-002 (monitoring nav description) | Blocker #6.1 (register route) | Verify -- blocker plan includes descriptionDefault, check if it shipped |
| CHAT-011 (demo mode not prominent) | v1 FE-003 | Related -- existing audit notes demo mode discovery issue |
| XC-004 (missing tutorials) | v1 FE-004 / Improvement 11 | Extends -- existing plan covers getting-started meta-tutorial but not per-page tutorials for watchlists/monitoring |
| MON-001 (assign bug) | Not in any existing audit | NEW -- critical bug found during this review |
| All MON-* issues | Not in v1/v2/ingest-knowledge audits | NEW -- monitoring page was not covered by any previous FTUE audit |
| All WATCH-* issues | Watchlists onboarding runbook (2026-02-24) | Supplements -- existing runbook covers guided/advanced paths but not specific UX copy issues |

---

## Key Files Referenced

| Component | File Path |
|-----------|-----------|
| PlaygroundChat | `apps/packages/ui/src/components/Option/Playground/PlaygroundChat.tsx` |
| PlaygroundEmpty | `apps/packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx` |
| NoProviderBanner | `apps/packages/ui/src/components/Common/NoProviderBanner.tsx` |
| Sidepanel Chat | `apps/tldw-frontend/extension/routes/sidepanel-chat.tsx` |
| EmptySidePanel | `apps/packages/ui/src/components/Sidepanel/Chat/empty.tsx` |
| ConnectionBanner | `apps/packages/ui/src/components/Sidepanel/Chat/ConnectionBanner.tsx` |
| SidepanelForm | `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx` |
| MonitoringDashboard | `apps/packages/ui/src/components/Option/Admin/MonitoringDashboardPage.tsx` |
| Admin Error Utils | `apps/packages/ui/src/components/Option/Admin/admin-error-utils.ts` |
| WatchlistsPlayground | `apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx` |
| WatchlistsOverview | `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx` |
| WatchlistsEmptyState | `apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsEmptyState.tsx` |
| Header Shortcuts | `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts` |
| Route Registry | `apps/packages/ui/src/routes/route-registry.tsx` |
| Tutorial Definitions | `apps/packages/ui/src/tutorials/definitions/playground.ts` |
| Chat Settings Types | `apps/packages/ui/src/types/chat-settings.ts` |
| Admin API Methods | `apps/packages/ui/src/services/tldw/domains/admin.ts` |
| Admin Schemas (backend) | `tldw_Server_API/app/api/v1/schemas/admin_schemas.py` |
