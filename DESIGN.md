---
name: tldw
description: Self-hosted research and media-analysis UI for source-grounded work.
colors:
  bg-light: "#f4f2ee"
  surface-light: "#ffffff"
  surface-2-light: "#f0ede7"
  elevated-light: "#fbfaf7"
  text-light: "#1f2328"
  text-muted-light: "#5b6472"
  text-subtle-light: "#6e7887"
  border-light: "#e2ddd3"
  border-strong-light: "#cfc8ba"
  bg-dark: "#0f1113"
  surface-dark: "#171a1f"
  surface-2-dark: "#1d2128"
  elevated-dark: "#232832"
  text-dark: "#e7e9ee"
  text-muted-dark: "#b3bac6"
  text-subtle-dark: "#9098a6"
  border-dark: "#2b313b"
  border-strong-dark: "#3b4350"
  primary: "#2f6fed"
  primary-strong: "#2456c7"
  primary-dark: "#5c8dff"
  primary-strong-dark: "#3e6ae0"
  accent: "#1fb59f"
  accent-dark: "#4fd1b0"
  success: "#22a07b"
  success-dark: "#3bd696"
  warn: "#d97706"
  warn-dark: "#f7b955"
  danger: "#e0586d"
  danger-dark: "#ff6b8b"
typography:
  display:
    fontFamily: "Space Grotesk, Inter, sans-serif"
    fontWeight: 600
    lineHeight: 1.1
    letterSpacing: "normal"
  headline:
    fontFamily: "Space Grotesk, Inter, sans-serif"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "normal"
  title:
    fontFamily: "Inter, system-ui, sans-serif"
    fontWeight: 600
    fontSize: "16px"
    lineHeight: 1.35
    letterSpacing: "normal"
  body:
    fontFamily: "Arimo, Inter, system-ui, sans-serif"
    fontWeight: 400
    fontSize: "14px"
    lineHeight: 1.43
    letterSpacing: "normal"
  message:
    fontFamily: "Arimo, Inter, system-ui, sans-serif"
    fontWeight: 400
    fontSize: "15px"
    lineHeight: 1.47
    letterSpacing: "normal"
  label:
    fontFamily: "Inter, system-ui, sans-serif"
    fontWeight: 600
    fontSize: "11px"
    lineHeight: 1.27
    letterSpacing: "0.04em"
  mono:
    fontFamily: "JetBrains Mono, Courier New, monospace"
    fontWeight: 400
    lineHeight: 1.4
rounded:
  sm: "2px"
  md: "6px"
  lg: "8px"
  xl: "12px"
  pill: "9999px"
spacing:
  unit: "4px"
  compact-unit: "3px"
  comfortable-unit: "5px"
  button-sm-y: "4px"
  button-sm-x: "10px"
  button-md-y: "6px"
  button-md-x: "14px"
  button-lg-y: "8px"
  button-lg-x: "20px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.surface-light}"
    rounded: "{rounded.md}"
    padding: "6px 14px"
    height: "36px"
  button-secondary:
    backgroundColor: "{colors.surface-2-light}"
    textColor: "{colors.text-light}"
    rounded: "{rounded.md}"
    padding: "6px 14px"
    height: "36px"
  button-danger:
    backgroundColor: "{colors.danger}"
    textColor: "{colors.surface-light}"
    rounded: "{rounded.md}"
    padding: "6px 14px"
    height: "36px"
  badge-status:
    backgroundColor: "{colors.surface-2-light}"
    textColor: "{colors.text-muted-light}"
    rounded: "{rounded.pill}"
    padding: "2px 8px"
  alert-info:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.surface-light}"
    rounded: "{rounded.lg}"
    padding: "12px"
  panel-card:
    backgroundColor: "{colors.surface-light}"
    textColor: "{colors.text-light}"
    rounded: "{rounded.xl}"
    padding: "16px"
  input-default:
    backgroundColor: "{colors.surface-2-light}"
    textColor: "{colors.text-light}"
    rounded: "{rounded.md}"
    padding: "8px 12px"
---

# Design System: tldw

## 1. Overview

**Creative North Star: "The Grounded Research Workbench"**

tldw should feel like a capable workbench for evidence-heavy thinking: calm enough for long sessions, explicit enough for operations, and structured enough to teach users where to go next. The system is product-first, with design serving source management, retrieval, chat, audio processing, evaluations, admin health, and durable work products.

The default expression is restrained and state-literate. Surfaces use warm, low-chroma neutrals in light mode and a dark operational palette for dim-room or infrastructure-heavy work. Blue carries primary commands, teal marks source-aware focus and active affordances, and semantic state colors do the hard work of recovery, readiness, and risk.

The system rejects the PRODUCT.md anti-references directly: generic AI-app gloss, pastel gradients, vague sparkle language, oversized marketing metrics, decorative agent mascots, hidden-controls simplicity, toy-like chat surfaces, cluttered enterprise admin screens, raw traceback dumps without recovery paths, modal-first routine flows, and visual drift between WebUI and extension.

**Key Characteristics:**

- Source-grounded and provenance-aware before decorative.
- Dense where expert work needs density, never dense by accident.
- Warm light mode for reading and onboarding, dark mode for sustained technical operation.
- Semantic state language for setup, auth, degraded service, retrying, blocked, empty, loading, error, and ready.
- Shared primitives and tldw-owned wrappers define product meaning; Ant Design remains an implementation substrate.

## 2. Colors

The palette is a restrained product system: warm paper-like light neutrals, charcoal dark-mode neutrals, command blue, source teal, and clear state colors.

### Primary

- **Command Blue** (#2f6fed): Primary calls to action, retrying states, selected controls, link-like emphasis, and important active affordances.
- **Command Blue Strong** (#2456c7): Hover and active treatment for primary actions in light mode.
- **Night Command Blue** (#5c8dff): Dark-mode primary actions and selected states, tuned brighter for contrast on charcoal surfaces.

### Secondary

- **Source Teal** (#1fb59f): Source-aware focus, citation-related affordances, active connection indicators, and calm confirmation that is not final success.
- **Night Source Teal** (#4fd1b0): Dark-mode focus and accent color. This also anchors focus rings in dark mode.

### Tertiary

- **Ready Green** (#22a07b): Completed, ready, and success states.
- **Setup Amber** (#d97706): Setup required, auth required, degraded service, warning, and user-action-needed states.
- **Review Rose** (#e0586d): Error, blocked, unavailable, danger, destructive, and permission-denied states.

### Neutral

- **Paper Field** (#f4f2ee): Light-mode app background. Use for broad page fields and low-emphasis work zones.
- **Work Surface** (#ffffff): Light-mode primary surface token. Use deliberately for dense product panels that need crisp separation.
- **Inset Linen** (#f0ede7): Light-mode secondary surface for inputs, list wells, inactive pills, and nested affordances.
- **Raised Parchment** (#fbfaf7): Light-mode elevated surfaces.
- **Ink** (#1f2328): Light-mode primary text.
- **Soft Ink** (#5b6472): Light-mode secondary text.
- **Hairline Linen** (#e2ddd3): Light-mode border and divider color.
- **Night Field** (#0f1113): Dark-mode app background.
- **Night Surface** (#171a1f): Dark-mode primary surface.
- **Night Inset** (#1d2128): Dark-mode secondary surface for input wells and nested panels.
- **Night Raised** (#232832): Dark-mode elevated surface.
- **Night Ink** (#e7e9ee): Dark-mode primary text.
- **Night Muted Ink** (#b3bac6): Dark-mode secondary text.
- **Night Hairline** (#2b313b): Dark-mode border and divider color.

### Named Rules

**The State Color Rule.** Success, warning, and danger colors belong to product state first. Do not spend them as decoration.

**The Teal Means Source Rule.** Teal should usually imply focus, source context, connectivity, or groundedness. Avoid using teal as random ornament.

**The Restrained Accent Rule.** Product screens should stay mostly neutral. Blue and teal are signposts, not wallpaper.

## 3. Typography

**Display Font:** Space Grotesk with Inter fallback
**Body Font:** Arimo and Inter with system fallback
**Label/Mono Font:** Inter for labels; JetBrains Mono with Courier New fallback for code-like content

**Character:** The pairing is functional and quietly technical. Space Grotesk gives headings enough identity without becoming a marketing voice; Arimo and Inter keep long product sessions readable; mono is reserved for diagnostics, code, IDs, and terminal-like details.

### Hierarchy

- **Display** (600, surface-specific size, 1.1 line height): Rare product identity moments, onboarding headers, and high-level workspace titles.
- **Headline** (600, surface-specific size, 1.2 line height): Page headers, pane titles, major setup stages, and admin section titles.
- **Title** (600, 16px, 1.35 line height): Card titles, state panel titles, modal titles, and compact section headings.
- **Body** (400, 14px, 1.43 line height): Default interface copy, settings descriptions, table content, and form help text.
- **Message** (400, 15px, 1.47 line height): Chat and generated-answer reading surfaces where a little more leading improves comprehension.
- **Label** (600, 11px, 0.04em letter spacing): Section labels, small metadata, badges, and compact product state labels.
- **Mono** (400, contextual size, 1.4 line height): Diagnostics, IDs, logs, code, endpoint paths, and values users may copy.

### Named Rules

**The Readable Evidence Rule.** Long source excerpts, generated text, notes, and chat answers should cap line length around 65 to 75 characters when possible.

**The Label Sparingly Rule.** Uppercase or tracked labels are for structure and metadata. Do not make normal body copy shout.

## 4. Elevation

tldw uses a hybrid of tonal layering, borders, and restrained shadows. Most depth is communicated by surface color and a 1px border; shadows are used for cards, elevated panels, modal surfaces, and hover-responsive affordances. Flat and outlined variants are first-class options for dense screens.

### Shadow Vocabulary

- **Low Shadow** (`0 1px 3px rgba(0,0,0,0.12)` in light mode; `0 1px 2px rgba(0,0,0,0.3)` in dark mode): Small surface separation and low-lift controls.
- **Panel Shadow** (`0 6px 18px rgba(0,0,0,0.08)` in light mode; `0 4px 12px rgba(0,0,0,0.25)` in dark mode): Cards, elevated panels, and contained work surfaces.
- **Modal Shadow** (`0 10px 30px rgba(0,0,0,0.28)`): Blocking overlays and high-priority dialogs.
- **Focus Glow** (`0 0 0 2px rgb(var(--color-focus) / 0.4)`): Form focus and active keyboard feedback.
- **Accent Glow** (`0 0 0 1px rgb(var(--color-primary) / 0.35), 0 0 24px -6px rgb(var(--color-primary) / 0.55)`): Rare emphasis for composer and Primer-aesthetic surfaces.

### Named Rules

**The Border Before Shadow Rule.** Prefer borders and tonal layering for normal product structure. Add shadow only when the surface truly lifts, floats, or needs interaction feedback.

**The No Decorative Blur Rule.** Backdrop blur is allowed for overlays and existing loading treatments. It is not a default card style.

## 5. Components

### Buttons

Buttons are compact, direct, and stateful. They should look like controls for work, not marketing CTAs.

- **Shape:** Rounded by default (6px), square option (2px), pill option only for chips or special compact affordances.
- **Primary:** Command Blue background with light text. Default medium size is 36px tall with 14px horizontal padding.
- **Hover / Focus:** Hover deepens the blue. Focus uses a visible teal focus ring with offset against the app background. Motion is 150ms ease-out and must respect reduced motion.
- **Secondary / Ghost / Tertiary:** Secondary uses surface-2 with text. Ghost is transparent with muted text and a surface-2 hover. Text variant is for link-like actions, not primary workflow steps.
- **Danger:** Review Rose background with light text, reserved for destructive actions.

### Chips

Chips and badges communicate status, filters, counts, and compact labels.

- **Style:** Rounded pills by default, medium weight text, 10px to 12px label sizes.
- **State:** Filled status chips use 10 percent tint backgrounds with colored text. Outline chips use 30 percent borders.
- **Dot:** Dot badges should reinforce status with shape, not replace readable text.

### Cards / Containers

Cards are contained work surfaces, not the default answer for every layout.

- **Corner Style:** 12px for panel cards and 8px for state panels and alerts. Use 6px for compact nested controls.
- **Background:** Light cards use Work Surface or Raised Parchment. Dark cards use Night Surface or Night Raised.
- **Shadow Strategy:** Use Panel Shadow only when a panel needs actual lift. Use flat or outlined variants in dense admin, table, or settings contexts.
- **Border:** 1px semantic border is standard. Do not use colored side stripes.
- **Internal Padding:** 16px is the normal card baseline; 12px for alerts and compact panels; 24px to 32px for large empty states.

### Inputs / Fields

Inputs are quiet wells inside the surface, optimized for repeated use.

- **Style:** Surface-2 background, 1px border, 6px radius, 8px by 12px padding.
- **Focus:** Border shifts to Command Blue and adds a teal focus ring. Focus must be visible by keyboard alone.
- **Error / Disabled:** Error uses Review Rose plus clear text. Disabled states reduce opacity and keep layout stable.
- **Ant Design Policy:** Ant Design inputs are restyled through shared tokens. Do not let raw Ant Design blue or spacing leak into product surfaces.

### Navigation

Navigation should be predictable and scan-friendly.

- **Style:** Use semantic background and border tokens. Active items should combine color, text weight, and clear position.
- **Typography:** Compact labels are acceptable for dense sidebars, but route names and primary destinations must remain readable.
- **Mobile:** Prefer explicit tabs, drawers, or pane selectors over hover-only controls. Touch targets should approach 44px where practical.

### State Panels and Recovery Callouts

State panels are the signature shared component family for setup, health, admin, and degraded product states.

- **Structure:** Readable state badge, concise title, next-action message, optional diagnostics, and primary/secondary action group.
- **States:** ready, unavailable, setup required, auth required, permission denied, degraded, retrying, blocked, empty, loading, and error.
- **Diagnostics:** Diagnostics should be visible when useful, collapsed when secondary, and copyable when the user may need to file an issue.
- **Copy:** Name the failing target, what still works, and the shortest recovery path.

### Empty and Loading States

Empty and loading states are part of the product workflow.

- **Empty:** Explain why content is absent and provide the next creation, import, setup, or browse action. Use examples only when they help users start real work.
- **Loading:** Prefer skeletons when preserving layout matters; use spinners or dots for compact work. Always name what is loading when the wait may be ambiguous.
- **Interrupted Work:** Long-running jobs and generation flows should show status and, when possible, cancellation or retry paths.

### Tables and Dense Lists

Tables and dense lists are expected product surfaces.

- **Style:** Sticky headers, semantic borders, muted row alternation, and horizontal overflow for small screens.
- **Density:** Support compact and comfortable density through tokens, not one-off padding.
- **Metadata:** Keep IDs, dates, provider names, source counts, and state labels scannable.

## 6. Do's and Don'ts

### Do:

- **Do** use the shared semantic color tokens from `tailwind-shared.css` and Tailwind mappings rather than local raw color values.
- **Do** use state colors for state: Ready Green for ready/success, Setup Amber for setup/auth/degraded/warning, Review Rose for error/blocked/unavailable/destructive.
- **Do** keep product screens mostly neutral, then use Command Blue for primary action and Source Teal for focus, source context, and groundedness.
- **Do** make setup, auth, provider readiness, background jobs, degraded services, permissions, and failures legible without digging through logs.
- **Do** give icon-only controls accessible names, visible focus states, and non-hover discovery on touch devices.
- **Do** keep WebUI and extension surfaces visually related through shared tokens, primitives, and state language.
- **Do** respect reduced motion. Animations should be subtle, short, and non-essential.

### Don't:

- **Don't** use generic AI-app gloss: pastel gradients, vague sparkle language, oversized marketing metrics, decorative agent mascots, and feature cards that say little.
- **Don't** make chat, research, or admin surfaces feel toy-like. Serious research needs accountable controls, visible provenance, and recoverable state.
- **Don't** hide critical provider, source, permission, cost, processing, or recovery state behind "smart" defaults.
- **Don't** create cluttered enterprise admin screens that bury the next action. Dense is acceptable; undirected is not.
- **Don't** show raw traceback dumps without recovery paths, diagnostics affordances, or copyable context.
- **Don't** make modals the first thought for routine work. Prefer inline disclosure, panels, drawers, or progressive controls when the task is not blocking.
- **Don't** use colored side-stripe borders, gradient text, or decorative glassmorphism as product defaults.
- **Don't** add local button, alert, badge, empty-state, loading, or recovery styles when a shared primitive already exists.
