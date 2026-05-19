# Adjustable Theme System — Design Document

**Date:** 2026-04-05
**Status:** Approved
**Scope:** WebUI (tldw-frontend) + Browser Extension (shared UI package)
**Out of scope:** Admin UI (separate HSL-based system), server-side theme persistence, community gallery

---

## 1. Overview

Expand the existing 17-color-token theme system into a full design-system-level theming solution covering colors, typography, shape, layout, and component variants. Add a dual-mode theme editor (quick + advanced) and JSON import/export for theme sharing.

**Architecture:** Flat token expansion (Approach A) with preset-plus-overrides UX for the quick editor (borrowed from Approach C). Tokens applied via CSS custom properties + data attributes on `<html>`. Ant Design components themed via ConfigProvider token mapping + programmatic `variant`/`shape` props.

---

## 2. Token System (~40 tokens)

### 2.1 Colors (19 tokens, per light/dark mode)

Existing 17 + 2 new shadow tokens:

| Token | Purpose | Example (light) |
|-------|---------|-----------------|
| `bg` | Page background | `244 242 238` |
| `surface` | Card/panel fill | `255 255 255` |
| `surface2` | Secondary surface | `240 237 231` |
| `elevated` | Elevated surface | `251 250 247` |
| `primary` | Primary action | `47 111 237` |
| `primaryStrong` | Primary hover | `36 86 199` |
| `accent` | Accent color | `31 181 159` |
| `success` | Success state | `34 160 123` |
| `warn` | Warning state | `217 119 6` |
| `danger` | Error/danger | `224 88 109` |
| `muted` | Muted elements | `102 112 133` |
| `border` | Default border | `226 221 211` |
| `borderStrong` | Strong border | `207 200 186` |
| `text` | Primary text | `31 35 40` |
| `textMuted` | Secondary text | `91 100 114` |
| `textSubtle` | Tertiary text | `110 120 135` |
| `focus` | Focus ring | `13 134 119` |
| `shadowSm` | Small shadow | `0 1px 3px rgba(0,0,0,0.12)` |
| `shadowMd` | Medium shadow | `0 6px 18px rgba(0,0,0,0.08)` |

Colors use RGB space-separated format for Tailwind opacity support. Shadows are full `box-shadow` strings. Both have separate light and dark values.

### 2.2 Typography (6 tokens, shared across modes)

| Token | Type | Default | Maps to |
|-------|------|---------|---------|
| `fontFamily` | string | `"Inter", sans-serif` | `--font-family` |
| `fontFamilyMono` | string | `"Courier New", monospace` | `--font-family-mono` |
| `fontSizeBody` | number (px) | `14` | `--font-size-body` |
| `fontSizeMessage` | number (px) | `15` | `--font-size-message` |
| `fontSizeCaption` | number (px) | `12` | `--font-size-caption` |
| `fontSizeLabel` | number (px) | `11` | `--font-size-label` |

These map to the **existing** semantic Tailwind classes (`text-body`, `text-message`, `text-caption`, `text-label`). Standard Tailwind sizes (`text-xs`, `text-sm`, `text-base`, `text-lg`) are **not overridden**.

**Available fonts** (bundled + system — no external loading):
Inter, Space Grotesk, Arimo, system-ui, Georgia, Courier New

### 2.3 Shape (5 tokens, shared)

| Token | Type | Default | Tailwind Default |
|-------|------|---------|-----------------|
| `radiusSm` | number (px) | `2` | `rounded-sm` = `0.125rem` ≈ `2px` |
| `radiusMd` | number (px) | `6` | `rounded-md` = `0.375rem` ≈ `6px` |
| `radiusLg` | number (px) | `8` | `rounded-lg` = `0.5rem` ≈ `8px` |
| `radiusXl` | number (px) | `12` | `rounded-xl` = `0.75rem` ≈ `12px` |
| `surfaceBlur` | number (px) | `0` | N/A |

Default values are chosen to match Tailwind's built-in defaults so the Default theme produces zero visual diff when radius tokens are wired up. Overriding `rounded-sm/md/lg/xl` is intentional — all ~1,669 existing usages become theme-aware.

### 2.4 Layout (5 tokens, shared)

| Token | Type | Default |
|-------|------|---------|
| `sidebarWidth` | number (px) | `260` |
| `sidebarCollapsedWidth` | number (px) | `64` |
| `headerHeight` | number (px) | `56` |
| `contentMaxWidth` | number (px) | `960` |
| `density` | enum | `"default"` |

**Density values:** `"compact" | "default" | "comfortable"`

Density is applied via **two mechanisms** (not `<html>` font-size — that would break standard text sizes and double-scale Ant Design):
- **Ant Design components:** `compactAlgorithm` added to ConfigProvider when `density === "compact"`
- **Custom components:** `data-density` attribute on `<html>` with targeted CSS overrides in `@layer components`

A `spacingUnit` CSS variable is derived from density and used in component-level overrides:
```css
[data-density="compact"]     { --spacing-unit: 3px; }
[data-density="default"]     { --spacing-unit: 4px; }
[data-density="comfortable"] { --spacing-unit: 5px; }
```

### 2.5 Component Variants (4 tokens, shared)

| Token | Values | Application Method |
|-------|--------|-------------------|
| `buttonStyle` | `"rounded" \| "square" \| "pill"` | Ant Design: `shape` prop. Custom: `data-button-style` CSS |
| `inputStyle` | `"bordered" \| "underlined" \| "filled"` | Ant Design: `variant` prop. Custom: `data-input-style` CSS |
| `cardStyle` | `"flat" \| "elevated" \| "outlined"` | CSS via `data-card-style` (no Ant Design Card used) |
| `animationSpeed` | `"none" \| "subtle" \| "normal"` | Sets `--duration-base` CSS variable |

**Split application strategy:**
- **Ant Design components** read variant tokens from theme context and receive them as React props (`variant`, `shape`). No CSS specificity wars.
- **Custom components** use `data-*` attribute selectors in CSS targeting `.tldw-btn`, `.panel-card`, `.panel-input`, etc.

---

## 3. Type Definitions

```typescript
interface ThemeColorTokens {
  bg: string; surface: string; surface2: string; elevated: string;
  primary: string; primaryStrong: string; accent: string;
  success: string; warn: string; danger: string;
  muted: string; border: string; borderStrong: string;
  text: string; textMuted: string; textSubtle: string;
  focus: string;
  shadowSm: string;
  shadowMd: string;
}

interface ThemeTypography {
  fontFamily: string;
  fontFamilyMono: string;
  fontSizeBody: number;
  fontSizeMessage: number;
  fontSizeCaption: number;
  fontSizeLabel: number;
}

interface ThemeShape {
  radiusSm: number;
  radiusMd: number;
  radiusLg: number;
  radiusXl: number;
  surfaceBlur: number;
}

interface ThemeLayout {
  sidebarWidth: number;
  sidebarCollapsedWidth: number;
  headerHeight: number;
  contentMaxWidth: number;
  density: "compact" | "default" | "comfortable";
}

interface ThemeComponents {
  buttonStyle: "rounded" | "square" | "pill";
  inputStyle: "bordered" | "underlined" | "filled";
  cardStyle: "flat" | "elevated" | "outlined";
  animationSpeed: "none" | "subtle" | "normal";
}

interface ThemeDefinition {
  id: string;
  name: string;
  description?: string;
  version: 1;
  builtin: boolean;
  palette: { light: ThemeColorTokens; dark: ThemeColorTokens };
  typography: ThemeTypography;
  shape: ThemeShape;
  layout: ThemeLayout;
  components: ThemeComponents;
  basePresetId?: string;   // Set when created via quick mode
}
```

---

## 4. Token Application Architecture

### 4.1 CSS Custom Properties (on `<html>` inline styles)

```css
/* Colors — existing pattern, per-mode */
--color-bg: 244 242 238;
--color-primary: 47 111 237;
/* ... all 17 color tokens ... */

/* Shadows — new, per-mode */
--shadow-sm: 0 1px 3px rgba(0,0,0,0.12);
--shadow-md: 0 6px 18px rgba(0,0,0,0.08);

/* Typography — shared */
--font-family: "Inter", sans-serif;
--font-family-mono: "Courier New", monospace;
--font-size-body: 14px;
--font-size-message: 15px;
--font-size-caption: 12px;
--font-size-label: 11px;

/* Shape — shared */
--radius-sm: 2px;
--radius-md: 6px;
--radius-lg: 8px;
--radius-xl: 12px;
--surface-blur: 0px;

/* Layout — shared */
--sidebar-width: 260px;
--sidebar-collapsed-width: 64px;
--header-height: 56px;
--content-max-width: 960px;
```

### 4.2 Data Attributes (on `<html>`)

```html
<html
  class="dark"
  data-density="default"
  data-button-style="rounded"
  data-input-style="bordered"
  data-card-style="elevated"
  data-animation="normal"
  style="--color-bg: 15 17 19; --font-family: 'Inter', sans-serif; ..."
>
```

### 4.3 Density CSS (in `tailwind-shared.css`)

```css
/* Spacing unit derived from density */
[data-density="compact"]     { --spacing-unit: 3px; }
[data-density="default"]     { --spacing-unit: 4px; }
[data-density="comfortable"] { --spacing-unit: 5px; }

/* Component-level density overrides */
[data-density="compact"] .panel-card { padding: calc(var(--spacing-unit) * 2); }
[data-density="compact"] .tldw-btn  { padding: calc(var(--spacing-unit) * 1) calc(var(--spacing-unit) * 3); }
[data-density="comfortable"] .panel-card { padding: calc(var(--spacing-unit) * 4); }
[data-density="comfortable"] .tldw-btn  { padding: calc(var(--spacing-unit) * 2) calc(var(--spacing-unit) * 5); }
```

### 4.4 Animation Speed CSS

```css
[data-animation="none"]   { --duration-base: 0ms; }
[data-animation="subtle"] { --duration-base: 75ms; }
[data-animation="normal"] { --duration-base: 150ms; }

/* Applied in @layer components transitions */
.panel-card   { transition: box-shadow var(--duration-base) ease-out; }
.panel-input  { transition: border-color var(--duration-base) ease-out; }
.tldw-btn     { transition: background-color var(--duration-base) ease-out; }
```

### 4.5 Component Variant CSS (custom components only)

```css
/* Buttons — custom components only; Ant Design uses shape prop */
[data-button-style="square"]  .tldw-btn { border-radius: var(--radius-sm); }
[data-button-style="rounded"] .tldw-btn { border-radius: var(--radius-md); }
[data-button-style="pill"]    .tldw-btn { border-radius: 9999px; }

/* Inputs — custom components only; Ant Design uses variant prop */
[data-input-style="underlined"] .panel-input {
  border-radius: 0;
  border: none;
  border-bottom: 1px solid rgb(var(--color-border));
}
[data-input-style="filled"] .panel-input {
  border: none;
  background: rgb(var(--color-surface2));
}

/* Cards */
[data-card-style="flat"]     .panel-card { box-shadow: none; border: none; }
[data-card-style="elevated"] .panel-card { box-shadow: var(--shadow-md); }
[data-card-style="outlined"] .panel-card {
  box-shadow: none;
  border: 1px solid rgb(var(--color-border));
}
```

### 4.6 Tailwind Config (expanded)

```javascript
// tailwind.config.js — theme.extend additions
{
  colors: { /* existing 17 color mappings unchanged */ },
  fontFamily: {
    body: "var(--font-family)",
    mono: "var(--font-family-mono)",
    // display, arimo remain as-is (static)
  },
  fontSize: {
    body:    ["var(--font-size-body)",    { lineHeight: "1.43" }],
    message: ["var(--font-size-message)", { lineHeight: "1.47" }],
    caption: ["var(--font-size-caption)", { lineHeight: "1.33" }],
    label:   ["var(--font-size-label)",   { lineHeight: "1.27" }],
    // text-xs, text-sm, text-base, text-lg UNTOUCHED
  },
  borderRadius: {
    sm:  "var(--radius-sm)",
    md:  "var(--radius-md)",
    lg:  "var(--radius-lg)",
    xl:  "var(--radius-xl)",
    // none, DEFAULT, 2xl, 3xl, full UNTOUCHED
  },
  boxShadow: {
    sm:    "var(--shadow-sm)",
    md:    "var(--shadow-md)",
    card:  "var(--shadow-md)",
    modal: "var(--shadow-md)",
  },
}
```

### 4.7 Ant Design Bridge (expanded)

```typescript
function buildAntdThemeConfig(
  tokens: ThemeColorTokens,
  typography: ThemeTypography,
  shape: ThemeShape,
  layout: ThemeLayout,
  isDark: boolean
): ThemeConfig {
  const algorithms = [isDark ? theme.darkAlgorithm : theme.defaultAlgorithm];
  if (layout.density === "compact") {
    algorithms.push(theme.compactAlgorithm);
  }

  return {
    algorithm: algorithms,
    token: {
      // Colors (existing)
      colorPrimary: rgbToHex(tokens.primary),
      colorSuccess: rgbToHex(tokens.success),
      colorWarning: rgbToHex(tokens.warn),
      colorError: rgbToHex(tokens.danger),
      colorTextBase: rgbToHex(tokens.text),
      colorBgBase: rgbToHex(tokens.bg),
      // Typography (new)
      fontFamily: typography.fontFamily,
      fontSize: typography.fontSizeBody,
      // Shape (new)
      borderRadius: shape.radiusMd,
      borderRadiusLG: shape.radiusLg,
      borderRadiusSM: shape.radiusSm,
      // Shadows (new)
      boxShadow: tokens.shadowSm,
      boxShadowSecondary: tokens.shadowMd,
    },
  };
}
```

### 4.8 Ant Design Variant Mapping

Components consuming Ant Design inputs/buttons read variant tokens from theme context:

```typescript
// Mapping theme tokens to Ant Design props
function useAntdVariants() {
  const { themeDefinition } = useTheme();
  const { buttonStyle, inputStyle } = themeDefinition.components;

  return {
    buttonShape: buttonStyle === "pill" ? "round" : "default",
    inputVariant: inputStyle === "underlined" ? "borderless"
                : inputStyle === "filled"     ? "filled"
                : "outlined",
  };
}
```

Components that render Ant Design `<Input>`, `<Select>`, `<Button>` call this hook and pass the mapped props. This avoids CSS specificity wars entirely.

---

## 5. Quick Mode Editor

### 5.1 The 10 Levers

| # | Lever | Control | Maps To |
|---|-------|---------|---------|
| 1 | Base preset | Dropdown | All tokens (starting point) |
| 2 | Primary color | Color picker | `primary`, `primaryStrong`, `focus` |
| 3 | Accent color | Color picker | `accent` |
| 4 | Background tint | Color picker | `bg`, `surface`, `surface2`, `elevated` |
| 5 | Text contrast | Slider | `text`, `textMuted`, `textSubtle` |
| 6 | Font family | Dropdown | `fontFamily` |
| 7 | Roundness | Slider | `radiusSm/Md/Lg/Xl` + `buttonStyle` |
| 8 | Density | 3-toggle | `density` (Ant Design compact + CSS overrides) |
| 9 | Sidebar width | Slider (200-400px) | `sidebarWidth` |
| 10 | Shadow intensity | Slider | `shadowSm`, `shadowMd` (both modes) |

### 5.2 Derivation Logic

**Background tint (lever 4):** Uses OKLCH color space. Takes the user's tint color and generates 4 surface colors at fixed lightness offsets:
- `bg` = base lightness
- `surface` = +3% lightness
- `surface2` = -2% lightness
- `elevated` = +5% lightness
- In dark mode, offsets are inverted (darker base, surfaces lighten upward)

Utility: `deriveSurfacePalette(tintHex: string, isDark: boolean) => { bg, surface, surface2, elevated }`

**Text contrast (lever 5):** Slider minimum is clamped to ensure WCAG AA (4.5:1 contrast ratio) between `text` and `bg`. Warning badge shown if contrast drops below AAA (7:1). Uses the existing `contrast.ts` utilities.

**Roundness (lever 7):** Deterministic mapping from slider position (0-100) to radius values:
- 0% (sharp): `radiusSm: 0, radiusMd: 0, radiusLg: 2, radiusXl: 4, buttonStyle: "square"`
- 50% (default): `radiusSm: 2, radiusMd: 6, radiusLg: 8, radiusXl: 12, buttonStyle: "rounded"`
- 100% (round): `radiusSm: 6, radiusMd: 12, radiusLg: 18, radiusXl: 24, buttonStyle: "pill"`

**Shadow intensity (lever 10):** Scales shadow opacity. Dark mode gets proportionally lower opacity (dark surfaces don't visually cast shadows).

Utility: `deriveShadows(intensity: number, isDark: boolean) => { shadowSm, shadowMd }`

### 5.3 UX Flow

1. User picks a base preset from dropdown
2. Adjusts levers — page updates live via immediate CSS variable injection
3. Clicks "Apply" → saves as custom theme with `basePresetId` set
4. "Cancel" reverts all changes
5. "Advanced" button switches to advanced mode with current values carried over

---

## 6. Advanced Mode Editor

### 6.1 Tabbed Sections

**Tab: Colors**
- 19-token grid with color pickers (17 colors + 2 shadow string inputs)
- Light/dark columns side by side
- Live WCAG contrast ratio display between `text*` tokens and `bg`
- Warning badges for AA/AAA violations

**Tab: Typography**
- Font family dropdown (curated list)
- Mono font dropdown
- 4 semantic size inputs (`body`, `message`, `caption`, `label`) with live preview text
- All sizes clamped to sane ranges (8-32px)

**Tab: Shape**
- 4 radius sliders with live preview rectangles
- Surface blur slider with preview
- Per-mode shadow string editors with visual preview

**Tab: Layout**
- Sidebar width slider (150-600px) with live resize
- Sidebar collapsed width slider (40-120px)
- Header height slider (40-80px)
- Content max-width slider (600-1400px)
- Density 3-toggle

**Tab: Components**
- Button style: 3 visual button previews to click
- Input style: 3 visual input previews to click
- Card style: 3 visual card previews to click
- Animation speed: 3-toggle with live transition preview

### 6.2 Persistent Controls (always visible)

- Theme name text input
- Light/dark mode toggle for previewing both modes
- Per-tab "Reset to preset defaults" button
- "Apply" / "Cancel" buttons

---

## 7. JSON Import/Export

### 7.1 Export Format

```json
{
  "tldw_theme": true,
  "version": 1,
  "exported_at": "2026-04-05T12:00:00Z",
  "theme": {
    "id": "custom-my-cozy-theme-abc123",
    "name": "My Cozy Theme",
    "description": "Warm tones with generous spacing",
    "version": 1,
    "builtin": false,
    "palette": {
      "light": { "bg": "244 242 238", "shadowSm": "0 1px 3px rgba(0,0,0,0.12)", "...": "..." },
      "dark": { "bg": "15 17 19", "shadowSm": "0 1px 2px rgba(0,0,0,0.3)", "...": "..." }
    },
    "typography": { "fontFamily": "Georgia", "fontSizeBody": 14, "...": "..." },
    "shape": { "radiusMd": 8, "surfaceBlur": 4, "...": "..." },
    "layout": { "sidebarWidth": 300, "density": "comfortable", "...": "..." },
    "components": { "buttonStyle": "pill", "inputStyle": "bordered", "...": "..." },
    "basePresetId": "solarized"
  }
}
```

Full resolved values (not diffs). Themes are self-contained and portable without needing the base preset.

### 7.2 Export Flow

1. "Export" button available on any custom theme in the theme picker
2. Built-in presets show "Duplicate & Export" (creates a copy first)
3. Downloads `tldw-theme-{slug}.json`

### 7.3 Import Flow

1. "Import Theme" → file picker (`.json` only)
2. Parse and validate via `validateThemeFile()`
3. On success: generate new `id`, set `builtin: false`, save to custom themes, auto-select
4. Toast: "Imported 'My Cozy Theme'"

### 7.4 Validation (`validateThemeFile()`)

```typescript
type ValidationResult =
  | { valid: true; theme: ThemeDefinition; warnings: string[] }
  | { valid: false; error: string }
```

**Hard rejections:**
- JSON not parseable
- `tldw_theme` marker missing
- `version > CURRENT_VERSION` → "This theme requires a newer version of tldw"
- Required sections missing (palette, typography, shape, layout, components)
- Color values don't match RGB triplet pattern
- Numeric values wildly out of bounds (negative, > 10000)
- String enums not in allowed values

**Soft warnings (allow with confirmation):**
- `fontFamily` not in curated list → "Font 'Helvetica' is not bundled and may not render consistently. Import anyway?"
- Contrast ratios below WCAG AA → "Some text/background combinations have low contrast"

### 7.5 Migration (`migrateTheme()`)

```typescript
const CURRENT_THEME_VERSION = 1;

function migrateTheme(raw: Record<string, unknown>): ThemeDefinition {
  const version = (raw.version as number) ?? 0;

  if (version > CURRENT_THEME_VERSION) {
    throw new Error("Theme requires a newer version of tldw");
  }

  if (version < 1) {
    // v0 (17-token schema, no version field) → v1
    // Backfill: typography, shape, layout, components from Default preset
    // Backfill: shadowSm/shadowMd in palette from Default preset
    // Set version to 1
  }

  return raw as ThemeDefinition;
}
```

Runs on `getCustomThemes()` load. **Persists migrated result** back to localStorage immediately after first migration to avoid re-running on every load.

---

## 8. Implementation Staging

| Stage | Scope | Risk | Notes |
|-------|-------|------|-------|
| **1** | Type definitions, `ThemeDefinition` expansion, migration logic, validation utilities | Low | No visual changes. Pure types and logic. |
| **2** | Shadow tokens added to palette (light/dark), updated `applyThemeTokens()`, Ant Design bridge expansion | Low | Extends existing pattern. Existing themes migrated automatically. |
| **3** | Typography tokens: CSS variables, Tailwind config for semantic sizes (`text-body` etc.), font-family wiring | Low | Only affects 4 semantic size classes, not standard Tailwind sizes. |
| **4** | Shape tokens: border-radius CSS variables, Tailwind `rounded-*` override, surface blur | Medium | **1,669 `rounded-*` usages shift to CSS vars.** Default values match Tailwind defaults → zero visual diff. Needs visual regression check. |
| **5** | Layout tokens: CSS variables for sidebar/header/content-width, component refactoring to use variables instead of hardcoded classes | High | **Most invasive stage.** Requires finding and replacing hardcoded layout values in components. Separate PR recommended. |
| **6** | Density system: `data-density` attribute, `--spacing-unit` CSS variable, targeted component overrides, Ant Design `compactAlgorithm` | Medium | New CSS rules in `@layer components`. Ant Design integration straightforward. |
| **7** | Component variant system: `.tldw-btn` class addition, `data-*` attribute CSS selectors, `useAntdVariants()` hook, adoption sweep to add `.panel-card`/`.tldw-btn` to components | Medium | Cross-cutting. Includes sweep to increase `.panel-card` and `.tldw-btn` adoption. |
| **8** | JSON import/export: file validation, migration on import, UI for import/export buttons | Low | Self-contained feature. |
| **9** | Quick mode editor: 10-lever UI, OKLCH derivation utility, contrast clamping, live preview | Medium | New component. Derivation logic (OKLCH, shadows) needs testing. |
| **10** | Advanced mode editor: 5-tab UI, per-token editing, visual variant pickers, reset-to-defaults | Medium | Largest UI component. Can reuse quick-mode preview infrastructure. |
| **11** | Update all 5 built-in presets (Default, Solarized, Nord, High Contrast, Rose Pine) to include new token sections | Low | Straightforward value assignment. |

**Critical path:** Stages 1-4 can proceed sequentially. Stage 5 (layout refactor) is the riskiest and can be parallelized with stages 8-10 (import/export and editors). Stage 7 (component variants) depends on stages 4 and 6.

---

## 9. Key Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| Don't override standard Tailwind text sizes | 5,729 usages; line-height tuples can't be expressed as single CSS variables |
| Override Tailwind `rounded-*` with CSS variables | Desired: makes all 1,669 usages theme-aware; default values match exactly |
| Density via targeted CSS, not `<html>` font-size | Font-size scaling would break standard text sizes and double-scale Ant Design |
| Ant Design variants via props, not CSS overrides | Avoids `!important` specificity wars with Ant Design's CSS-in-JS |
| Shadows per-mode (light/dark) | Dark backgrounds need subtler or no shadows |
| Bundled + system fonts only | No network dependency, no privacy concerns, self-contained |
| `surfaceBlur` in shape (not components) | It's a numeric CSS value, not an enum like component variants |
| Full resolved JSON export (not diffs) | Themes are portable without needing the base preset |
| Import warns (not rejects) for unknown fonts | Flexibility for manual JSON editing; font may be a valid system font |
| Forward version rejection on import | Prevents corrupt state from themes created by newer tldw versions |
| `migrateTheme()` persists after first run | Avoids re-running migration logic on every localStorage read |

---

## 10. Scope Exclusions

- **Admin UI** — has its own HSL-based theme system, not covered
- **Google Fonts / external font loading** — not supported
- **Server-side theme persistence** — themes are client-only (localStorage + Plasmo storage)
- **Community gallery / shareable URLs** — out of scope, JSON file export only
- **Standard Tailwind text sizes** (`text-xs/sm/base/lg/xl`) — not overridden
- **`rounded` (DEFAULT), `rounded-none`, `rounded-full`** — not overridden, only `sm/md/lg/xl`
