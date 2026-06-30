# Theming Guide

tldw's WebUI ships with a set of built-in themes (including **Primer**, the default for new users) and supports user-authored and whitelabel themes via JSON import/export. This guide documents the theme schema and the authoring workflow.

## Built-in themes

| ID | Name | Notes |
|---|---|---|
| `default` | Default | Original tldw palette |
| `primer` | Primer | Cyberpunk cyan-on-black; "Parchment Primer" in light mode. **Default for new users.** |
| `solarized` | Solarized | Warm academic palette |
| `nord` | Nord | Cool arctic tones |
| `high-contrast` | High Contrast | WCAG AAA compliant |
| `rose-pine` | Rose Pine | Muted purple-pink |

New users see `primer` on first load. Existing users keep whatever preset they had stored. The migration is one-shot (`tldw:themeMigrationVersion` in `localStorage`).

## Theme JSON schema

A theme is a single JSON object with both **light** and **dark** palettes:

```json
{
  "id": "my-theme",
  "name": "My Theme",
  "description": "A short description",
  "version": 1,
  "builtin": false,
  "palette": {
    "light": { /* ColorTokens */ },
    "dark":  { /* ColorTokens */ }
  },
  "typography": {
    "fontFamily": "\"Inter\", system-ui, sans-serif",
    "fontFamilyMono": "\"JetBrains Mono\", \"Courier New\", monospace",
    "fontSizeBody": 14,
    "fontSizeMessage": 15,
    "fontSizeCaption": 12,
    "fontSizeLabel": 11
  },
  "shape":      { "radiusSm": 2, "radiusMd": 6, "radiusLg": 8, "radiusXl": 12, "surfaceBlur": 0 },
  "layout":     { "sidebarWidth": 260, "sidebarCollapsedWidth": 64, "headerHeight": 56, "contentMaxWidth": 960, "density": "default" },
  "components": { "buttonStyle": "rounded", "inputStyle": "bordered", "cardStyle": "elevated", "animationSpeed": "normal" }
}
```

### ColorTokens

Each palette (`light` and `dark`) is an object of **RGB triples as space-separated strings** — e.g., `"92 225 230"`. This lets Tailwind apply opacity via `rgb(var(--color-primary) / 0.08)`.

| Token | Use |
|---|---|
| `bg` | App background |
| `surface` | Primary surface (cards, panels) |
| `surface2` | Secondary surface (hover, nested) |
| `elevated` | Elevated surface (popovers, modals) |
| `primary` | Brand primary (buttons, accents) |
| `primaryStrong` | Darker primary (hover states) |
| `accent` | Secondary accent (highlights) |
| `success` | Success states |
| `warn` | Warning states |
| `danger` | Error/destructive states |
| `muted` | Muted surface/border blend |
| `border` | Standard borders |
| `borderStrong` | Emphasized borders |
| `text` | Primary text |
| `textMuted` | Secondary text |
| `textSubtle` | Tertiary text (captions, hints) |
| `focus` | Focus ring |
| `shadowSm` | Full CSS `box-shadow` string for small elevation |
| `shadowMd` | Full CSS `box-shadow` string for medium elevation |

### Typography

- `fontFamily` / `fontFamilyMono` — full CSS `font-family` values, quoted as strings.
- Font sizes in pixels.
- Fonts must be locally available. Bundled: Inter, Space Grotesk, Arimo, JetBrains Mono.

### Shape, layout, components

Enums are validated at import time. See [`apps/packages/ui/src/themes/types.ts`](../../apps/packages/ui/src/themes/types.ts) for the full TypeScript type.

## Contrast requirements

Built-in themes must pass the baseline in `apps/packages/ui/src/themes/__tests__/contrast-baseline.test.ts`:

- **WCAG AA text contrast** on primary reading surfaces (`text`/`textMuted` on `bg`/`surface`/`surface2`).
- **3:1 non-text contrast** for the `focus` indicator against `bg` and `surface`.

User-authored themes aren't forced through these checks on import, but ignoring them produces inaccessible interfaces. When whitelabeling, add your theme to `primer-preset.ts`-style file and run the contrast suite before shipping.

## Authoring workflow

### From the UI

1. Open **Settings → Appearance → Theme**.
2. Click **Create custom theme** (or duplicate an existing one).
3. Edit colors, typography, shape in the Theme Editor modal.
4. **Export** as a JSON file for sharing or version control.
5. **Import** a peer's theme JSON via the same modal.

### Programmatically (whitelabel)

1. Create a new file `apps/packages/ui/src/themes/my-brand-preset.ts` matching the shape of [`primer-preset.ts`](../../apps/packages/ui/src/themes/primer-preset.ts).
2. Export a `ThemeDefinition` with `builtin: true`.
3. Import and register in [`presets.ts`](../../apps/packages/ui/src/themes/presets.ts):
   ```ts
   import { myBrandTheme } from "./my-brand-preset"

   const PRESETS: ThemeDefinition[] = [
     defaultTheme,
     primerTheme,
     myBrandTheme,
     // ...
   ]
   ```
4. Run the test suite:
   ```bash
   cd apps/packages/ui
   npx vitest run src/themes/__tests__
   ```
   The `contrast-baseline` test will fail if your palette doesn't meet AA contrast. Adjust until green.

### Making your theme the default

To flip the default for *new* users to your brand:

1. Bump `CURRENT_USER_PREFERENCE_MIGRATION` in [`user-preference-migration.ts`](../../apps/packages/ui/src/themes/user-preference-migration.ts).
2. Extend the migration function to write your preset ID for the new version.

Migration preserves existing user selections — only installs with no stored preset get the new default.

## Composer variants × theme

The chat-composer variants (V1 Terminal Stack, V3 Split Brief, V5 Radial Command) are **theme-agnostic**: they reference only CSS custom properties (`--color-primary`, `--glow-primary`, etc.). Any registered theme reskins all three variants automatically.

Optional glow tokens (`--glow-primary`, `--glow-accent`, `--glow-success`) are defined in `tailwind-shared.css` derived from the active theme's `primary`/`accent`/`success` — your brand palette drives the glow color without additional work.

## File reference

- [`apps/packages/ui/src/themes/types.ts`](../../apps/packages/ui/src/themes/types.ts) — TypeScript type for `ThemeDefinition`
- [`apps/packages/ui/src/themes/presets.ts`](../../apps/packages/ui/src/themes/presets.ts) — Built-in theme registry
- [`apps/packages/ui/src/themes/primer-preset.ts`](../../apps/packages/ui/src/themes/primer-preset.ts) — Primer theme source
- [`apps/packages/ui/src/themes/user-preference-migration.ts`](../../apps/packages/ui/src/themes/user-preference-migration.ts) — One-shot default migration
- [`apps/packages/ui/src/themes/validation.ts`](../../apps/packages/ui/src/themes/validation.ts) — Import-time schema validation
- [`apps/packages/ui/src/assets/tailwind-shared.css`](../../apps/packages/ui/src/assets/tailwind-shared.css) — Global CSS custom properties and `@font-face` rules
