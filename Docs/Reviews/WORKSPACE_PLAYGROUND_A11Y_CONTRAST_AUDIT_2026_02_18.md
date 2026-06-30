# Workspace Playground Accessibility Contrast Audit (Category 11 Stage 3)

Date: 2026-02-18  
Scope: `apps/packages/ui/src/components/Option/ResearchWorkspace/`

## Checklist

- [x] Verify source icon token pairing (`text-text-muted` on `bg-surface2`) is AA-compliant.
- [x] Verify mobile tab badge token pairing is AA-compliant after remediation.
- [x] Record contrast measurements across built-in themes.
- [x] Add automated regression tests for the audited pairs.

## Measured Contrast Values (WCAG AA target >= 4.5:1 for normal text)

| Theme | Mode | `textMuted/surface2` | `text/surface2` | `white/success` (pre-fix badge risk) | `white/primary` (reference) |
|---|---|---:|---:|---:|---:|
| default | light | 5.12 | 13.52 | 3.29 | 4.55 |
| default | dark | 8.27 | 13.29 | 1.87 | 3.13 |
| solarized | light | 4.72 | 9.57 | 3.20 | 3.68 |
| solarized | dark | 9.22 | 10.47 | 3.20 | 3.68 |
| nord | light | 6.39 | 9.25 | 2.04 | 4.03 |
| nord | dark | 7.09 | 7.49 | 2.04 | 2.00 |
| high-contrast | light | 10.75 | 17.62 | 5.60 | 8.28 |
| high-contrast | dark | 11.03 | 16.67 | 1.63 | 2.62 |
| rose-pine | light | 4.53 | 6.05 | 6.11 | 7.27 |
| rose-pine | dark | 4.71 | 11.51 | 5.22 | 2.09 |

## Outcome

- `textMuted/surface2` passes AA across all built-in themes (minimum observed: **4.53:1**).
- Previous studio mobile badge pairing (`text-white` on `bg-success`) fails AA in several themes (minimum observed: **1.63:1**).
- Workspace mobile badges were updated to use `text-text` on `bg-surface2` to maintain AA across themes.

