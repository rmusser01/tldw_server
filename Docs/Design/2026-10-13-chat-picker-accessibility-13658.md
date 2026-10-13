# Chat picker accessibility — TASK-13658

Use existing readable foreground tokens for the model picker group headings and help link, overriding AntD defaults where they otherwise win the CSS cascade. Accent/background tokens are not guaranteed small-text contrast colors; no global palette or dependency change is needed. Add the existing translated startup launch caption as its accessible name. Keep all model selection, focus, navigation and startup dispatch contracts.

Actual current build axe: dark Current3.67/help4.33, light Current2.73; all require4.5. Whole-page roleplay/history/status text failures are separate outstanding scope; this unit does not claim to fix those. No baseline lowering, CSS security bypass, new theme API or redesign. ADR assessment: no new ADR; existing semantic token and shared component architecture.

Expanded configured/all-catalog scans expose nested provider headings, scope hint and active catalog-toggle text. Use text-text-muted for secondary prose and text-text for selected control text; preserve its accent border/background and aria-pressed indication. These repair the actual text owners, without altering icon contrast, provider keys, sorting, selection or dispatch.
