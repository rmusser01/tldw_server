# Settings generation timeout repair

Ordinary Settings save now uses 120-second generation budgets (Extended: 240 seconds), preserves deliberate custom limits, and allows Custom→Balanced to apply the preset. Author and independent runs each pass 25 tests across four suites. Scoped ESLint has no new diagnostics (33 existing warnings). Native source-answer acceptance and final compiler comparison remain pending.

The 28 broader baseline fixture failures are retained and addressed separately as UAT154. Stored old ten-second settings are not automatically migrated; native follow-up must explicitly select Balanced or Reset, save, and repeat the actual source request. These are automated component and transport checks, not real model acceptance. Bandit does not apply to TypeScript-only changes.

Final integration verification is retained in [the combined bundle](../followup151-154-combined/README.md):3326/168 shared-UI and332/17 WebUI pass separately; TypeScript remains at90existing diagnostics with0added/removed. Native152/153 acceptance remains pending.
