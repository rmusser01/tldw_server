# UAT183 / TASK13260.120 — additional historical persona fixtures

Associated and reopened before edits. Exact pre-UAT200 replay confirms both v25/v36 fixtures fail the existing Notes v59 registry collision guard: they create today's full database, rewrite its version backward, and retain newer tables. This is the already tracked fixture defect, not a new product defect or a UAT200 regression.

1. RED retained from current and exact pre-UAT200 store replay. Review the already accepted historical v21/v39 fixtures and the actual v25→26 and voice compatibility contracts.
2. Build real V4 then every registered SQLite migration up to25 or36 under a seed-only initializer/version cap. Assert each version, absence of newer registry, and correct historical persona/voice schema. Restore normal initialization before the current-head upgrade. For25, assert the exact25→26 step and seed persona/memory once their tables exist; preserve an earlier character row too. For36, seed existing persona and voice rows. Preserve every existing final columns/index/head assertion. No product changes or guard relaxation.
3. Focused tests then the original full adjacent command without deselections, baseline-aware Ruff/Bandit, assertion-preservation comparison and a frozen test-only review packet. Parent owns independent review, task/tracker/git/native.
