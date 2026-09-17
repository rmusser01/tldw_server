# UAT217 / TASK13260.151 — PostgreSQL literal keyword search

## Repair

The non-simple-token PostgreSQL branch binds its ILIKE operand as `ILIKE (?)` instead of `ILIKE ?`. The existing SQL converter interprets the bare `?` between ILIKE and ESCAPE as a possible JSONB operator and leaves it unconverted, while converting the LIMIT normally. Parentheses remove that ambiguity. Original parameter-bound backslash/%/_ escaping, SQLite behavior, simple-token full-text search, quote rejection, owner scope, ordering and limit are unchanged. No shared converter change or mixed native placeholders are introduced.

## Causal stages

Logs use the official required-PG runner and existing DB_Management fixture. All source and failed candidates are retained privately.

- Original22-control baseline in stages: `uat217-literal-search-red`6 PostgreSQL failures/14 pass (20cases), then ordered combined backslash/bang case1 PostgreSQL failure/1 SQLite pass. The underscore-only simple token uses full-text search and passes; it is not a failed literal-branch case.
- First `ESCAPE '!'` candidate failed: `uat217-literal-search-green` is merely a historical label; its actual result is **7fail/15pass**. Changing escape character did not change the converter's ILIKE/ESCAPE classification. `failed-escape-only.patch` and source snapshot retain it.
- Initial placeholder-count inference incorrectly named LIMIT. Actual prepared SQL/source inspection proved the unconverted operand is before ESCAPE. All current diagnosis uses that corrected cause; original evidence is preserved.
- Approved private comparison replays the real store method against real backends, without editing production: explicit `%s` operand22pass/0skip22.72s; parenthesized `?` operand22pass/0skip24.28s. Parentheses were selected as the smaller existing-style correction.
- Final frozen combined run202pass/0skip/2warnings284.51s includes these22 controls; all six integrated source/test hashes remain unchanged. The combined source contains separately attributed209 owner predicates and216 trigger method.

The permanent `test_keyword_literal_search_backends.py` checks literal plus/hyphen/%/underscore/bang/backslash/combinations against decoys, bound limited ordering, and existing empty/quote rejection on both backends. `owned.patch` contains only the two-character production edit and new test file. No model/native acceptance is claimed. Current scoped Ruff/Bandit and test formatting are clean; production/test source hashes are in209's integrated manifest.
