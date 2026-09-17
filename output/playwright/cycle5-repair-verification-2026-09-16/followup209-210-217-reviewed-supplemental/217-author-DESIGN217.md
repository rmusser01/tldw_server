# UAT217 / TASK13260.151 — literal keyword search

## Corrected causal diagnosis

Original and scoped PostgreSQL literal keyword queries raise ProgrammingError. Actual prepared statements contain one fewer driver placeholder than parameters because the **ILIKE operand before ESCAPE** is retained as `?` by the shared converter's JSONB-operator heuristic. LIMIT converts normally. The first report incorrectly inferred LIMIT from counts alone; source inspection and the failed escape experiment corrected that attribution.

The first proposed `ESCAPE '!'` implementation still failed (7 PostgreSQL failures,15 controls passed). It was not accepted or called repaired. Its source/patch and diagnostic are retained. Changing the escape character does not affect ILIKE/ESCAPE token classification.

## Selected minimal correction

Use `ILIKE (?) ESCAPE '\'` in this PostgreSQL-only branch. Parentheses unambiguously identify the bound operand to the existing converter. Keep original bound backslash/%/_ escaping, SQLite path, simple-token FTS, owner predicate, input rejection, ordering and limits. No shared SQL translator edit.

Compared alternatives through the real method, official PostgreSQL fixture and SQLite: native `%s` operand22PASS; parenthesized `?` operand22PASS. Select parentheses to retain the current placeholder style and change only two characters. Tests cover literal plus, hyphen, wildcard characters, bang, backslash, combinations, nonmatching decoys, ordered bound limits, and existing empty/quote rejection.
