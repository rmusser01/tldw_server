# UAT181 Notes bootstrap plan

## Stage1: complete causal request chain
Goal: reproduce the observed keyword/collection/Notes predecessor sequence and isolate every reached read.
Success criteria: real second constructor fails while first connection lives; valid data and caller/SQLite controls retain expected behavior.
Tests: actual three-handler bootstrap in two orders and empty/populated data; individual reads; Notes-only constructor; caller transaction controls.
Status: Complete —14 causal failures,30 controls pass,zero skips.

## Stage2: exact ownership repair
Goal: change only the four proven pure-query opt-ins.
Success criteria: same44 tests pass; adjacent keyword/collection/Notes controls, Ruff and Bandit pass; AST shows only four keyword additions.
Tests:44 bootstrap +47 adjacent +1 deleted-note-count control;92 total,zero skips.
Status: Complete.

## Stage3: independent review and native acceptance
Goal: reviewer verifies frozen source/tests; parent repeats the actual native read/replacement flow.
Success criteria: independent review clear and native acceptance without retained locks blocking replacement.
Status: Awaiting independent review/native acceptance. Production/test source is frozen. If the full bootstrap or next native acceptance fails again, stop patching and reassess architecture before another change.
