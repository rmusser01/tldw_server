# Reviewed authenticated content scope repair (UAT257/259)

Configured single-user key/cookie and cached authentication paths now activate the existing content scope from canonical principal claims. Existing authority-matched session roles survive; stale authority does not carry a prior role. No SQL, RLS, permissions, schema or provider settings changed.

Actual permission-first endpoints reproduced Media delete404 and empty RAG contexts under restricted PostgreSQL. Focused final38 and independent72 SQLite/PostgreSQL tests pass with zero skips, including actual cookie mint/validation, request ordering, cache authority, ownership denial, restore and no-match controls. Initial fixture mistakes and failed checks are retained in test receipts.

Broad adjacent authentication run had95pass/1failure before a request; the failure also reproduces with both original auth modules from baseline278. Independent baseline attribution assigns it to separately tracked virtual-key SQL guard defect262. Its repair/broad rerun remains pending; native Rowan QA and Trash/Restore acceptance remain pending explicit source upgrade.

Ruff reports one unchanged SIM110. Bandit production8unchanged/0new; test114B101 assertion findings. Compile passes. No full-matrix acceptance. Payloads are original bytes, gzip where required, scanned against known credentials/JWT patterns. Unsafe candidate outputs are kept local hash-only; private runtime logs are excluded. External review references are local hash-only where omitted, not a standalone replay guarantee.
