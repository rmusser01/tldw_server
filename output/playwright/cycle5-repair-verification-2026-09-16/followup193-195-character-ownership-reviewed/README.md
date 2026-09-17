# Character ownership, scoped PostgreSQL names, and exemplar search

Integrated df0fbbd73a; test-only historical migration corrections48f89447fc. Author96focused+146adjacent+5OSCE and independent242+5 pass, all zero skips. The reviewer caught a stale fresh-head67 test expectation; its causal1failure/4controls and correction are retained. PostgreSQL schema68 uses per-owner names; SQLite remains67. Real cold cache/default/factory controls preserve factory ownership and immutable snapshots.

The original native role was superuser/BYPASSRLS. Existing forced RLS rejects foreign SQL under a verified ordinary role; no ordinary-role leakage is claimed. New store owner predicates protect the accepted privileged configuration. Native character/chat and search acceptance remains pending. The shared ChaCha count repair197 is separately attributed and excluded from this independent review. UAT183 test-only extension is complete; prior162+4 evidence remains valid.

Known runtime credentials and JWT/PEM patterns scanned, zero matches. Original evidence remains unchanged.
