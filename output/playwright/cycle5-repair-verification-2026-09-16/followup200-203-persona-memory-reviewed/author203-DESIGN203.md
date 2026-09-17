# UAT203 / TASK13260.141 — persona memory count row

Separate from200. The private officialPG probe bypassed archive/deleted predicates and confirmed successful COUNT query then KeyError0 at count_persona_memory_entries. SQLite returns the expected count. Parent approved exactly one backend-specific count field selection; SQL, filters, transaction flags, empty fallback and return count semantics unchanged.

Before production: permanent realPG/SQLite empty/populated, all archived/deleted combinations, owner/persona/type controls. Minimal GREEN selects PostgreSQL row['count'] and retains SQLite row[0]. Shared predicate/read-scope200 change is a prerequisite and separately attributed. Static/security and independent review before parent integration/native.
