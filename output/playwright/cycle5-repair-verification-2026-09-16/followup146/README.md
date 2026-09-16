# Current PostgreSQL startup-policy validation

UAT146 replaces obsolete startup policy names with the six current policies installed by canonical schema setup. RLS configuration and policy predicates are unchanged.

Author56tests and independent30tests pass with0skips (overlapping runs, not summed). Real normal single-user and multi-user APIs complete startup and return health200 with the frozen source hashes. Checkpoints record the starting revision, exact changed source and log offsets; repaired-start logs exclude the earlier146 failure. PostgreSQL AuthNZ initialization also passes in both modes without test flags or fallback SQLite.

Startup exposed separate147Chat/Notes deadlock,148MCP Media writable probe failure and149Collections backfill failure. Both owned APIs were stopped after evidence capture; profiles and official fixture holders remain. Health200 and this bounded policy repair do not certify those subsystems, browser workflows or a full UAT. Privileged fixture roles do not certify DB-level tenant isolation.

Production Bandit0 and Ruff0new findings; baseline warnings remain documented. Evidence is checked against known private UAT values, JWT patterns and credential-bearing URLs.
