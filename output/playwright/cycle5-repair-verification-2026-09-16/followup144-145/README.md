# PostgreSQL initialization and bootstrap-fixture repairs

UAT144 honors explicitly selected PostgreSQL during normal single-user initialization. UAT145 corrects three obsolete direct-user-write test fixtures using the existing canonical helper; all31 assertions remain unchanged.

Combined42tests pass with0skips; independent12/12 and4/4 required-PG checks pass. Production Ruff/Bandit0; existing-test Ruff0new findings. Earlier failures and the restricted-host reachability attempt remain separately labeled.

Actual normal AuthNZ CLI initialization on a second entirely fresh official-fixture profile exits0, without fallback SQLite or application test flags. Subsequent full API startup exits3 on separate UAT146: the content validator requires deliberately removed legacy media policies. These results do not certify native PostgreSQL workflows or a full UAT. Fixture roles are privileged; database-level tenant isolation is not inferred.

The manifest records exact original/retained hashes and a scan against private UAT credentials, JWT patterns and credential-bearing PostgreSQL URLs. No secrets or runtime profiles are included.
