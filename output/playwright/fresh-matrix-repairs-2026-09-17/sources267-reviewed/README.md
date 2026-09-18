# Reviewed Sources PostgreSQL repair (UAT267)

The local Sources service now uses PostgreSQL-compatible table definitions, the established guarded connection and placeholder conversion, generated-ID RETURNING reads, and command row counts that preserve its active-job fence. SQLite behavior and owner predicates remain intact. No global guard, database pool or route changes.

Root independent checks pass nine actual SQLite/official PostgreSQL service cases and29 worker/API/cleanup cases, zero skips. Author checks include two PostgreSQL lifecycle cases, seven SQLite cases, eleven service cases and16 API cases. Bandit reports no production findings and55 test assertions. The sole Ruff finding reproduces on baseline.

Earlier failing attempts, including fixture-discovery and PostgreSQL query errors, are retained. Original Alice Sources UI catalogue200 remains required on a committed-source upgrade before native267/client264 close. Safe evidence uses exact bytes or lossless gzip; private credential/log inputs are excluded or explicit hash-only. No full-matrix acceptance.
