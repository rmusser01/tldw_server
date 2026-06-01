# Text2SQL

Text2SQL turns natural-language questions into guarded, read-only SQL execution against registered data sources. The package centralizes connector lookup, canonical source registration, SQL policy checks, execution budgeting, and service-level orchestration used by the Text2SQL API and RAG database retrieval paths.

## Start Here

- `service.py` contains the core Text2SQL service orchestration.
- `sql_guard.py` enforces read-only SQL policy and query normalization.
- `executor.py` defines the executor protocol and SQLite read-only executor.
- `source_registry.py` normalizes canonical source identifiers.
- Related API surface: `app/api/v1/endpoints/text2sql.py`.
- Related tests: `tests/Text2SQL/`.

## Responsibilities

- Register and resolve Text2SQL sources through a connector registry.
- Generate or accept SQL, then guard it as a single read-only statement before execution.
- Apply result-size and query-budget constraints to returned rows.
- Normalize source identifiers so endpoint, RAG, and ACL logic agree on source names.
- Execute SQLite queries through a read-only executor boundary.
- Surface policy violations as explicit service errors instead of silently mutating SQL.

## Module Map

- `service.py` - high-level Text2SQL orchestration.
- `sql_guard.py` - read-only statement validation, normalization, and policy violations.
- `executor.py` - executor protocol and SQLite implementation.
- `connectors.py` - connector registry and source connector types.
- `source_registry.py` - canonical source registration and normalization helpers.

## How It Connects

- `app/api/v1/endpoints/text2sql.py` exposes query and source APIs.
- `app/api/v1/schemas/text2sql_schemas.py` defines request and response contracts.
- `app/core/RAG/rag_service/database_retrievers.py` and the unified RAG pipeline use Text2SQL source registry behavior for database retrieval.
- Security tests cover RBAC and source ACL behavior around the endpoint surface.

## Extension Points

- For a new data source, add connector behavior in `connectors.py`, register a canonical source in `source_registry.py`, and add source-registry tests.
- For SQL policy changes, update `sql_guard.py` and `tests/Text2SQL/test_sql_guard.py`.
- For execution changes, update `executor.py` and result budgeting tests.
- For endpoint contract changes, update `text2sql_schemas.py`, `text2sql.py`, and security/ACL tests together.

## Testing

- `tests/Text2SQL/test_connectors.py`
- `tests/Text2SQL/test_source_registry.py`
- `tests/Text2SQL/test_sql_guard.py`
- `tests/Text2SQL/test_sql_executor_read_only.py`
- `tests/Text2SQL/test_result_budgeting.py`
- `tests/Text2SQL/test_service.py`
- `tests/Text2SQL/test_imports.py`
- `tests/Security/test_text2sql_rbac_and_acl.py`

## Gotchas

- SQL guarding must fail closed. Do not bypass `sql_guard.py` for endpoint or RAG execution.
- Source names must remain canonical across endpoint, RAG, and ACL layers.
- Result budgeting is part of the resource-control boundary and should be tested when query execution changes.
