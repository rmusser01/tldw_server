# EMAIL-M0-006 sensitive logging audit — TASK-13376.2

## Scope and contract

Information-and-higher server diagnostics for synthetic file-email upload, EML/ZIP/MBOX parsing, document-like persistence, native EmailMessages writes, SQLite full-text search, native backend execution failures, and server access logs exclude raw body, sensitive header, credential, metadata, filename, title, keyword, SQL-parameter and exception-message values. Failures retain a static operation, a bounded exception class name (80 characters), numeric counts, status codes or internal numeric identifiers. No exception object, traceback or diagnostic-local dump is attached to email failure events.

The existing request/response, validation, exception propagation and transaction contracts remain intact. Shared validation and database boundary diagnostics use the same safe summaries across formats. Where existing document orchestration needs detailed non-email diagnostics, explicit email conditionals select static messages and bounded exception types.

## Findings and fixes

| Boundary | Leak demonstrated | Guardrail |
| --- | --- | --- |
| Upload and validation | Original filenames, validation issue strings, MIME/Yara metadata and exception strings | Static acceptance/rejection stages, count/byte totals and exception type |
| Archive and sanitization | Member paths, extracted output paths, sanitizer exceptions | Counts/static outcomes; type-only failure |
| Parser | Filename and exception text on parse/chunk/container/cleanup errors | Static operation and bounded exception type, no traceback |
| Chunk options | Serialized arbitrary chunk/template metadata | Option count and type-only template failure |
| Document persistence and media repository | Titles, filenames, keywords, SQL errors and transaction tracebacks | Static stage/count/internal IDs and bounded error type |
| Native DB execution, sync, FTS and version creation | SQL/parameter/error text, keyword values, traceback locals | Static operation, row count/internal IDs and exception type; propagation preserved |
| Multipart orchestration | Worker exception text, HTTP detail, setup exception and filename/path | Email-only type/status/outcome logs; response errors unchanged |
| Uvicorn access records | Raw or URI-encoded email query text and header/credential query values | Drop query on email namespace and exact media search route; preserve route/status and unrelated logs |

## Regression method

`test_email_sensitive_logging_13376_2.py` adds an actual Loguru INFO sink that renders structured extras and enables backtrace/diagnose. Four synthetic sentinels represent body, RFC email header, credential and metadata fields; injected exceptions echo all four. Real upload validation, parser, primary media repository, native SQLite email persistence and search execute. Fault injection targets the raising dependency rather than replacing the log implementation. Uvicorn records pass through the application's actual `InterceptHandler`, after app logging configuration, into a real sink. Tests assert both private values' absence and meaningful error-type/level or successful processing signals.

Client HTTPX/HTTPCore logs are excluded from the server capture because TestClient emits its outgoing query; actual Uvicorn server request lines are separately covered. Email routes and media-search requests redact queries including encoded sentinels and apostrophes. Existing access-log tests preserve audio ticket redaction, routing, status and unrelated queries.

No live Gmail, LLM, embedding model or external network calls were used. Optional embedding dispatch diagnostics use mocked adapters/providers and synthetic results only. Real PST parsing remains covered by the attachment owner's environment-dependent fixture suite, outside this logging-specific run; fake-reader parser coverage validates static high-level diagnostics. DEBUG diagnostics, third-party provider/client logs and authentication logs are outside this narrow gate.

## Verification evidence

All commands activate the shared project `.venv`, run from the isolated worktree and set `PYTHONPATH` to that worktree. Red/green captures established leaks at upload, parser, persistence, keyword/sync/execution/FTS/version boundaries, rollback, raw chunk options and actual Uvicorn request lines before safe diagnostics were applied. The retained tests prevent those regressions.

The final focused logging, pool and access-log run passed 83 tests, including two real SQLite backend schema-failure captures added after review. Those captures failed before changing the lower backend's schema diagnostic to a static, type-only event; the original DatabaseError cause remains available to callers. Evidence: `/tmp/email_sqlite_schema_red_13376.log` and `/tmp/email_sqlite_schema_green_13376.log`. The combined core suite passed 328 tests with two known real-PST fixture skips (`/tmp/email_core_regression_13376.log`).

Ruff is clean across the touched Python scope. Bandit reports zero findings/errors across touched production and executable-probe files (`/tmp/email_bandit_all_13376.json`); the final lower-schema scope also reports zero findings/errors (`/tmp/bandit_email_sqlite_schema_13376.json`). Executable synthetic probes reject optimized Python before setup, so their validation assertions cannot be removed by `-O`. No tests were disabled to obtain these results.

## Ownership

One repository owner reviews the resulting changes and release evidence. Independent source review checked the reached logging boundaries, metric contracts and preserved exception/transaction behavior; confirmed findings were reproduced and fixed before final validation.
