## Stage 1: Regression Coverage
**Goal**: Pin the accepted legacy WebSearch review findings with focused tests.
**Success Criteria**: New tests fail against the current implementation for secret logging, all-provider failure propagation, non-interactive review behavior, bounded evidence payloads, and runtime validation.
**Tests**: Focused WebSearch unit tests under `tldw_Server_API/tests/WebSearch/unit/`.
**Status**: Complete

## Stage 2: Legacy WebSearch Hardening
**Goal**: Patch `Web_Search.py` without changing the live `Web_Scraping` endpoint path.
**Success Criteria**: Provider logs redact API keys, all-provider failures surface as structured errors, user review does not call `input()`, evidence omits full article text by default, and DuckDuckGo uses explicit runtime validation.
**Tests**: New focused WebSearch unit tests plus existing legacy sanitizer tests.
**Status**: Complete

## Stage 3: Documentation Cleanup
**Goal**: Make `core/WebSearch/README.md` accurately describe this folder as legacy/non-routable.
**Success Criteria**: README no longer advertises provider support as the active endpoint implementation.
**Tests**: Review/diff check.
**Status**: Complete

## Stage 4: Verification and Task Finalization
**Goal**: Verify the touched scope and record results in Backlog.
**Success Criteria**: Focused pytest, compile check, Bandit, and `git diff --check` complete; Backlog task records verification and final summary.
**Tests**: Focused pytest command, compileall, Bandit on touched app scope, diff check.
**Status**: Complete
