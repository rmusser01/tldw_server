## Stage 1: Regression Tests
**Goal**: Capture the outbound-policy bypasses before changing production code.
**Success Criteria**: Tests fail for direct workflow `pdf_url` download and tokenizer `_http_post` bypassing the central HTTP policy.
**Tests**: Focused pytest cases in `tldw_Server_API/tests/Workflows/adapters/test_research_adapters.py` and `tldw_Server_API/tests/Writing/test_tokenizer_resolver_unit.py`.
**Status**: Complete

## Stage 2: Workflow Research HTTP Policy
**Goal**: Route project-owned research adapter HTTP calls through the central HTTP helper layer.
**Success Criteria**: Direct `pdf_url`, PubMed, Semantic Scholar, patent, and DOI requests use central egress/proxy/trust-env behavior while preserving existing response shapes.
**Tests**: Research adapter regression tests plus existing focused research adapter tests.
**Status**: Complete

## Stage 3: Tokenizer Resolver HTTP Policy
**Goal**: Replace raw tokenizer/counting `requests.post` use with the central sync HTTP helper path.
**Success Criteria**: Tokenizer provider calls inherit central egress denial, redirect checks, proxy validation, and `trust_env=False` defaults; explicit local provider behavior remains testable.
**Tests**: Tokenizer resolver regression tests plus existing unit tests for remote count/tokenizer adapters.
**Status**: Complete

## Stage 4: Verification And Task Record
**Goal**: Prove the remediation and record it for repeatable audit follow-up.
**Success Criteria**: Focused pytest passes, `git diff --check` passes, Bandit runs on touched production files, and `TASK-12146` records final verification.
**Tests**: `python -m pytest` focused files, `git diff --check`, and Bandit on touched production files.
**Status**: Complete
