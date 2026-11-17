# B07 Watchlist arXiv abstract extraction repair

Associated Backlog task: TASK-13260.278.18.83.45 (root recorded this unit before edits).
Workspace: uat-native-regenerate-20261008/tldw_server2, c700e42.
Scope: shared direct Trafilatura strategy and default strategy-selection boundary; new causal extraction/Watchlist tests only.
Root owns runtime, profile, browser, actual provider calls and live acceptance.

## Diagnosis and design

The actual first Watchlist RSS run found/ingested 50 entries without source errors, but two saved arXiv entries and the briefing contained arXivLabs text. The blocking article scraper clears custom route handlers and uses the shared generic extraction pipeline. The default cluster extractor can select the dominant unrelated body before Trafilatura is reached; the generic Trafilatura fallback can select the same unrelated text. Successful article content is then preferred over the feed summary by the RSS consumer. The connected causal qualification established both paths. Public source reads confirmed distinct abstracts in citation_abstract and blockquote.abstract.mathjax and the same unrelated Labs section.

For canonical URL-parsed arxiv.org /abs/ paper paths only, prefer the publisher abstract metadata or explicit abstract block through the installed HTML parser. Only default extraction with no explicit strategy, handler, custom fallback or schema/LLM/regex/cluster settings moves the existing Trafilatura step before generic clustering. Explicit callers retain their original order. Preserve generic extraction when no usable abstract exists. Keep original URL, title/author/date, ContentMetadataHandler formatting, fetch/redirect/egress, deduplication, owner scope and pipeline policies intact. No body blacklist, new dependency, network integration or model change.

ADR check: ADR required: no. Governing [ADR-026](../ADR/026-security-outbound-egress-and-ssrf-policy.md) retains the shared outbound policy; this is a content selection correction within the existing extraction contract, with no changed API, persistence, authority or architectural rule. Parent task ADR note is root-owned.

Separate read-only findings: automatic briefing metadata has provenance/source_counts but no source_count or immutable report_snapshot_path; report UI therefore renders zero source_count and Live provenance only. Overview counts every active job lacking next_run_at as needing schedule repair, including intentionally manual-only monitors. Neither is changed by this extraction unit.

## Stage 1: Reproduce at the connected extraction boundary
**Goal**: Demonstrate incorrect abstract content using real Trafilatura and the blocking Watchlist consumer.
**Success Criteria**: A normal configured causal test fails on unchanged production for the wrong content; generic/fallback and guard controls are meaningful.
**Tests**: Synthetic HTML shaped like the publisher page; citation and block abstract paths; real blocking article → Watchlist consumer → owned SQLite item/Collections persistence; exact source URL; ordinary/full-paper/non-arXiv/no-abstract fallback; existing egress and redirect guard lifecycle.
**Status**: Complete

## Stage 2: Minimal shared selection correction
**Goal**: Select the supported publisher abstract only for canonical arXiv abstract pages.
**Success Criteria**: Causal tests pass, ordinary body/metadata/fallback behavior remains, no policy or ingestion change.
**Tests**: New targeted module and genuinely affected extraction/orchestration/fetcher consumers.
**Status**: Complete

## Stage 3: Static qualification and independent/live review
**Goal**: Freeze a reviewable source/test unit for root.
**Success Criteria**: Scoped syntax/Ruff and baseline-current Bandit including tests; independent review; root's later actual recovery under its declared live cap.
**Tests**: Activated Python 3.12 normal pytest configuration; outer 600 seconds and normal per-case 300; existing primary environment for static Bandit only.
**Status**: In Progress

## Attempt and evidence ledger

Maximum three failed issue attempts, then stop/reassess. Fixture/setup/operator failures are distinct from causal red and never establish acceptance. Capture normal exits, elapsed time and bounded failures in memory; no proof/log/report files. No second actual Watchlist run or live request by this agent.

Pre-test source capture: first output-token cap truncated source JSON and the in-memory JSON guard rejected it. A corrected read retained complete pre-unit bytes; no test or source mutation occurred.

Attempt 1: unchanged production causal RED, natural exit 1, 3.661s outer / 1.46s pytest; 7 failed, 12 passed, 59 warnings. Failures are actual generic Labs content selected instead of publisher abstract, including blocking Watchlist consumer and owned SQLite run. Twelve generic/fallback/path/egress controls passed. No setup failure. Before green, connected schema read found Collections stores content_hash rather than a content column; the new downstream assertion was corrected to exact summary/URL/source/hash, retaining Watchlist summary/content oracles. No extra baseline replay. Scoped baseline Bandit 0 findings/0 errors (existing Python3.11.13/Bandit1.9.3 AST scanner); Ruff baseline 0 findings after normal-cache sandbox denial was resolved through authorized escalation.

Attempt 2: partial-source qualification natural exit 1, 4.536s outer / 1.73s pytest, 2 failed, 17 passed, 59 warnings. The connected blocking consumer and owned SQLite persistence still received Labs; all direct abstract controls now passed. Source trace confirms default cluster strategy succeeds before the final Trafilatura step. Root reviewed/accepted the connected completion: only default extraction with no explicit strategy/handler/custom fallback/schema/LLM/regex/cluster settings and a usable canonical publisher abstract moves the existing Trafilatura step first. Semaphore, cancellation, retry, trace/finalize and formatting remain canonical. This second source correction does not reset the issue cap; retain 17 passes and qualify only the two connected failures, then first execution of every new explicit-override/fallback control plus genuinely affected old pipeline consumers. Controls authored before the connected production edit. Scope now includes extraction/pipeline.py in addition to the direct strategy. Its exact pre-unit bytes and zero Ruff/Bandit baseline retained in memory.

Connected attempt 3: only the two previously failed blocking/owned-run IDs passed, natural exit 0, 14.229s outer / 5.65s pytest, 58 warnings. Retained the other 17 passing cases without replay.
First affected phase: 43 passed, 29 deselected, 6 warnings, natural exit 0, 5.152s outer / 2.53s pytest. Scope was the 11 newly authored real pipeline controls plus 32 existing extraction/default-order/cancellation/copy/metadata and blocking-caller contracts; exact selector retained in memory and below. No provider/network/runtime replay.
Total current evidence: 30 distinct new cases by split qualification (17 + 2 + 11), plus 32 distinct existing affected cases = 62 distinct passes. There is no single combined 30/62-case green execution claim.

Affected files: test_phase4_extraction_pipeline.py and test_phase4_article_compatibility.py; existing test bytes unchanged. Selector:
`test_real_pipeline or test_strategy_order_normalization or test_default_regex or test_explicit_empty_order or test_disabling_llm or test_pipeline_copies_handler or test_pipeline_deep_copies or test_canonical_pipeline_builds_dependencies or test_pipeline_cancellation or test_pipeline_cluster_observes or test_direct_trafilatura or test_blocking_`

Static qualification: three touched Python paths compile/AST PASS without writing compilation artifacts; current Ruff 0 findings, both production baselines 0 findings. Existing primary Python3.11.13/Bandit1.9.3 static scanner: both production baselines 0 findings/0 errors; current two production + new tests 0 findings/0 errors. The new test module initially had only 42 B101 pytest-assert findings, resolved through narrow per-assert explanatory comments; AST equality preserved. No other security code was reported. No dependencies installed.

Self-review read complete two changed source hunks/functions and all 30 new case bodies/fixtures. Source identity, generic/full-paper/noabstract fallback, every explicit override, formatter and canonical semaphore/retry/cancellation/finalize paths stay. Tests supply external HTML/feed/policy/executor/embedding boundaries and use actual parser, Trafilatura, strategy pipeline, blocking scraper, Watchlist consumer and owned SQLite DB abstractions. This does not qualify actual arXiv fetch/redirect DNS or a full authenticated API/UI request, PostgreSQL, model behavior or actual live recovery. Existing incorrect saved items are not rewritten/backfilled. Independent source review and root's controlled live qualification remain OPEN; Stage 3 must stay In Progress.
