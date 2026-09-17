# UAT223 / TASK13260.161 — Note citation route

Frozen two-file repair: provenance.py:_build_note_route plus test_provenance.py. Backend emitted /notes/<id>, which the actual browser rejected404 when clicking the original pack2/deck10 card a33c73dc's Deep dive to source link. Existing frontend option-notes.tsx selects the note from source_ref_id on /notes. The builder now encodes that authoritative source ID as a query value and preserves nonempty mapping/string locators. A locator cannot replace source_ref_id. Other production AST nodes are identical to baseline; media/message targets and source/citation metadata unchanged. This preserves locator data, not a claim of text-anchor scrolling.

Causal RED:12 failures/9 deselected on prior source; all failures demonstrate unsupported route paths. Cases cover two IDs (ordinary and delimiter/Unicode-bearing), missing/empty/mapping/string locators and attempted source_ref_id replacement. Three existing full response expectations are updated only to the correct URL; no assertion removed. GREEN:68 passed/0skip/4 warnings34.22s with official PostgreSQL required, across provenance, real citation response, pack response and endpoint suites. Source/test remain frozen after this run. Ruff0; Bandit production/test0 findings/errors (testB101 excluded); both files compile, scoped whitespace clean.

Command, after activating .venv:

```
TLDW_UAT_EVIDENCE_LABEL=uat223-route-green node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/StudyPacks/test_provenance.py tldw_Server_API/tests/StudyPacks/test_citation_response_timestamps.py tldw_Server_API/tests/StudyPacks/test_study_pack_response_timestamps.py tldw_Server_API/tests/StudyPacks/test_study_pack_endpoints_api.py -q --tb=short
```

Baseline source/test, causal output, green redacted log, static receipts, owned.patch and exact review snapshots/manifests are retained here. The native original404 remains under .tmp/uat198-181-native-20260917/alice220-deep-dive-{open,settled,events}.txt. Independent review and same-original-card native source click remain pending. Native API82583 still runs7acc8b001a without this fix. Task222's pending ChaCha ownership changes are outside this source/test unit and were not present during the68-case run.
