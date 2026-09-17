# UAT220 / TASK13260.158 — Cited-card response timestamps

## Result and scope

Frozen two-file unit: `tldw_Server_API/app/api/v1/schemas/study_packs.py` and new `tldw_Server_API/tests/StudyPacks/test_citation_response_timestamps.py`. Relative to the independently released UAT219 schema, production changes consist of one six-line before-validator in **FlashcardCitationResponse**, converting actual datetime values with `isoformat()`. The datetime import already existed. Every other top-level schema AST node matches the released219 baseline. No endpoint, worker, database, source lookup, ownership or public field type changed.

Permanent causal RED: **5 failed /12 passed /0 skipped**,10.12s. The actual PostgreSQL assistant routes failed for a cited card both with and without StudyPack membership; UTC/offset/naive schema cases account for the other three failures. SQLite, empty-citation, owner exclusion, string/null and invalid-type controls passed. Final combined GREEN: **97 passed /0 skipped /4 warnings**,46.58s, on final frozen bytes. The configured pytest output does not expand warning details.

Frozen owned-manifest SHA256: `01d8c4ac5eb3895718064a8a6fd7c5ee78ecd3171bf066261f281a549613e03e`.

## Cause and approved boundary

The real PostgreSQL citation table returns created_at/last_modified as Python datetime values. `FlashcardProvenanceStore.list_citations` preserves them; the actual flashcard assistant endpoint returns the same citation in both `citations` and `primary_citation`. FastAPI validates these against Optional[str] fields in FlashcardCitationResponse, producing four response validation errors. The prior private actual-PG probe isolated this from StudyPackSummaryResponse by creating no StudyPack membership; its SQLite control passed.

The repair follows the existing Deck, Study, Review and UAT219 Summary schema convention. Only datetime objects are converted. UTC, non-UTC offsets and naive datetime values retain their representation. Existing strings and null remain exact, and integers, objects, lists and date-only objects still fail normal Pydantic validation. No schedule, row, source identity, owner, version or stored timestamp is rewritten.

## Permanent controls and honest limits

The new17 tests cover:

- Official disposable PostgreSQL and ordinary SQLite content, using a real persisted note/deck/card/citation and actual FastAPI GET `/api/v1/flashcards/{uuid}/assistant`. Both citation list and primary citation return exact expected timestamps and source/card/owner/version identities. Tests cover both citation-only and cited-StudyPack cards; nested pack metadata remains compatible with219.
- Full card/citation row equality before and after reads, unchanged note content and unchanged StudyPack row when present.
- Uncited cards remain200 with an empty list and null primary citation. A different authenticated actor using its real selected database receives404 and cannot change the owner’s card/citations.
- UTC, offset, naive, exact string and null schema values; unchanged non-time fields and idempotent revalidation. Four unrelated invalid input types are rejected for both timestamp fields.

The reused219 `pack_api` fixture uses the official PG configuration and normal SQLite content; JobManager explicitly uses a disposable SQLite database. Authentication uses established dependency overrides. These are real route, serializer and storage controls, not native authentication acceptance or model inference. No original native card/job was read or changed by this agent. Parent-owned same cited-card readback remains pending after the combined restart.

## Exact commands and retained receipts

Causal RED:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat220-causal-red node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/StudyPacks/test_citation_response_timestamps.py -q --tb=short
```

Final97; use a fresh evidence label for independent review:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat220-green node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs \
 tldw_Server_API/tests/StudyPacks/test_citation_response_timestamps.py \
 tldw_Server_API/tests/StudyPacks/test_study_pack_response_timestamps.py \
 tldw_Server_API/tests/StudyPacks/test_study_pack_schemas.py \
 tldw_Server_API/tests/StudyPacks/test_study_pack_endpoints_api.py \
 tldw_Server_API/tests/Flashcards/test_study_response_timestamp_contract.py \
 tldw_Server_API/tests/Flashcards/test_review_response_timestamp_contract.py -q --tb=short
```

The explicit-Jobs helper preserves mandatory official PG configuration while removing globally injected JOBS_DB_URL, preventing older SQLite Jobs fixtures from being silently redirected. Required PG cases actually ran; none were skipped.

The packet preserves original sibling probe receipts, full permanent RED/command/test bytes, final GREEN/command, the released219 schema baseline and release receipt, final snapshots,219-based `owned.patch`, owned/evidence manifests, and static JSON. The RED test was formatted afterward and its fixture import made an explicit re-export to satisfy Ruff; assertions are unchanged.

## Static checks and attribution

Ruff0 findings on both final files. Initial F811 fixture-import findings are retained in `ruff.json`; explicit fixture re-export resolves them in `ruff-final.json`. Bandit production and tests:0 findings/0 parse errors; only pytest B101 excluded for the test file. Both files compile; no trailing whitespace; AST comparison proves only FlashcardCitationResponse changed relative to219. Exact source hashes are stable in `verification.json` and `owned-manifest.json`.

UAT219 was independently clear and explicitly released before this production edit; its Summary class and test are unchanged. UAT220 owns only the additional citation validator and its new regression. No task, tracker, browser, runtime, provider, staging or commit actions were taken. Independent source review and native acceptance remain parent-owned.
