# UAT240 / TASK13260.182: actual Study outcome metrics

Source is frozen in source-freeze.json. Three production expressions change in ChaChaNotes_DB.py: daily lapses count stored was_lapse, session increments and reconstruction count non-lapse reviews with rating!=0. Scheduler, owner/date/deck filters, transaction behavior and response shape are unchanged.

Canonical was_lapse is NOT NULL with false default. Historical outcomes remain authoritative, including intentionally false values; no fabricated rating fallback or history rewrite. Again0 in new/learning queues is incorrect recall but not yet a lapse of learned material. All current schedulers accept API-only1/4 as recall alongside Hard2/Good3/Easy5, and tests preserve this actual behavior. The old legacy SM2 helper's threshold does not replace current scheduler outcomes.

Causal RED28fail/45pass,0skip: actual SQLite/PostgreSQL review→analytics with both schedulers, new/review, all six accepted ratings, stored historic outcomes and repeated scheduled Hard. First unchanged-test GREEN73pass/0skip. Extended historical session reconstruction, UTC date boundaries and actual PostgreSQL second owner add9controls. Canonical review bits and saved card lapses are asserted before analytics; session live/rebuilt counters are verified.

Adjacent initial6fail/208pass,0skip exposed five session/one HTTP fixture assumptions. Four tests using1 to represent incorrect recall now use actual Again0, retaining their original aggregate/race assertions. The explicit Hard2 reconstruction test now expects3correct recalls. The HTTP lapse test graduates a real card via Easy5 before Again0, counts all3actual reviews, and preserves mean answer time with setup's absent duration. No tests were disabled or product guards changed. Retained initial receipts explain the transition.

Final combined296pass/0skip/4existing warnings,251.14s: analytics82, existing sessions and endpoint214. Official isolated PostgreSQL required. These are controlled scheduler/data tests, not real-model/native UAT acceptance.

Static checks on all4touched Python paths: compile clean; Bandit0 findings/parse errors (tests omit only B101 assert checks); Ruff0 for production/analytics/sessions and14existing identical endpoint-test diagnostics. Baseline/current logical-path checks and JSON comparisons are retained. No production file swapping; no held profile, runtime, browser, provider or native database changed.

Independent review and original native Hard/re-rate statistics remain pending. Design compatibility paragraph is in Docs/Design/2026-09-17-fresh-matrix-repairs-231-246.md.
