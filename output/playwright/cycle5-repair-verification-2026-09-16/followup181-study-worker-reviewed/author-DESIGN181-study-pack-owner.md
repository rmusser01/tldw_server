# UAT181 / TASK13260.118 — bounded StudyPack job ownership

Root approved one explicit independent chacha_operation around handle_study_pack_job database acquisition through its existing finally. Use a literal with block inside async function, not a synchronous ContextDecorator on async code. Preserve validation before acquisition, public generation service caller ownership, Media cleanup, existing commits/results/errors and SDK behavior. No shared DB/migration edits.

Causal retained probe: real cached accessor→actual service→asyncio.to_thread note read→controlled model failure leaves one open IDLE executor checkout after event-loop finally. Private outer-owner control returns it; Suggestions same-thread missing-session control also returns its checkout. This is a checkout leak, not a proven lock, pool-exhaustion or native StudyPack defect.

Permanent tests before production: official requiredPG with actual accessor/service/thread; model/source failures; success via fake model JSON and actual persistence; repeated cached jobs; cancellation while backend source query is in flight; unrelated PG outer pending-write decisions; SQLite actual-success/failure controls. Existing StudyPacks worker/service controls retain validation, regeneration rollback and Media-failure cleanup coverage. Any newly exposed persistence defects are reported separately.

Stages:1 tests/RED complete (causal and harness receipts retained);2 minimal owner block and separate197 prerequisite complete;3 authorGREEN/static complete, source frozen for independent review. Root owns integration/native/tracker.
