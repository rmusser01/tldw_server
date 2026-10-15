# Roleplay Resume PostgreSQL rows — TASK-13704

Task identity corrected on 2026-10-15: this PR's historical TASK-13642 is now TASK-13704; the old ID belongs to a separate task on dev. This report keeps its stable filename and original evidence.

Actual PostgreSQL append/replay failed because its mapping row was read at numeric index0 for prompt-byte usage. Both database queries now alias the unchanged aggregate as prompt_bytes and read that name. Original-parent version reads use the existing version column. Byte arithmetic, refusal threshold, conversation/live-parent identity and source ownership fences are unchanged.

The representative PostgreSQL test failed before the repair and passed afterward. The first connected run passed282 and failed20, all at positional reads in corruption/rollback fixtures. Five fixture functions now read history_admission_json or aliased conversation_count while retaining every corruption and rollback assertion. A four-case actual two-engine control verifies a four-byte emoji is refused before full payload reading at a3-byte history cap and reaches that boundary at4 bytes. A deliberate fault at the payload-read boundary isolates this SQL gate from later complete-prompt limits; both cases retain exact history and no admitted input.

Final contract and endpoint qualification passes306tests, zero skips,1851runtimewarnings,397.34s and process exit0. It includes original-parent authority, append/replay, generation/commit, stored coercion refusal and the completion endpoint quota/replay controls. The selected fixture and byte controls also passed46/46. Independent bounded review is clear. Production and test Bandit have0 findings/scan errors (intentional test assertions excluded only in the test scan); Ruff passes. Both changed production methods and the full test file meet Black formatting. The whole production file already needs formatting on remote dev; its unrelated formatting is unchanged.

ADR required: no; named row access preserves the existing frozen Resume and original-source policy, including [ADR-071](../ADR/071-conversation-source-incarnation.md). No API schema or client source changed. Normal exact-file hooks and publication are recorded in the task. Remaining whole-PR findings, final independent review, rebase and merge are open; this is not a whole-PR approval.

PR: https://github.com/rmusser01/tldw_server/pull/3229
