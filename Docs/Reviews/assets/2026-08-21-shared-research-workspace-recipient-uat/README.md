# Shared Recipient Workspace UAT Evidence

Final live acceptance ran against the multi-user SQLite backend, local Next.js WebUI, and a healthy local OpenAI-compatible provider through explicit Chrome CDP only.

`evidence.json` records a passed final27 run: all 15 acceptance checks passed, the strict request ledger has no failures, Chats settings requests were bounded to two `200` responses, and the idempotency race recorded only `200`/`409` statuses with replay-equivalent turn hashes. The provider record is `local-llm` / `Qwen2.5-0.5B-Instruct`.

The screenshots cover the desktop shared-source surface, grounded cited answers, mobile source preview, and fail-closed revoked state. They were visually inspected for overflow, overlap, compact hierarchy, citation readability, and absence of owner or recipient-local content after revocation.

No credentials, prompt/answer bodies, cleanup metadata, or machine paths are committed. The run-specific cleanup manifest and redacted execution log remain outside the repository with restrictive permissions.
