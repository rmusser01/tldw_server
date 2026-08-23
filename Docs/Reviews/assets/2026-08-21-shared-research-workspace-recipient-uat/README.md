# Shared Recipient Workspace UAT Evidence

Final live acceptance ran against the multi-user SQLite backend, local Next.js WebUI, and a healthy local OpenAI-compatible provider through explicit Chrome CDP only.

`evidence.json` records the passed `final32-fix2-1787446794-16413` run: all 15 canonical acceptance checks passed, both bounded transition ledgers and the strict interaction ledger closed without unexpected operations, every transition operation records its exact allowed and observed statuses, Chats settings requests were bounded to two `200` responses, and the idempotency race recorded only `200`/`409` statuses with replay-equivalent turn hashes. The provider record is truthfully `local-llm` / `Qwen2.5-0.5B-Instruct`.

The loopback-only transparent local forwarding probe observed three unchanged provider requests. It validates body bounds, JSON, request count, byte identity, sentinels, mutations, and tool/function payloads before contacting the loopback model. The evidence retains only bounded counts, hashes, and booleans; raw prompts were neither persisted nor logged.

The screenshots separately cover the desktop shared-source surface, grounded cited answers, the mobile two-tab core surface, the mobile source-preview sheet, and the fail-closed revoked state. They were visually inspected for overflow, overlap, compact hierarchy, citation readability, and absence of owner or recipient-local content after revocation.

The committed evidence JSON contains no credentials, prompt/answer bodies, cleanup metadata, or machine paths. Prompt and answer bodies are also absent from the external execution log; the visible transcript is intentionally retained in screenshots as acceptance evidence. The run-specific cleanup manifest and execution log remain outside the repository with mode `0600`.
