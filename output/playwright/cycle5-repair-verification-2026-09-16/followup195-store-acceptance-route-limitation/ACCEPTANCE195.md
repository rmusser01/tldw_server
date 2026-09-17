# UAT195 — store acceptance and explicit native-route limitation

**Verdict: UAT195 can be accepted under the existing TASK13260.133 criteria using AC2's explicit route-limitation alternative.** The repaired database behavior is independently verified on current PostgreSQL69 and SQLite67. Authenticated native HTTP search and populated exemplar selection were not verified, and must not be reported as native passes.

This is an acceptance audit, not a new implementation or a relaxation of the task. Parent owns task/tracker closure. No UI, native runtime/data, credentials, source, git, or task mutation was performed. Only this private evidence packet was written, and the existing tests ran through official disposable fixtures.

## Existing criteria and repaired behavior

`task-criteria.txt` records the official CLI task read. AC1 requires owned PostgreSQL search with absent, partial and explicit optional filters, result/count/deleted behavior, and SQLite controls. AC2 explicitly permits “relevant native acceptance or explicit route limitation,” alongside independently reviewed minimal repair, causal RED, required-PG GREEN and scoped lint/Bandit.

The retained repair, integrated as `df0fbbd73a`, types the nullable emotion/scenario placeholders with `CAST(? AS TEXT)` in the existing search branches. The original failure was PostgreSQL SQLSTATE42P18, including supplied filters because the separate IS NULL placeholders were untyped. This was not solely a default-filter failure.

Prior evidence is retained at `output/playwright/cycle5-repair-verification-2026-09-16/followup193-195-character-ownership-reviewed/`:

- `author-uat195-owned-search-red.redacted.log`: 11 PostgreSQL failures / 11 SQLite passes, 30.44s.
- `author-uat195-driver-cause*`: actual driver cause.
- `independent-REVIEW193-194-195-183.md`: independent typed-filter/source review and the combined 242 passes / zero skips; production Bandit zero findings/errors, Ruff baseline comparison zero additions. That earlier report also states its non-native limits.

## Fresh verification on current schema

**22 passed / zero skipped / zero deselected / four warnings, 23.52s, exit0.** This is the unchanged permanent suite: 11 PostgreSQL and 11 SQLite cases. It creates real characters/exemplars and exercises omitted, emotion-only, scenario-only, both and empty filters across browse and text search. The additional case covers soft-deleted exclusion, rhetorical filtering, total before pagination and no-match results.

```sh
source .venv/bin/activate &&
TLDW_UAT_EVIDENCE_LABEL=uat195-acceptance-current69 node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs \
  tldw_Server_API/tests/DB_Management/test_character_exemplar_search_backends.py -q --tb=short
```

The exact safe command and redacted output are copied into this packet. PostgreSQL was mandatory; Docker autostart was disabled; the runner used the existing owned test cluster and official `pg_database_config` isolation. No native database was queried or changed. The fixture uses the normal `CharactersRAGDB` initializer, whose current PostgreSQL head is69 (`ChaChaNotes_DB.py:756`); SQLite stays67.

Before/after hashes are identical:

| File | SHA256 |
| --- | --- |
| `chacha/character_store.py` | `93ed045b9561eb119d1ce279e643c8ebf819989f54e2e40135cb7be8a7fdc952` |
| `ChaChaNotes_DB.py` | `6089c5e0cac45fd0dd2c5253ad906b20108746e35a9189bf3934fc1eca9506b7` |
| `test_character_exemplar_search_backends.py` | `909a738fd56c5e9ed88e4d885b965cf5ec70a069f8b5ddf0e88f10be8e6c21a0` |

CharacterStore is byte-identical to the previously independently reviewed repair. This fresh run is store acceptance, not authenticated HTTP acceptance. No new implementation was made that would require new static findings attribution.

## Actual frontend and backend route inventory

The frontend `@/*` and `~/*` aliases resolve to `apps/packages/ui/src/*` (`apps/tldw-frontend/tsconfig.json:33`). The audit searched that shared UI source plus `apps/tldw-frontend` for `exemplars/search`, `search_character_exemplars` and `searchCharacterExemplars`. There were zero JS/TS matches; the exact command and negative result are in `frontend-search-receipt.json`. This is a bounded source inventory, not a claim about every possible external client.

1. **Normal Character editor:** `components/Option/Characters/CharacterEditorForm.tsx:802` offers `message_example`. `utils.ts:1530` puts it into the ordinary card payload; `hooks/useCharacterCrud.tsx:181` and the update mutation use create/update character calls. `services/tldw/domains/characters.ts:616` and `:830` target `/characters` POST/PUT. Editing that field does not create or search rows in `character_exemplars`.
2. **Persona Garden “Voice & Examples”:** `components/PersonaGarden/VoiceExamplesPanel.tsx:101` calls `listPersonaExemplars`; `services/tldw/domains/characters.ts:1008` targets `/persona/profiles/{id}/exemplars`. That is the separate persona store, not the repaired character-exemplar query. Its visible examples must not be counted as UAT195 evidence.
3. **Tracked Character chat UI:** `hooks/chat/useCharacterChatMode.ts:954` calls `streamCharacterChatCompletion`; `services/tldw/domains/chat-rag.ts:1377` uses `/chats/{id}/complete-v2`. The actual endpoint starts at `character_chat_sessions.py:6020`, prepares its character context and calls `perform_chat_api_call` at `:6497`. It does not call `select_character_exemplars` or `search_character_exemplars`. Its persona preview helper uses a separate persona-exemplar list. A successful ordinary tracked Character chat therefore does not directly accept this search repair.
4. **Explicit character-exemplar HTTP search:** `characters_endpoint.py:1378` defines `POST /characters/{character_id}/exemplars/search`, uses the authenticated owner DB dependency, and verifies the owned card. `_search_character_exemplars_hybrid_best_effort` at `:641` forwards query/filter fields to the repaired store. The schema at `character_schemas.py:327` defaults the filters to absent and embedding scores to false. The route exists, but this audit found no normal frontend control invoking it.
5. **Generic completion augmentation:** `chat.py:4668` can select character exemplars when a character context exists, the strategy is not off, and there is user text. Default strategy resolution at `:1915` is `default`; `:4707` calls `select_character_exemplars`. The selector's candidate loader calls the repaired search with omitted emotion/scenario at `persona_exemplar_selector.py:307`, then catches search errors and tries the existing list fallback (`:319`–`:333`). The separate `/characters/{id}/exemplars/select/debug` route also uses that selector. This makes the repaired query relevant to backend completion, but a successful completion alone cannot prove that search, rather than fallback, succeeded or that a populated exemplar was selected.

All inspected sources are hash-bound in `inspected-source-hashes.json`. Existing in-process exemplar endpoint integration tests provide additional automated route coverage in the retained earlier review; they are not a native HTTP/browser receipt.

## Explicit acceptance boundary

The parent's unauthenticated browser request failed authentication before it could exercise this search. It supplies no positive search acceptance evidence. The normal current UI surfaces inspected above do not expose the explicit character-exemplar search route, and no authenticated native search or native populated-selector receipt was obtained.

Accordingly, record UAT195 as **store repair accepted with explicit native-route limitation**, not “native search passed.” AC1 is verified by fresh required-PG/SQLite controls; AC2's already-approved alternative is satisfied by this concrete route inventory plus the retained independent review/static/causal evidence. No additional credentials or UI feature are necessary to meet that existing criterion. Any future authenticated external-client route qualification should be described separately and use an authorized ordinary workflow.

Automatic approval review rejected the parent's proposed session-header interception helper as unauthorized credential extraction. This audit did not retry it, capture/read/export session headers or tokens, or use an indirect workaround. That rejected helper is neither a product failure nor a reason to claim native coverage.
