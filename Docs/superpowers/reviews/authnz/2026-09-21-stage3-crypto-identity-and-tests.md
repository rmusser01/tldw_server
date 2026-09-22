# Stage 3 — Key derivation, credential verification, and the test-tree boundary

Date: 2026-09-21. Read-only.

## Scope

The credential core of AuthNZ — HMAC/KDF key derivation, API-key hashing and verification, session and
MFA encryption, and the base64 token codec (seed cluster **C1**) — plus the five-way AuthNZ test-tree
split and what it does to dual-backend coverage. Closes with the synthesis for the module.

## Code Paths Reviewed

Key derivation:

- `crypto_utils.py` (246 LOC) — module docstring at `:1-8`: *"Currently exposes a uniform HMAC key
  derivation routine to avoid drift between JWTService, APIKeyManager, CSRF, and SessionManager."*
  `_derive_hmac_key_from_source (67-101)` uses PBKDF2-HMAC-SHA256, 100,000 iterations
  (`_HMAC_KDF_ITERATIONS`, `:37`), with a **per-secret domain-separated salt**
  (`_derive_hmac_kdf_salt`, `:34-36`) plus a legacy fixed-salt candidate (`_HMAC_KDF_SALT_LEGACY`,
  `:22`). Memoized in a 64-entry LRU keyed on a fingerprint (`:60-102`), with the rationale and the
  measured cost documented at `:42-58`.
- `derive_hmac_key_candidates (142-235)` — the ordered candidate list; emits both a v2 and a
  legacy-salt key per source.
- Adopters: `jwt_service.py:19-20` (uses at `:141, :732, :734, :779, :793, :795`),
  `api_key_manager.py:32-33` (`:84, :500, :502`), `csrf_protection.py:20` (`:106`),
  `mfa_service.py:32` (`:123, :178`), `key_resolution.py:19` (`:99, :113`),
  `repos/telegram_runtime_repo.py:13` (`:39`), and outside the module
  `api/v1/endpoints/chat.py:307` and `api/v1/endpoints/embeddings_v5_production_enhanced.py:82`.
- Bypassing copy: `session_manager.py:_derive_secret_key_candidates (459-505)`.

Credential verification:

- `api_key_crypto.py` (135 LOC) — `parse_api_key (32-51)`, `kdf_hash_api_key (54-74)`,
  `verify_kdf_hash (78-106)`, `is_kdf_hash (109-111)`, `_b64encode (114-115)`,
  `_b64decode (117-119)`. `API_KEY_KDF_ITERATIONS = 210_000` at `:13`.
- `api_key_manager.py:_verify_new_format_key (508-535)`, `_verify_legacy_key (537-551)`,
  `hash_candidates (496-506)`.
- `key_resolution.py:resolve_api_key_by_hash (55-144)`, `_compute_legacy_hmac_digests (29-52)`.
  Callers: `llm_budget_middleware.py:189`, `llm_budget_guard.py:43`,
  `api/v1/endpoints/authnz_debug.py:96`.
- `jwt_service.py:hash_password_reset_token (764-787)`, `hash_password_reset_token_candidates (789-806)`
  — 200,000 iterations each, the latter once **per key candidate**. Reached from
  `api/v1/endpoints/auth.py:2678` on the unauthenticated `POST /reset-password` route.

Encryption at rest:

- `session_manager.py:_init_encryption (230-238)`, `_get_or_create_encryption_key (239-457)` — a
  218-line function; Fernet candidate list at `:235`; re-entered lazily from `encrypt_token (911)` and
  `decrypt_token (926)`.
- `mfa_service.py:_ensure_cipher_candidates (121-133)`, `_decrypt_secret (149-161)` — the same
  rotation-candidate shape, but correctly sourced from `derive_hmac_key_candidates`.
- ADR-027 compliance: `user_provider_secrets.py:9-10, 104, 116, 120` and
  `admin_webhook_secrets.py:8-9, 49, 69` both use
  `Security.crypto.{encrypt,decrypt}_json_blob_with_key`. **This is correct and is not a finding** —
  ADR-027 names exactly these two as expected consumers of the shared `_enc: aesgcm:v1` envelope, and
  they comply. `grep -rn "AESGCM|aesgcm"` across `core/AuthNZ/` returns no local envelope format.

## Tests Reviewed

Located by import-grep.

| Test file | What it protects | Downgrades risk? |
| --- | --- | --- |
| `tests/AuthNZ/unit/test_api_key_crypto.py` | 4 tests: `parse_api_key` happy/legacy/edge, and one KDF round-trip. `verify_kdf_hash` is exercised only at `:43-44` with a well-formed hash, right key and wrong key | Partly — the happy path is covered; **no malformed stored hash is ever passed** |
| `tests/AuthNZ/unit/test_api_key_manager_validation.py` | `APIKeyManager` validation paths | Yes for the manager |
| `tests/AuthNZ/unit/test_key_resolution.py` | `resolve_api_key_by_hash` | Partly — see authnz-10; it does not assert parity with `APIKeyManager` |
| `tests/AuthNZ/unit/test_crypto_utils_key_cache.py` | the memoization at `crypto_utils.py:60-102` | Yes |
| `tests/AuthNZ/unit/test_crypto_utils_public_keys.py` | that public keys are excluded from derivation | Yes |
| `tests/AuthNZ/unit/test_security_guards.py` | production fallback-key guard | Yes |
| `tests/AuthNZ/unit/test_session_manager_configured_key.py` | explicit `SESSION_ENCRYPTION_KEY` handling | Partly — covers the configured branch, not `_derive_secret_key_candidates` |
| `tests/AuthNZ/integration/test_single_user_cookie_session.py` | end-to-end cookie session | Partly |
| `tests/Admin_Webhooks/test_crypto.py` | the ADR-027 envelope helpers | Yes |
| `tests/AuthNZ_Postgres/test_auth_enhanced_mfa.py` | MFA on Postgres | Yes for MFA |
| `tests/performance/test_authnz_multiuser_sqlite_load.py` | load shape | No — SQLite |

No test asserts that `session_manager`'s session-encryption key derivation and
`crypto_utils.derive_hmac_key_candidates` agree, because by construction they do not. No test asserts
that `key_resolution.resolve_api_key_by_hash` and `APIKeyManager.validate_api_key` accept and reject
the same keys, although the source comments claim they mirror each other.

## Validation Commands

```
$ grep -rn "derive_hmac_key" tldw_Server_API/app | grep -v "AuthNZ/crypto_utils.py" | wc -l
      30      # across jwt_service, api_key_manager, session_manager, key_resolution,
              # csrf_protection, mfa_service, telegram_runtime_repo, chat.py, embeddings_v5

$ grep -rn "PBKDF2\|pbkdf2" tldw_Server_API/app/core/AuthNZ/ | grep -c "hashlib.pbkdf2_hmac\|PBKDF2HMAC("
       5      # api_key_crypto:66, api_key_crypto:99, jwt_service:781, jwt_service:798,
              # session_manager:492

$ python3 -c "import hashlib; hashlib.pbkdf2_hmac('sha256', b'x', b'salt', 10, dklen=0)"
ValueError: key length must be greater than 0.

$ python3 -c "import base64; base64.urlsafe_b64decode('abc!' + '=' * (-4 % 4))"
binascii.Error: Incorrect padding

$ grep -rn "AESGCM\|aesgcm\|encrypt_json_blob" tldw_Server_API/app/core/AuthNZ/ | grep -c "Security.crypto\|_with_key"
       6      # all in user_provider_secrets.py and admin_webhook_secrets.py — ADR-027 compliant

$ grep -n "^def test\|^async def test" tldw_Server_API/tests/AuthNZ/unit/test_api_key_crypto.py | wc -l
       4

$ grep -rn "isolated_test_environment" tldw_Server_API/tests/AuthNZ_Unit tldw_Server_API/tests/AuthNZ_SQLite tldw_Server_API/tests/AuthNZ_Postgres tldw_Server_API/tests/AuthNZ_Federation | wc -l
       0

$ grep -rl "core\.AuthNZ" tldw_Server_API/tests | wc -l
    1243
```

Test-tree sizes (files / LOC / own conftest):
`AuthNZ` 226 / 53,734 / yes (2,083 LOC) · `AuthNZ_Unit` 93 / 30,569 / **no** ·
`AuthNZ_SQLite` 48 / 11,984 / yes (16 LOC) · `AuthNZ_Postgres` 21 / 4,779 / yes (36 LOC) ·
`AuthNZ_Federation` 10 / 2,675 / **no**. Total 398 files / 103,741 LOC.
Of the 1,243 test files importing `core.AuthNZ`, **352 are inside the five trees and 891 (72%) are
outside them.**

## Findings

### FINDING authnz-9 — The shared key-derivation module names SessionManager in its own docstring; SessionManager rolls its own with a different salt strategy and iteration count

```
axis:        duplication
class:       adoption-gap
severity:    High
sites:       canonical: crypto_utils.py:derive_hmac_key_candidates (142-235), backed by
               _derive_hmac_key_from_source (67-101); module docstring at :1-8 explicitly scopes it
               to "JWTService, APIKeyManager, CSRF, and SessionManager"
             bypassing copy: session_manager.py:_derive_secret_key_candidates (459-505), reached from
               _get_or_create_encryption_key (239-457) at :443, then _init_encryption (230-238)
             the same file imports the canonical helper at session_manager.py:42-43 and uses it
               correctly at :884-886 for token *hashing* — so one file contains both the adoption and
               the bypass
canonical:   crypto_utils.py:derive_hmac_key_candidates (142)
destination: n/a — adopt the existing canonical. Nothing new is needed.
knowledge:   "which configured secrets are key material, in what order, and how they are stretched".
             Written twice, and the two versions disagree on all four sub-decisions:
               1. salt — canonical uses a per-secret domain-separated salt (`_derive_hmac_kdf_salt`,
                  :34-36) and additionally emits a legacy fixed-salt candidate for rotation;
                  session_manager uses one static global salt `b"session_encryption_salt_v1"` (:494)
                  and emits a single candidate.
               2. iterations — 100,000 (`_HMAC_KDF_ITERATIONS`, :37) vs 600,000 (:496).
               3. source ordering — canonical pre-hashes `SINGLE_USER_API_KEY` with SHA-256
                  (`add_source(..., prehash=True)`, :186) "for parity with legacy logic";
                  session_manager feeds the raw key bytes (:485, :502).
               4. memoization — canonical is memoized with a documented rationale (:42-58);
                  the copy is not.
scenario:    n/a — this is the duplication axis and I am not claiming a live defect. Session
             encryption is self-consistent today because both write and read go through the same
             derivation. The cost is that the next change to the secret-precedence rule is guaranteed
             to be applied to one of the two. A concrete instance already exists in the tree:
             `crypto_utils.derive_hmac_key_candidates` grew a production guard refusing the
             deterministic test fallback when `ENVIRONMENT in {production, prod}` (:214-219) and grew
             the exclusion of `JWT_PUBLIC_KEY` from key material (:180, :203). `session_manager`'s
             copy carries its own, separately written, comment about excluding public keys
             (:474-475, :479) — the same decision, reasoned out twice. The next such decision has a
             50% chance of landing in only one.
impact:      High. This is the module's designated anti-drift helper, it names the bypassing class in
             its own docstring, and the bypass is in the file that encrypts session tokens at rest.
             It is also the cheapest High in this ledger to act on, because the adoption target
             already exists and the same file already imports it.
tests:       import-grep reachability, not measured coverage.
             `tests/AuthNZ/unit/test_crypto_utils_key_cache.py`, `test_crypto_utils_public_keys.py`
             and `test_security_guards.py` cover the canonical; `tests/AuthNZ/unit/test_session_manager_configured_key.py`
             covers the explicit-`SESSION_ENCRYPTION_KEY` branch. Nothing covers
             `_derive_secret_key_candidates`, and nothing asserts the two agree.
effort:      moderate, and NOT because the code is hard. Adopting the canonical changes the derived
             Fernet keys, so existing encrypted session rows stop decrypting unless the current
             derivation is retained as a trailing rotation candidate. The mechanism for that already
             exists — `_fernet_candidates` (:118, :235) is an ordered list and `decrypt_token`
             (:919-950) already walks it. So the shape is: canonical candidates first, current
             `_derive_secret_key_candidates` output appended as legacy, remove after one release.
owner-only:  no
confidence:  confirmed (the docstring scope, the bypass, and all four divergences, each read at the
             cited lines); assumption (that the duplicated public-key reasoning indicates real past
             drift rather than deliberate independent authorship).
```

### FINDING authnz-10 — Two API-key verification paths kept in sync by comment, and one has dropped the constant-time compare

```
axis:        correctness
class:       divergent-copies
severity:    Medium
sites:       key_resolution.py:resolve_api_key_by_hash (55-144), key-id branch at :78-107
             api_key_manager.py:_verify_new_format_key (508-535), _verify_legacy_key (537-551)
             the coupling is stated in prose, not code:
               key_resolution.py:62-63 — "computes ordered HMAC-SHA256 digests ... (derive_hmac_key_candidates)"
               key_resolution.py:126 — "Important: mirror APIKeyManager.hash_candidates (HMAC-SHA256 with secret key)"
               key_resolution.py:132 — "Dialect-aware query (aligns with APIKeyManager.validate_api_key)"
               key_resolution.py:42-43 — "The implementation MUST remain byte-for-byte compatible"
canonical:   NONE — two peers, neither designated
destination: the legacy-HMAC comparison belongs in `api_key_crypto.py`, which already owns the
             new-format `verify_kdf_hash`; that module's single responsibility is "API-key hash
             construction and verification" and the legacy digest comparison is the same
             responsibility. `key_resolution` and `api_key_manager` would both call it.
knowledge:   "does this presented API key match this stored hash". Three sub-decisions are duplicated:
             which hash candidates to derive, how to compare, and what a key-id miss means.
scenario:    The comparison diverged. `api_key_manager.py:534` is
             `any(hmac.compare_digest(stored_hash, cand) for cand in hash_candidates)` — constant
             time. `key_resolution.py:107` is `if stored_hash and stored_hash in digests` — plain
             CPython string equality, which short-circuits on length and then on first differing
             byte. Both compare a stored 64-char hex HMAC digest against attacker-influenced derived
             digests, on the legacy-hash branch reached after a successful `key_id` lookup. I am
             **not** claiming this is exploitable: a remote timing oracle over Python string equality
             on a 64-char hex string, where recovering the stored digest still leaves the attacker
             needing to invert HMAC-SHA256, is not a practical attack. What it is, concretely, is two
             copies of one security decision where one copy has silently lost the discipline the
             other keeps — and the file that lost it is the one whose own docstring at `:42-43` says
             the implementation "MUST remain byte-for-byte compatible".
impact:      Medium. The live risk is drift, not timing: `resolve_api_key_by_hash` feeds
             `llm_budget_middleware.py:189` and `llm_budget_guard.py:43`, so a future change to
             candidate derivation or expiry semantics applied to only one path means budget
             enforcement and authentication disagree about which keys are valid. The dropped
             `compare_digest` is the evidence that this drift has already started.
tests:       import-grep reachability, not measured coverage.
             `tests/AuthNZ/unit/test_key_resolution.py` and
             `tests/AuthNZ/unit/test_api_key_manager_validation.py` test the two paths separately.
             No test feeds the same key to both and asserts the same verdict.
effort:      cheap. Extract one `verify_legacy_hmac_hash(api_key, stored_hash, key_materials) -> bool`
             into `api_key_crypto.py` using `hmac.compare_digest`, call it from both, and add one
             parity test that asserts both entry points agree on accept and reject.
owner-only:  no
confidence:  confirmed (both implementations, the comment-based coupling, the compare divergence);
             assumption (that the timing difference is not practically exploitable — stated as my
             judgement, which is why this is Medium and not a security escalation).
```

### FINDING authnz-11 — Cluster C1: the base64 padding idiom is shared with pagination cursors, but this site's decode feeds an unguarded KDF length

```
axis:        correctness
class:       divergent-copies
severity:    Medium
sites:       api_key_crypto.py:_b64decode (117-119) — `padding = "=" * (-len(encoded) % 4)` then
               `base64.urlsafe_b64decode(encoded + padding)`, byte-identical in form to the 21 other
               sites the briefing enumerates
             api_key_crypto.py:_b64encode (114-115) — the matching `.rstrip("=")` encoder
             api_key_crypto.py:verify_kdf_hash (78-106) — the consumer
canonical:   NONE — 22 independent writings of one idiom, no shared helper anywhere in the repo
destination: a single `core/Utils/base64_codec.py` whose one responsibility is "unpadded URL-safe
             base64 encode/decode", exposing **two named functions, not one**:
             a strict one (`validate=True`, raises on any non-alphabet byte) for signed and crypto
             tokens, and a lenient one for opaque pagination cursors whose callers already wrap the
             decode in a 400 response. Explicitly NOT Utils.py and NOT http_client.py.
             **The security boundary must not be flattened into one function.** Rationale below.
knowledge:   "how an unpadded URL-safe base64 string becomes bytes, and what happens when it is not
             one".
scenario:    Two things separate this site from the pagination copies, and both argue for the
             two-function destination.
             (1) Trust boundary is inverted. The pagination copies decode a client-supplied cursor;
             this one decodes a value the same module produced via `_b64encode` and stored in
             `api_keys.key_hash`. The input is the database, not the request.
             (2) The decoded **length** is load-bearing here and inert there. `verify_kdf_hash` wraps
             its two `_b64decode` calls in `try/except Exception: return False` (:94-98), but the
             `hashlib.pbkdf2_hmac` call at :99-105 is outside any try, and it passes
             `dklen=len(expected)` — the length of the decoded stored digest. A stored hash of the
             form `pbkdf2_sha256$210000$<salt>$` (empty final segment, from a truncated write, a
             partial migration, or a manual edit) splits cleanly into four parts, `_b64decode("")`
             returns `b""` without raising, and `pbkdf2_hmac(..., dklen=0)` raises
             `ValueError: key length must be greater than 0.` (verified above) **out of
             `verify_kdf_hash`**, where every caller expects a bool. `key_resolution.py:93` and
             `api_key_manager.py:526` both call it bare. The same gap leaves `iterations`
             unbounded — `int(iterations_raw)` at :89-92 accepts any integer, so a corrupted
             iteration count is a hang rather than a `False`.
             Because `urlsafe_b64decode` defaults to `validate=False`, non-alphabet bytes in a
             corrupted stored hash are silently discarded rather than rejected, which is exactly how
             a short-but-nonempty decode arises. That leniency is *correct* for a pagination cursor
             and *wrong* here — which is the divergence the consolidation must preserve rather than
             erase.
impact:      Medium. It needs an already-corrupted stored hash, so it is not a live-traffic bug; but
             the failure mode is a 500 on the authentication path rather than a 401, for every
             request presenting that key, and the guard is a two-line fix.
tests:       import-grep reachability, not measured coverage.
             `tests/AuthNZ/unit/test_api_key_crypto.py` has 4 tests and calls `verify_kdf_hash` twice
             (`:43-44`), both with a well-formed hash. No malformed, truncated, or
             wrong-scheme stored hash is ever passed. `tests/AuthNZ/unit/test_key_resolution.py` and
             `test_api_key_manager_validation.py` reach it only through the happy path.
effort:      cheap for the guard (bring `pbkdf2_hmac` inside the existing `try`, or reject
             `len(expected) == 0` and out-of-range `iterations` explicitly, plus three table-driven
             tests). Moderate and cross-module for the 22-site codec consolidation, which is a
             separate piece of work and needs the design-first treatment because it spans nine
             modules and two trust classes.
owner-only:  no for `api_key_crypto.py`; yes for the seven `api/v1/endpoints/*` sites in the wider
             C1 cluster.
confidence:  confirmed (the idiom, the two trust classes, the unguarded `pbkdf2_hmac`, the
             `dklen=0` ValueError, and the 4-test coverage); probable-risk (that a truncated stored
             hash occurs — I did not find a code path that writes one; `kdf_hash_api_key` always
             produces a 32-byte digest).
```

### FINDING authnz-12 — Hundreds of thousands of PBKDF2 rounds run synchronously on the event loop, in a module that already uses `asyncio.to_thread` for far cheaper work

```
axis:        efficiency
class:       n/a
severity:    Medium
sites:       api_key_crypto.py:verify_kdf_hash (78-106) — 210,000 rounds
               (`API_KEY_KDF_ITERATIONS`, :13), called from the async
               `key_resolution.py:resolve_api_key_by_hash (93)` and
               `api_key_manager.py (526)` with no offload
             jwt_service.py:hash_password_reset_token_candidates (789-806) — 200,000 rounds
               **per key candidate**, called from the async `api/v1/endpoints/auth.py:2678`
             jwt_service.py:hash_password_reset_token (764-787) — 200,000 rounds, from
               `api/v1/endpoints/auth.py:2611`
             session_manager.py:_derive_secret_key_candidates (459-505) — 600,000 rounds per
               configured secret, from the sync `_init_encryption (230)`, itself re-entered lazily
               from the async `encrypt_token (911)` and `decrypt_token (926)`
cost-driver: PBKDF2-HMAC-SHA256 rounds executed on the asyncio event-loop thread, blocking every other
             coroutine for the duration. Scales with (a) concurrent API-key-authenticated requests
             x 210,000 rounds each, and (b) for password reset, with the number of configured
             secrets: `derive_hmac_key_candidates` emits two keys per source (a v2-salt and a
             legacy-salt candidate, `crypto_utils.py:227-234`) across up to five sources
             (`SINGLE_USER_API_KEY`/`API_KEY_PEPPER`, `JWT_SECRET_KEY`, `JWT_PRIVATE_KEY`,
             `JWT_SECONDARY_SECRET`, `JWT_SECONDARY_PRIVATE_KEY`), so a deployment configured for key
             rotation runs up to 10 x 200,000 = 2,000,000 rounds in one synchronous call.
knowledge:   n/a
scenario:    n/a — efficiency axis.
impact:      Medium, deliberately not High, for two reasons I checked. The reset-password route is
             rate-limited before the KDF runs (`api/v1/endpoints/auth.py:2650-2661`,
             `_reserve_auth_rg_requests(policy_id="authnz.reset_password", entity=f"ip:{ip_addr}")`),
             which caps the amplification; and `verify_kdf_hash` at 210,000 rounds is a deliberate,
             correct choice for API-key storage. The finding is not "the KDF is too slow" — it is
             that it runs **on the event loop**. The module has already established the fix: it uses
             `asyncio.to_thread` at `initialize.py:675`, `User_DB_Handling.py:700` and `:1171`,
             `database.py:1153`, `:1155`, `:1177`, `email_service.py:833`, `alerting.py:353` and
             `:389`, `repos/api_keys_repo.py:165`, and `repos/admin_monitoring_repo.py:74` — for
             schema checks, SMTP delivery, and file writes, all of which are cheaper than 210,000
             PBKDF2 rounds. The five heaviest CPU operations in AuthNZ are the ones that skipped it.
             The team has also already measured this exact class of problem and fixed it once:
             `crypto_utils.py:42-58` documents "POST /api/v1/chat/completions — 8 derivations, 110 ms
             of a 167 ms request" as the reason `derive_hmac_key` is memoized. That memoization does
             not help `verify_kdf_hash` or `hash_password_reset_token_candidates`, whose PBKDF2 input
             is the presented credential and therefore cannot be cached.
tests:       import-grep reachability, not measured coverage.
             `tests/performance/test_authnz_multiuser_sqlite_load.py` is the only load-shaped test
             reaching this code and it is SQLite-only; it does not measure event-loop stall.
effort:      cheap. `await asyncio.to_thread(verify_kdf_hash, api_key, stored_hash)` at the two call
             sites, and the same for the two `jwt_service` reset-token helpers at their endpoint call
             sites. `session_manager._init_encryption` runs once per manager instance and is a lower
             priority. No behaviour change, no new dependency, and the pattern is already idiomatic
             in this module.
owner-only:  no for `key_resolution.py` and `api_key_manager.py`; yes for the two `auth.py` call
             sites.
confidence:  confirmed (the iteration counts, the synchronous call sites, the candidate-count
             multiplication, the existing `asyncio.to_thread` usage, and the rate limit that bounds
             the reset path); assumption (the wall-clock cost per request — I did not benchmark).
```

### FINDING authnz-13 — The five-way AuthNZ test split has no written rule, the directory names misdescribe their contents, and the real Postgres tree is none of them

```
axis:        duplication
class:       true-duplication
severity:    Medium
sites:       tests/AuthNZ (226 files / 53,734 LOC / conftest.py 2,083 LOC)
             tests/AuthNZ_Unit (93 / 30,569 / no conftest)
             tests/AuthNZ_SQLite (48 / 11,984 / conftest.py 16 LOC)
             tests/AuthNZ_Postgres (21 / 4,779 / conftest.py 36 LOC)
             tests/AuthNZ_Federation (10 / 2,675 / no conftest)
             the de-facto partition: .github/workflows/ci.yml:473-474 (authnz-unit),
               :733-734 (gap-verified-6), :936-946, :948-949 (auth-postgres),
               :961-962 (auth-sqlite), :964-968 (auth-unit-a-l / auth-unit-m-z), :951-959
canonical:   NONE
destination: two trees — `AuthNZ` (no DB / mocked) and `AuthNZ_DB` (backend-parametrized) — with the
             26 duplicated subjects becoming `@pytest.mark.parametrize("backend", ["sqlite", "postgres"])`
knowledge:   "where does a new AuthNZ test go". Nothing in the repository answers it. An exhaustive
             grep of `*.md`, `*.toml`, `*.ini`, `*.cfg`, `*.yml`, `*.yaml` for the four suffixed
             directory names returns **no prose defining the split**. `tests/README.md` has a Postgres
             section but never names them. `CLAUDE.md:245-248` documents the fixture, not the trees.
             `pyproject.toml:1459-1472` lists some of the files only for `per-file-ignores`.
             The only operative partition is `ci.yml`, and it slices `AuthNZ/unit` **alphabetically**
             (`test_[a-l]*.py` / `test_[m-z]*.py`) — which is the proof that the axis is shard
             wall-clock, not domain.
scenario:    n/a — duplication axis.
impact:      Medium. Three concrete costs.
             (1) **The names misdescribe the contents.** `AuthNZ_SQLite/conftest.py` is a 16-line
             re-export shim that does not force SQLite. `AuthNZ_Postgres/conftest.py` imports its
             entire fixture set (`setup_test_database`, `reset_singletons`, `event_loop`,
             `clean_database`, `test_db_pool`, `real_audit_service`) from `AuthNZ/conftest.py` — it is
             a 21-file satellite of a 226-file parent. The actual Postgres tree is
             `tests/AuthNZ/integration` (76 files), which holds the Postgres twins for 16 of the 38
             subjects that look SQLite-exclusive.
             (2) **Duplication is hidden by naming, not absent.** Exact basename collisions across the
             five trees: 0. Exact test-function-name collisions: 0. Strip the `_sqlite` / `_pg` /
             `_postgres` suffixes and **26 file-stem collisions and 32 function-name collisions**
             appear, every one of them on the SQLite-vs-Postgres axis. The clearest case is
             `AuthNZ_Postgres/test_authnz_llm_usage_log_router_columns_pg.py` (47 LOC) against
             `AuthNZ_SQLite/test_authnz_llm_usage_log_router_columns_sqlite.py` (45 LOC): the 15-column
             assertion set and both index assertions are byte-identical, and only the introspection
             differs (`information_schema` query vs `PRAGMA table_info`).
             (3) **`AuthNZ_Unit` is misnamed and is the largest redundancy.** 93 files / 30,569 LOC
             sitting beside `AuthNZ/unit` at 138 files / 32,975 LOC with the same nominal purpose. In
             practice `AuthNZ_Unit` is an endpoint-authorization matrix — 39 of 92 test files drive a
             `TestClient`, 37 filenames contain `claims`, 31 contain `permissions` — not unit tests.
             Its content is the concern of the adjacent `auth-dependencies` ledger; its *name* is why
             nobody can tell.
tests:       n/a — the tests are the subject.
effort:      moderate as a consolidation; **cheap for the minimum useful step**, which is writing the
             rule down. Right now nothing tells a contributor where a new test goes, so every new
             test entrenches the split further.
owner-only:  no
confidence:  confirmed (all counts, the absent documentation, the conftest contents, the ci.yml
             alphabet slicing, the 26 and 32 collision counts, and the byte-identical assertion
             block); assumption (that two trees is the right target shape rather than three).
```

### FINDING authnz-14 — The per-test Postgres fixture is used by zero tests in all four satellite trees, including the one named `AuthNZ_Postgres`

```
axis:        correctness
class:       n/a
severity:    High
sites:       fixture: tests/AuthNZ/conftest.py:isolated_test_environment (631-741) —
               function-scoped, **not autouse**, must be requested explicitly
             requested by 56 test files plus 3 conftests, distributed as:
               tests/AuthNZ/integration 45, tests/Resource_Governance/integration 3,
               tests/MCP_unified 2, tests/AuthNZ/unit 1 (test_session_refresh_cache.py),
               tests/AuthNZ/property 1, tests/Collections 1, tests/Evaluations 1, tests/Tools 1,
               tests/Watchlists 1, tests/wizard 1, plus conftests at
               tests/MCP_unified/conftest.py:1 and tests/Resource_Governance/conftest.py:22
             requested by **0** files in tests/AuthNZ_Unit, tests/AuthNZ_SQLite,
               tests/AuthNZ_Postgres, tests/AuthNZ_Federation
canonical:   n/a
destination: n/a
knowledge:   n/a
scenario:    `CLAUDE.md:245-248` designates this fixture as the sanctioned way to get a real Postgres
             ("provisions a per-test Postgres database"; "never roll your own database setup"). It
             does the full job: `CREATE DATABASE tldw_test_<8 hex>`, `CREATE EXTENSION pgcrypto`,
             inline schema creation, a `TestClient`, and a clean `pytest.skip` when Postgres is
             unreachable after attempting a Docker start (hard-failing instead when
             `TLDW_TEST_POSTGRES_REQUIRED` is set). It is genuinely good infrastructure. It is used by
             45 files in `tests/AuthNZ/integration` and by **nothing** in the directory literally
             named `AuthNZ_Postgres` — which instead re-imports `test_db_pool` from the parent
             conftest and marks itself `pytest.mark.postgres`. Combined with stage 2's coverage
             counts, this is the mechanism behind the module's central risk: of 31 test files
             reaching the five most-branched repos, 19 are SQLite-only, 6 drive the Postgres branch
             against a **stub pool** (an object with `.pool` set, which proves the branch is taken and
             nothing about the SQL), and only 6 touch a real Postgres.
             `repos/generated_files_repo.py` — 35 backend branches, the most in the module, and the
             file that assembles SQL with `format_map(locals())` at `:383` and `:471` — has exactly
             one Postgres-branch test and it is a stub.
             `repos/billing_repo.py` — 19 branches including the `TRUE`/`1` boolean literal swap at
             `:93`, `:95` — has none at all.
impact:      High. The repo-wide audit records the 2026-09-21 cross-user isolation finding that
             leaks fail to converge because the SQLite/PostgreSQL split is only tested on SQLite.
             This finding is the concrete mechanism for that, in the module where it matters most,
             and it is a *fixable* one: the harness is already built, already auto-starts Docker,
             already skips cleanly, and is already used by 45 files. The gap is adoption, not
             capability.
tests:       n/a — the tests are the subject.
effort:      cheap per repo. Adding `isolated_test_environment` to the existing
             `tests/AuthNZ_SQLite/test_authnz_*_repo_sqlite.py` files as a second parametrized
             backend is mechanical; the fixture handles provisioning and skipping. Start with
             `generated_files_repo` and `billing_repo`, which have the most branches and the least
             coverage.
owner-only:  no
confidence:  confirmed (the fixture definition and behaviour, the 56-file usage list, the zero count
             in all four satellite trees, and the per-repo real-PG / stub-PG / SQLite-only split).
```

## Suggested Refactor/Actions

1. **(cheap) Guard `verify_kdf_hash`.** Move the `hashlib.pbkdf2_hmac` call at `api_key_crypto.py:99`
   inside the existing `try`, or reject `len(expected) == 0` and out-of-range `iterations` explicitly
   before it. Add three table-driven cases to `tests/AuthNZ/unit/test_api_key_crypto.py` — empty
   derived segment, non-base64 salt, absurd iteration count — all asserting `False`, not a raise.
   Addresses authnz-11's guard half.

2. **(cheap) Offload the KDFs.** `await asyncio.to_thread(...)` around `verify_kdf_hash` at
   `key_resolution.py:93` and `api_key_manager.py:526`, and around the two reset-token helpers at
   their `auth.py` call sites (owner-only). The pattern is already used ten times in this module for
   cheaper work. Addresses authnz-12.

3. **(cheap) Extract one legacy-HMAC verifier** into `api_key_crypto.py` using `hmac.compare_digest`,
   called from both `key_resolution.py:107` and `api_key_manager.py:534`, plus one parity test that
   feeds the same key to both entry points and asserts the same verdict. Addresses authnz-10.

4. **(moderate, needs a design doc) Adopt `crypto_utils.derive_hmac_key_candidates` in
   `SessionManager`,** keeping the current `_derive_secret_key_candidates` output appended as a
   trailing rotation candidate so existing encrypted sessions keep decrypting; remove it after one
   release. The candidate-list mechanism at `session_manager.py:118, :235, :919-950` already supports
   this. Needs `Docs/Design/2026-MM-DD-authnz-session-key-derivation-design.md`, an ADR entry (it
   changes a cryptographic key-derivation decision), a Backlog task linking both, and a staged
   `IMPLEMENTATION_PLAN_authnz-session-keys.md`. Addresses authnz-9.

5. **(cheap, highest value per line changed) Add `isolated_test_environment` to the repo tests that
   have no real-Postgres coverage,** starting with `generated_files_repo` and `billing_repo`.
   Addresses authnz-14 and the coverage half of authnz-8 and authnz-2.

6. **(cheap) Write the test-tree rule down** in `tests/README.md` before consolidating anything. One
   paragraph saying what belongs in each of the five trees. Every new test added before that
   paragraph exists entrenches the split further. Then, separately and later, propose the two-tree
   consolidation as its own design doc. Addresses authnz-13.

## Synthesis for the module

AuthNZ is not a slop module. It has a real shared crypto helper with a stated anti-drift purpose, a
real datetime helper in the right place, a real dual-backend abstraction, ADR-027 compliance where the
ADR expects it, and one of the cleanest layering records in `app/core/` (3 API imports against a
repo-wide 161). The problem is uniform and structural: **the helpers exist and are only partly
adopted, and the thing that would have caught the resulting drift — a test that runs the same
assertion against both backends — exists exactly once, for one table.**

Three findings share one root and should be sequenced together, because fixing the last one first
makes the others safe: authnz-14 (the Postgres fixture is unused in four of five trees) enables
authnz-8 (SQL that differs in what it writes) and authnz-2 (two schema-evolution models with no
parity check). Do authnz-14 first.

Two findings are adoption gaps against helpers that name their own bypassers: authnz-9
(`crypto_utils` names SessionManager in its docstring) and authnz-5 (`repos/datetime_utils.py` exists
and 11 siblings ignore it). Both are cheap in code and expensive in care, because both change values
already persisted. authnz-5 additionally requires correcting the canonical helper *before* adopting
it — the widely-copied semantics are the lossy ones.

The cheapest real wins, in order: guard `verify_kdf_hash` (authnz-11), offload the KDFs (authnz-12),
add the migration-registry invariant test (authnz-3), and make `_normalize_sqlite_sql` fail loudly
(authnz-1). None needs a design document; together they are perhaps 60 lines of change and they
remove a 500-on-auth, an event-loop stall, a silent version-numbering drift, and a silent SQL rewrite.
