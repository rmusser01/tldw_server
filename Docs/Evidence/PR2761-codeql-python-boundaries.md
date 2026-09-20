# PR 2761: individual Python non-path CodeQL dispositions

Tracking: **TASK-13013.3.1**, child of TASK-13013.3. Date: 2026-09-10.

This review covers exactly 18 branch-open alerts, individually mapped below. The
inventory `/tmp/pr2761-codeql-current-alerts.json` identifies analysis commit
`43165c8c82f6df9bab56f425879ac286aecbc5ee`; the inspected worktree is
`28797892b1e021dad55cbc736532e702f0963ae8`. `git diff` between those commits is empty
for all ten assigned source/test files. Full Python SARIF was read from
`/tmp/pr2761-python-current.sarif.json`, including source-to-sink code flows.

The Notes cursor factory is repaired after reproducing a publicly known signing
key in asymmetric-JWT configurations (2662). The other 17 proposed dispositions
are **false positive for the named weakness and traced path**, supported by the
specific evidence listed. No query configuration, suppression, gate, or alert
state was changed; this is not a statement that these entire modules have no
vulnerabilities.
Parent review owns any eventual individual GitHub disposition. Machine-readable
mapping: `/tmp/pr2761-codeql-python-dispositions.json`.

## HTTP request sinks

All four SARIF source families (Watchlists draft-source payload, Reading payload,
media form input, and connector cursor) enter the shared async request wrappers.
These inputs can intentionally select a public remote resource. This is an
egress-validation boundary, not a fixed-host-only HTTP API. Both wrappers call
`_avalidate_egress_or_raise` before acquiring a client and at every redirect hop.
`evaluate_url_policy` rejects private/reserved addresses and unresolved DNS by
default, returns the accepted address set, and rejects changed answers against
the request's pinned set. `_prepare_pinned_transport_target` replaces the network
hostname with an accepted address and overwrites caller-supplied Host while
preserving original HTTP/TLS identity. The tainted textual URL alone does not
represent the effective socket destination after this validation.

| Alert | Exact scanned sink | Individual result |
|---|---|---|
| 2337 | `http_client.py:2893`, `client.request(...transport_url...)` | The non-bounded httpx branch gets the validated address through `accepted_resolved_ips` and passes original SNI explicitly. Wrapper sets `follow_redirects=False`. The new httpx/unbounded cases verify public-IP transport, Host/SNI preservation, and denial of private targets, private redirect hops, and rebinding. |
| 2603 | `http_client.py:2862`, `client.stream(...transport_url...)` | The bounded httpx branch uses the same prepared destination and SNI extension; streaming does not bypass policy. New httpx/bounded cases exercise this exact branch, including a successful two-byte body and all three denial boundaries. |
| 2604 | `http_client.py:3000`, `session.request(...url...)` | Here `url` has already been replaced by `_prepare_pinned_transport_target`; `server_hostname` preserves TLS identity. `allow_redirects=False` is unconditional. New aiohttp cases cover both bounded and unbounded bodies and assert actual session arguments plus denial boundaries. |

New `tests/http_client/test_codeql_egress_boundaries.py` runs the real policy and
wrapper loops. Only DNS and socket-facing transport are controlled. It explicitly
disables the wrapper's pytest hostname relaxation so accepted-IP pinning is
exercised. Twenty cases passed. An attacker-supplied Host is overwritten; DNS
switching from a public address to loopback fails before the first request;
public-to-loopback redirects perform only the original public request.

Limits: trusted administrator policy overrides and explicitly configured internal
endpoints can permit internal resources by design. These tests are not live TLS
handshakes or a proof for arbitrary third-party client implementations. Existing
TLS-pinning tests exercise the separate certificate check. No bypass was found in
any of the three traced sinks under the default untrusted-URL policy.

## XPath compilation and evaluation

All three traces originate at authenticated `watchlists.py:2884` draft-source
configuration. They contain a **complete extraction expression**, evaluated over
the supplied/fetched HTML document. There is no interpolation of a purported data
value into an authorization query or access to an XML credential database.

| Alert | Exact scanned sink | Individual result |
|---|---|---|
| 2600 | `Web_Scraping/selectors/engine.py:235`, cached `XPath(expr)` | The cache contains compiled expressions, not documents or results. A new test reuses one expression on two distinct documents and returns only each supplied document's content. Runtime and validation reject variables, unions and axes before compilation. |
| 2601 | `engine.py:252`, uncached `XPath(expr)` | The compile-only validation path deliberately checks user-authored selector syntax; it does not perform data access. The uncached test verifies the same document isolation. `_selector_validation_error` applies the same safety gate as runtime. |
| 2602 | `engine.py:294`, bounded `XPath(f"({expr})[position() <= ...]")` | The expression is intentionally executable; the appended predicate bounds node-set output. Existing tests establish node-set, attribute and scalar parity. The new normal/bounded tests try `document(file_uri)/secret/text()` against a real private fixture; both fail with no returned data because this XPath evaluator has no XSLT `document()` I/O. |

No application XPath extensions are registered by the selector module. Limits on
length, predicates, descendant steps and function calls are retained. These facts
disprove the specific injection premise; they do not establish a hard CPU bound
for every XPath expression. Existing output limits, worker execution policy and
sanitized errors remain unchanged.

## Hashing paths

Each reported algorithm is SHA-256 or HMAC-SHA256. This rule warns about *password
hashing without an expensive KDF*, so data identity and digest use must be traced
individually; a secret-like field name is insufficient.

| Alert | Exact scanned sink | Source, downstream use, and proof |
|---|---|---|
| 2666 | `TTS/gateway_config.py:749` | SARIF starts at **`config.allow_user_api_key` at line 846**, a typed boolean, and carries that exact dictionary key into canonical configuration JSON. `api_key` is excluded from `output_fields`. The new test rotates the actual key without changing generation, then changes the permission boolean and observes a new generation. This is a credential-name heuristic false positive on a permission bit. |
| 2615 | `Prompt_Management/prompt_studio/mcts_optimizer.py:1149` | SARIF carries `[Dictionary element at key api_key]` through `strip_sensitive_durable_mapping` and `_clean_durable_value` despite the explicit sensitive-key `continue` branch. The actual digest contains only provider behavior. New top-level/nested API-key, password and access-token tests prove equality with credential-free configuration; the existing durable-cache suite proves key rotation does not create a new behavior cache entry. |
| 2264 | `AuthNZ/csrf_protection.py:113` | The source is `manager.validate_api_key`'s return at line 324, but line 329 extracts **integer `user_id`**, not the submitted credential. That integer becomes the HMAC message, with a separate PBKDF2-derived server key. The new async test traverses the exact lookup: two different presented credentials resolving to user 42 have the same suffix; user 43 has a different one. Existing CSRF tests verify wrong-user token rejection. |
| 2664 | `AuthNZ/crypto_utils.py:38` | SHA-256 derives a domain-separated per-secret **salt**, consumed by `hashlib.pbkdf2_hmac` at lines 86–92 (100,000 iterations, 32-byte result). The returned authentication key is the PBKDF2 output, not this SHA digest. Existing key-cache tests independently recompute PBKDF2 and compare current and legacy modes. |
| 2665 | `AuthNZ/crypto_utils.py:79` | The digest indexes the bounded, process-local 64-entry KDF memoization cache. It is not a stored password verifier or returned authentication key. Cache values are the same PBKDF2-derived keys described above. Tests verify equality with uncached derivation, source rotation, distinct legacy modes, bounded entries and cache reset. Access to this process memory already grants the derived MAC keys; the index introduces no password-database shortcut. |
| 2663 | `AuthNZ/byok_runtime.py:893` | Input is an administrator-configured **provider API key**. The digest supplies a credential revision to `_gateway_scope_token` at line 2419, invalidating the internal provider cache when the key rotates. It is not used to verify an account password. New digest tests and the existing `test_gateway_admin_scope_tracks_config_and_key_rotation` show stable revisions for the same key and distinct opaque scopes for key/configuration rotation, with no raw key or intermediate fingerprint in the scope representation. |
| 2613 | `AuthNZ/byok_runtime.py:1896` | The exact generation fields contain only `access_token`; the digest labels a provider-issued OAuth access-token generation for refresh coalescing. New tests verify access-token rotation changes the digest and refresh-secret rotation alone does not. Existing concurrent OAuth refresh tests verify the surrounding generation/coalescing behavior. This is token version identification, not human password verification. |
| 2614 | `AuthNZ/byok_runtime.py:1913` | This separate refresh-publication digest covers `access_token` and `issued_at`. Its two SARIF paths end at those respective tuple slots. It detects refreshed-token publication, not credential validity. New tests cover access-token rotation and exclude refresh-secret changes; existing OAuth refresh tests exercise publication/coalescing. |
| 2662 | `Notes_Graph/suggestion_api.py:672` | **Repaired real signing-key gap found by tracing.** Settings permits RS256/ES256 with a private signing key and no symmetric JWT secret; multi-user mode needs no single-user API key. The old factory then selected the public literal `notes-graph-cursor-local`. Two regressions construct real RS256/ES256 Settings using generated private PEM keys and successfully forge a cursor under the old code (expected rejection fails). The factory now calls existing `derive_hmac_key(settings)`, which includes private-key material and PBKDF2, instead of local SHA-256/fallback. Both forgery tests now reject the cursor; authentic cursors still round-trip. This repairs cursor integrity; a cross-tenant data-access bypass was not demonstrated. |

Cryptographic limits: fast hashes are appropriate here for high-entropy provider
tokens, configured signing-key normalization and cache identity. These dispositions
do not endorse weak administrator-selected credentials or authorize storing human
passwords using SHA-256. An initial assumption that the Notes fallback was unreachable was disproved by
checking the asymmetric settings branch and running the forgery tests. The fix
uses the shared KDF in all authentication modes; already-issued pagination cursors
will be invalidated once after deployment and clients must restart pagination.
No legacy publicly known fallback signature is accepted.

## Regular expressions

| Alert | Exact scanned sink | Individual result |
|---|---|---|
| 2598 | `Notes/attachment_policy.py:206` | `fullmatch` is reached only after `1 <= len(value) <= 255`; short-circuit ordering excludes oversized values. The MIME pattern has two nonempty token runs separated by a slash excluded from both token classes. The new tests accept an exact 255-character valid MIME, reject 256, and replace the regex with a fail-if-called object to prove a million-character value never reaches it. The described unbounded adversarial string does not reach this sink. |
| 2596 | `tests/Web_Scraping/test_phase4_safe_regex.py:905` | The catastrophic `(?:a\|aa)+İ$` expression is a deliberate regression fixture passed to `search_untrusted`, never an unbounded in-process regex operation. The test requires the real subprocess worker to return `regex_timeout` within its 20ms execution budget, terminate/reap the child, close pipes, and leave no startup reader thread. It passed unchanged. |
| 2597 | `test_phase4_safe_regex.py:1343` | The separate substitution fixture `(?:a\|aa)+$` deliberately triggers backtracking in `sub_untrusted`'s isolated worker. Its unchanged test requires timeout, a reaped worker, closed pipes and no reader-thread leak. It passed unchanged. Removing these adversarial patterns would remove the test's security purpose. |

## Verification and limits

All Python commands activated the repository `.venv` first. The two asymmetric
cursor forgery tests failed against the original source, then passed after the
minimal factory fix. Other new tests provide evidence for existing safe boundaries.
Three earlier failures in the new tests were incomplete fixtures
(missing mandatory gateway fields and the OAuth `credential_version=2` tag),
corrected without changing source behavior.

- Hashing/cursor/credential-free cache suites plus initial new boundary tests:
  **167 passed**, 38 existing warnings; `/tmp/pr2761-python-hashing-final.log`.
- Existing BYOK OAuth/generation/gateway-admin subset: **38 passed**, 112 deselected;
  `/tmp/pr2761-python-byok.log`.
- Broader HTTP/selector/regex suites: **521 passed, 4 failed**, 9 warnings;
  `/tmp/pr2761-python-transport-selectors.log`. Two unchanged fork/admission tests
  fail in `multiprocessing.SemLock` with `ENOSPC` before production execution
  (the volume has 439 GiB free). Two unchanged million-character substitution
  tests exceed their 100ms budget, and still fail on an isolated rerun; no timeout
  was increased and no test was disabled. The two specific flagged adversarial
  tests passed in both runs. The host semaphore shortage and large-output timing
  failures remain broader verification limitations, not evidence that these
  deliberate timeout fixtures permit unbounded regex execution.
- Cursor repair verification: **36 passed**, 7 warnings, across new boundary
  tests, Notes suggestion API tests and shared key/public-key exclusion tests;
  `/tmp/pr2761-cursor-asymmetric-green.log`. The confirmed two-case red run is
  `/tmp/pr2761-cursor-asymmetric-red.log`.
- The final added-tests/exact-regex run is recorded in
  `/tmp/pr2761-python-boundaries-final.log`. Ruff passes for the source and two
  new test files. Bandit reports zero findings across that same scope with only
  ordinary pytest assertions (`B101`) excluded;
  `/tmp/pr2761-python-boundaries-bandit-final.json`. `git diff --check` passes.

## Existing main instances and global disposition scope

The parent queried active main instances in
`/tmp/pr2761-codeql-main-instances.json`. Both 2337 and 2264 remain open at main
`d9c245ac14c40df855d1ab6cd19b3c137b16b47b`; a GitHub dismissal would cover those
instances too, so their main boundaries were inspected independently.

- **2337, main `http_client.py:2182`:** `_prepare_pinned_transport_target` at
  lines 977–1040 already substitutes an accepted IP and overwrites Host while
  retaining TLS identity. `_afetch_httpx` validates initially at 2487 and on every
  hop at 2612, passes accepted addresses at 2556 and disables automatic redirects
  at 2554. Main `Security/egress.py:676–706` already rejects unresolved/private
  destinations and changed DNS answers. The guarded-sink disposition therefore
  does not depend on any release-only response-streaming change. This main check
  is source inspection, not a claim that the entire main runtime suite ran.
- **2264, main `csrf_protection.py:113`:** the whole CSRF file is byte-identical
  to the reviewed candidate. Main `crypto_utils.py` already derives the MAC key
  using 100,000-round PBKDF2; the candidate adds memoization around the same KDF.
  The HMAC message is still the selected integer user ID, not a password.

The final post-repair added-tests/exact-regex run passes **42 tests** with four
existing warnings (`/tmp/pr2761-python-boundaries-final.log`). No main source or
alert state was changed by this review.
