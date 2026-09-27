# Trusted Proxy Identity and Login Lockout Isolation Design

**Status:** Draft for written-spec review

**Tracking task:** TASK-13013.5

**Deferred architecture task:** TASK-13144

**Baseline:** `origin/dev` at `f676e23549ea8ed82ef53493260621a05b281863`

## Purpose

Harden the public server's client-IP and password-login lockout boundaries so
trusted reverse proxies cannot accidentally expose spoofable identity and one
attacker behind a shared proxy or NAT cannot lock unrelated users out of the
application.

This is a release-readiness repair, not a global request-identity rewrite. It
introduces one small pure resolver that AuthNZ and Resource Governor call from
their existing integration points. Their existing environment-variable
families remain supported.

## Current Problems

Two independent client-IP implementations currently trust forwarding headers
only when the physical peer is configured as trusted, but both select the
leftmost forwarded value. A proxy that appends to an inbound
`X-Forwarded-For` header therefore leaves an attacker-controlled value in the
selected position. The implementations can also drift because AuthNZ and
Resource Governor parse the same security boundary separately.

Password login separately checks and records a raw client-IP lockout bucket.
All users behind one correctly resolved proxy or NAT therefore share that
bucket. One attacker can consume it and block unrelated users before their
credentials are evaluated. Resource Governor already supplies the broader
IP-scoped request/spray control, so the password-login lockout can be isolated
to a client-and-login pair while retaining the existing account-wide bucket.

## Goals

- Resolve client identity only from a valid physical peer and explicitly
  trusted proxy hops.
- Make direct, single-proxy, multi-proxy, spoofed, malformed, IPv4, and IPv6
  behavior deterministic.
- Reuse one pure standard-library resolver from AuthNZ and Resource Governor.
- Preserve the existing AuthNZ and Resource Governor configuration names.
- Prevent one shared-IP attacker from consuming another login identifier's
  client-side password lockout bucket.
- Preserve the existing account-wide username lockout as a second layer.
- Reset the exact password-login buckets after ordinary or MFA success.
- Avoid a database schema migration, a new runtime dependency, and raw
  client/login pairs in new lockout identifiers.

## Non-Goals

- Rewriting `request.client` globally in middleware.
- Changing audit, setup, authorization, WebSocket, or unrelated middleware
  identity semantics.
- Supporting RFC 7239 `Forwarded`, hostname proxy entries, or non-IP
  forwarding values.
- Renaming or consolidating the `AUTH_*` and `RG_*` environment variables.
- Replacing Resource Governor's IP-wide request controls.
- Redesigning MFA throttling or the lockout database.

The broader HTTP/WebSocket middleware rewrite is tracked independently by
TASK-13144. It depends on this task but is outside TASK-13013 and cannot block
the current release.

## Architecture

### Shared pure resolver

Add a focused module under `tldw_Server_API/app/core/Security/` that depends
only on Python's `ipaddress` and collection types. Its public operation accepts:

- the physical peer string;
- trusted proxy IP/CIDR strings;
- zero or more ordered `X-Forwarded-For` field values; and
- an optional single-address forwarding field value.

It returns a canonical compressed IP string or `None`. It does not read process
environment, import FastAPI, inspect `Request`, log header contents, or replace
`request.client`.

AuthNZ's `resolve_client_ip()` remains the compatibility wrapper for
`AUTH_TRUST_X_FORWARDED_FOR` and `AUTH_TRUSTED_PROXY_IPS`. Resource Governor's
`derive_client_ip()` remains the compatibility wrapper for
`RG_TRUSTED_PROXIES` and `RG_CLIENT_IP_HEADER`. Each wrapper extracts request
data and delegates the trust decision to the shared operation. AuthNZ returns
`None` when no canonical IP exists, and its login helper maps that condition to
`unknown`. Resource Governor returns its existing `unknown` sentinel. The
current Resource Governor shortcut that treats any non-IP peer as loopback is
removed from the security path.

### Resolution contract

1. Parse the physical peer as an IP literal. Invalid or missing peers produce
   no resolved IP and never authorize a forwarding header. Compatibility
   wrappers map no result to their existing safe sentinel where required.
2. Canonicalize valid IPv4 and IPv6 values with `ipaddress.ip_address(...).compressed`.
3. Parse trusted entries as exact hosts or CIDRs. Ignore invalid configured
   entries. An empty or wholly invalid list trusts no proxy.
4. If forwarding is disabled or the physical peer is not trusted, ignore every
   forwarding field and return the canonical physical peer.
5. When `X-Forwarded-For` is present, join repeated field occurrences in wire
   order, split the resulting list, and require every token to be a plain valid
   IP literal. Empty or malformed tokens reject the entire chain and fall back
   to the physical peer without consulting a secondary header.
6. Walk the valid chain from right to left. Skip addresses in the configured
   trusted proxy networks. The first untrusted address is the client. This
   ignores attacker-prepended values once the actual untrusted client hop is
   reached. If every forwarded address is trusted, fall back to the physical
   peer rather than inventing a client identity.
7. AuthNZ uses `X-Forwarded-For` when present and consults `X-Real-IP` only when
   `X-Forwarded-For` is absent. A repeated, comma-separated, or invalid
   `X-Real-IP` value falls back to the physical peer.
8. Resource Governor treats `RG_CLIENT_IP_HEADER=X-Forwarded-For`
   case-insensitively as a list header. Other configured header names use the
   strict single-address rule.

Forwarding values with ports, brackets, zone identifiers, `for=` syntax, or
other decorations are invalid. Supporting those forms would add ambiguous
parsing outside the existing contract.

### Compatibility boundaries

The following names and opt-in behavior remain unchanged:

- `AUTH_TRUST_X_FORWARDED_FOR`
- `AUTH_TRUSTED_PROXY_IPS`
- `RG_TRUSTED_PROXIES`
- `RG_CLIENT_IP_HEADER`

Operators that configure both subsystems must use equivalent trusted-proxy
sets and compatible headers. The documentation will state this directly.
Equivalent configuration produces the same canonical identity; intentionally
different configuration remains possible for backward compatibility.

The only intentional behavior change for a valid enabled configuration is
that `X-Forwarded-For` is evaluated from the trusted edge inward instead of
accepting its leftmost value. Deployments whose ingress overwrites the header
continue to resolve the same client. Deployments whose ingress appends gain
spoof resistance.

## Login Lockout Isolation

### Stable composite key

Add one small login helper that builds a versioned deterministic lockout identifier
from the resolved client identity and attempted login identifier.

- Normalize the attempted identifier with `strip().lower()`, exactly matching
  `fetch_user_by_login_identifier()`.
- Use the canonical resolved IP, or the stable `unknown` sentinel when no IP is
  available.
- Encode the two strings as a compact JSON array and hash the UTF-8 bytes with
  standard-library SHA-256.
- Store the result as `login-client-v1:<64 lowercase hex characters>`.

The key must not use Resource Governor's `hash_entity()` because that helper's
configuration and fallback behavior are not the durable cross-process
identity contract required by the database-backed lockout tracker. A known
test vector will freeze the composite-key format.

The version prefix prevents collision with existing raw username/IP
identifiers and allows a later contract change without a schema migration.
Existing raw-IP rows are no longer checked or extended by this login path. The
repository has no general failed-attempt retention sweep, so those legacy rows
may remain inert until explicit database maintenance or a future compatible
cleanup task; this change does not guess which historical identifiers were IPs
and which were IP-shaped usernames.

SHA-256 prevents the new pair from being stored as plaintext but is not treated
as a secrecy boundary. The database already retains the separate account
identifier, and this task does not introduce or manage another application
secret merely to key short-lived lockout state.

### Password-login flow

The endpoint computes the normalized attempted identifier and composite key
before the pre-fetch lockout check.

- Pre-fetch: check the composite client/login bucket.
- User not found: record only the composite bucket. Resource Governor remains
  the IP-wide defense against high-cardinality username spraying.
- User found: also check the existing account-wide bucket keyed by the stored
  canonical username before password verification.
- Invalid password: record both the composite bucket and the account bucket.
- Successful non-MFA login: reset the exact composite bucket used for the
  attempt and the account bucket.
- MFA challenge: store the opaque composite key alongside `user_id` and
  `session_id` in the existing server-side ephemeral value. Do not store the
  raw IP/login pair there.
- Successful MFA login: require the cached key to match the exact versioned
  key shape, then reset it and the account bucket. Do not derive the reset key
  from the MFA request's potentially different network path.

Alias behavior is deliberate. Login by username and login by email may have
different composite buckets, while both resolve to the same account-wide
bucket after user lookup. Case and surrounding whitespace cannot create extra
buckets. A successful login resets only the exact composite bucket used by
that flow plus the account bucket; it does not erase unrelated attack history.

MFA token failures continue to use the existing MFA Resource Governor policy.
This task changes password-login failure isolation, not MFA attempt policy.

## Failure Handling and Privacy

- Untrusted peers never gain header authority.
- Any malformed forwarded chain fails closed to the physical peer.
- A malformed primary `X-Forwarded-For` chain cannot fall through to
  `X-Real-IP`.
- Invalid physical peers produce no canonical identity rather than being
  treated as loopback.
- Raw forwarding-header values and parser exception text are never logged.
- New lockout rows contain only the versioned digest, not the new raw
  client/login pair. The existing account bucket remains unchanged.
- Resolver failures must not bypass the account-wide lockout or Resource
  Governor controls.

## Expected Files

The implementation plan will confirm exact paths, but the intended scope is:

- one new shared security resolver and its focused tests;
- thin changes to AuthNZ `ip_allowlist.py` and Resource Governor `deps.py`;
- login/MFA lockout-key changes in `app/api/v1/endpoints/auth.py`;
- focused AuthNZ and Resource Governor regression tests;
- operator documentation for the retained proxy environment variables; and
- Backlog records for TASK-13013.5, completed TASK-13013.4 metadata, and the
  deferred TASK-13144 architecture task.

No database model, migration, dependency manifest, global middleware, or
WebSocket implementation belongs in this change.

## Verification Strategy

### Resolver matrix

- direct valid IPv4 and IPv6 peers;
- forwarding disabled;
- untrusted peer with spoofed forwarding fields;
- one trusted proxy with overwritten and appended XFF;
- multiple trusted proxies;
- attacker-prepended XFF values;
- repeated XFF field occurrences in wire order;
- all-trusted chains;
- malformed, empty, decorated, and mixed-validity chains;
- XFF precedence over X-Real-IP;
- strict single-address custom headers;
- invalid trusted entries and invalid physical peers; and
- canonical IPv4/IPv6 output.

### Login and MFA matrix

- exact stable composite-key vector and prefix validation;
- case/whitespace normalization parity with database lookup;
- same proxy plus different attempted users remain isolated;
- same client/login pair locks at the configured threshold;
- known-account failures still trigger the account-wide bucket;
- username/email aliases share the account bucket;
- unknown-user failures use the composite bucket;
- spoofed forwarding values cannot select another lockout bucket;
- ordinary success resets the exact composite and account buckets;
- MFA challenge carries only the opaque original composite key;
- MFA success from a different request IP resets the original key; and
- malformed cached MFA keys are ignored or rejected safely and never reset an
  attacker-selected identifier.

### Completion gates

- focused shared-resolver, AuthNZ, Resource Governor, login, and MFA tests;
- existing AuthNZ and Resource Governor regression modules;
- Python compilation and Ruff on touched Python;
- Bandit on touched production Python with no new Medium/High findings;
- `git diff --check`;
- required backend/security/coverage CI shards; and
- documentation/config examples checked for the retained environment names.

## Rollout and Rollback

No configuration migration is required. Proxy-header trust remains disabled by
default for AuthNZ and inactive for Resource Governor unless both its trusted
proxy set and header name are configured.

The change can be rolled back as one code revision without transforming stored
data. Older code ignores the versioned composite rows. Its window-reset logic
handles any legacy failed-attempt rows it consults after rollback, and expired
legacy lockout rows are pruned when checked.

At rollout, a currently active raw-IP lockout cannot be mapped to a composite
key because it has no attempted-login component. The new path therefore stops
consulting that one legacy bucket immediately. The existing account bucket and
Resource Governor remain active during this transition. Operators should
validate equivalent AuthNZ and Resource Governor proxy settings in staging
before production rollout.

## Reviewed Alternatives

### Global `request.client` middleware rewrite

This would provide one conceptual identity everywhere, but it changes audit,
WebSocket, setup, authorization, and middleware behavior together and risks
losing the immutable physical peer used to decide whether headers are trusted.
It is deferred to TASK-13144 with an explicit consumer inventory, separate
physical-peer storage, staged rollout, observability, and rollback.

### Keep both resolvers and patch each independently

This is a smaller immediate diff but preserves duplicated trust logic and
future drift. One pure shared resolver with thin compatibility wrappers is the
smallest durable correction.

### Keep raw IP lockout and add exceptions for proxies

Proxy-specific exceptions weaken brute-force protection and cannot distinguish
shared clients. The composite bucket plus existing account bucket and Resource
Governor layers isolates unrelated users without removing controls.

### Keyed HMAC lockout identifiers

HMAC would make offline guessing of the composite more expensive but requires
a stable key lifecycle and migration contract. The lockout database already
contains a raw account identifier, and the task adds no raw pair. Versioned
SHA-256 is deterministic across workers and adequate for this scoped release
repair.

## Decision

Proceed with the shared pure trusted-hop resolver, thin existing-config
wrappers, versioned client/login lockout key, and exact MFA reset propagation.
Keep global request-client rewriting deferred under TASK-13144.
