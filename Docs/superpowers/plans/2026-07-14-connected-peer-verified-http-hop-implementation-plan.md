# Connected-Peer-Verified HTTP Hop Implementation Plan

> **Task:** TASK-12971
>
> **Dependency:** TASK-12968.1 (complete)
>
> **Consumer:** TASK-12968.2 (blocked until this plan is complete)

**Goal:** Deliver one reusable async HTTP/1.1 primitive that performs exactly one request hop, dials only a validated DNS address, preserves the approved Host/SNI identity, verifies the actual connected peer, ignores ambient credentials/network configuration, and bounds response processing before materialization.

**Architecture:** Add a new, isolated `Security/http_hop.py` module rather than modifying the legacy cached HTTP clients. Reuse HTTPcore 1.x's documented async streaming and custom-network-backend interfaces: a fresh one-connection pool is created per hop with HTTP/1.1 only and `retries=0`; a small backend replaces the requested hostname with one validated IP while retaining the original hostname in HTTPcore's origin so Host and TLS SNI remain correct. The backend verifies `server_addr` before request bytes are sent. Its stream wrapper independently caps and scans raw response bytes through the final non-informational header block, so aggregate 1xx/final headers and body wire bytes are bounded before HTTPcore/h11 sees them; this security property does not depend on HTTPcore's internal parser constant. HTTPcore handles framing, while this module incrementally bounds content decompression and parser input. Redirects are returned as ordinary responses.

**Tech stack:** Python 3.10-compatible stdlib (`asyncio`, `dataclasses`, `ipaddress`, `ssl`, `zlib`), existing bounded DNS resolver machinery, explicit `httpcore[asyncio]>=1.0.9,<2` and `certifi>=2024.2.2`, pytest/pytest-asyncio, one local raw HTTP smoke server plus deterministic fake HTTP/TLS streams, Ruff, Black, compileall, Bandit.

**Non-goals:** No redirects, retries, connection pooling across calls, proxy support, cookie jar, `.netrc`, browser impersonation, automatic authentication, JSON parsing, result-URL dereferencing, or migration of existing `http_client.py` callers. TASK-12968.2 owns route-policy orchestration and parser semantics; authenticated scraping remains deferred.

## Public contract to deliver

File: `tldw_Server_API/app/core/Security/http_hop.py`

```python
@dataclass(frozen=True, slots=True)
class HTTPHopLimits:
    dns_timeout_seconds: float = 2.0
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 10.0
    write_timeout_seconds: float = 5.0
    total_timeout_seconds: float = 20.0
    max_request_target_bytes: int = 8 * 1024
    max_request_header_bytes: int = 16 * 1024
    max_request_headers: int = 64
    max_request_body_bytes: int = 1024 * 1024
    max_response_header_bytes: int = 64 * 1024
    max_response_headers: int = 128
    max_wire_bytes: int = 2 * 1024 * 1024
    max_decompressed_bytes: int = 4 * 1024 * 1024
    max_parser_input_bytes: int = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class NormalizedHTTPHopRequest:
    scheme: Literal["http", "https"]
    host: str
    port: int
    method: Literal["GET", "HEAD", "POST"]
    target: str
    headers: tuple[tuple[str, str], ...] = ()
    body: bytes = b""
    limits: HTTPHopLimits = field(default_factory=HTTPHopLimits)


@dataclass(frozen=True, slots=True)
class HTTPHopResponse:
    status_code: int
    headers: tuple[tuple[str, str], ...]
    body: bytes
    resolved_ips: tuple[str, ...]
    connected_ip: str
    response_header_bytes: int
    wire_bytes: int


class HTTPHopError(Exception):
    code: HTTPHopErrorCode
    retryable: bool


async def request_http_hop(
    request: NormalizedHTTPHopRequest,
) -> HTTPHopResponse: ...
```

The public function accepts exactly one pre-parsed request object rather than a raw URL or transport configuration. Construction rejects non-canonical scheme/host/method/port/target/header forms, userinfo-shaped or legacy numeric hosts, CR/LF injection, duplicate headers, caller-controlled `Host`/framing/connection headers, and limit values that are non-finite, boolean, non-positive, or internally inconsistent. Explicit route-policy headers are the only header input; there is no ambient client/session input. Tests use a module-private execution seam with injected resolver/backend protocols; those controls are not exported and cannot broaden the production call.

`response_header_bytes` is the aggregate raw plaintext byte count for every informational response plus the final response status/header block. `wire_bytes` is the raw plaintext byte count after the final header terminator, including transfer framing and trailers. The wrapper requests at most the smaller of its fixed read chunk and the applicable remaining ceiling plus one, and fails before passing overflow bytes to HTTPcore. For an accepted response, decompressed and parser-input counts are both `len(body)`; separate response fields would duplicate that value, although their independent ceilings remain enforced internally.

## Stage 1: Freeze contracts, dependency floor, and DNS policy

**Goal:** Make invalid requests and unsafe DNS sets fail before a socket can be opened.

**Success criteria:** The public dataclasses/error codes are immutable and Python 3.10-compatible; HTTPcore 1.x and the explicit CA bundle are direct supported dependencies; DNS is offloaded from the event loop, resolves once, retains the complete A/AAAA answer set, and rejects the entire set when any address is private, reserved, transition/translated, malformed, or otherwise non-global.

**Tests:** Constructor/limit validation, legacy numeric and malformed IP forms, canonical public IPv4/IPv6, empty/private/mixed/malformed resolver answers, deduplication, one resolver call, DNS timeout/saturation compatibility, and the dependency floor.

**Status:** Complete

### TDD tasks

1. Add failing contract tests in `tldw_Server_API/tests/Security/test_http_hop_contract.py` for:
   - canonical `http`/`https`, ASCII IDNA host, explicit port, uppercase `GET`/`HEAD`/`POST`, origin-form target, immutable header tuples, and bounded byte bodies;
   - rejection of raw/uppercase/trailing-dot/zone-ID/userinfo-shaped hosts, bracket misuse, legacy numeric IPv4 (`127.1`, integer, octal, hex), CR/LF, forbidden hop-by-hop overrides, invalid ports, and invalid limits;
   - stable `HTTPHopError.code`, bounded generic text, and no request target/header/body/cause leakage.
2. Add failing dependency-floor assertions to `tldw_Server_API/tests/Security/test_dependency_security_floor.py` for the direct `httpcore[asyncio]>=1.0.9,<2` and `certifi>=2024.2.2` requirements.
3. Run RED:

   ```bash
   source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
   python -m pytest -q \
     tldw_Server_API/tests/Security/test_http_hop_contract.py \
     tldw_Server_API/tests/Security/test_dependency_security_floor.py
   ```

4. Add both direct dependencies in `pyproject.toml` beside `httpx`.
5. Add a public, side-effect-free raw-address resolver wrapper in `tldw_Server_API/app/core/Security/egress.py` that reuses the existing bounded resolver slots without reading egress allowlists/profiles. Preserve `_resolve_host_ips()` compatibility and add its focused regression to `tldw_Server_API/tests/Security/test_egress.py`.
6. Offload that blocking resolver wrapper with `asyncio.to_thread()` in the default hop resolver. Add an event-loop heartbeat test plus DNS timeout/wall-clock coverage; document that an OS resolver worker may finish after caller cancellation while the existing resolver-slot cap remains held until worker completion.
7. Implement only the request/limit/error contracts, raw DNS adapter, canonical IP-set validation, and the module-private injected resolver seam in `http_hop.py` until the Stage 1 tests pass. Keep the public `request_http_hop(request)` signature transport-free.
8. Run GREEN plus Ruff, Black, compileall, and `git diff --check` on the touched Stage 1 files.

### Commit

`feat(security): define secure one-hop HTTP contract`

## Stage 2: Bind one physical request to the validated peer

**Goal:** Dial one selected member of the validated set without hostname re-resolution while keeping the route hostname as Host and TLS SNI.

**Success criteria:** A fresh HTTPcore pool performs at most one connection/request with `retries=0`, `http1=True`, `http2=False`, and no proxy; the delegate backend receives the selected IP and approved port; absent/malformed/mismatched peer IP **or port** closes the stream and fails before HTTP bytes; HTTPS `start_tls()` requires the approved hostname; redirect responses are returned and never followed. TLS uses a new `ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)` loaded only from the explicit certifi path, with hostname checking, certificate verification, TLS 1.2+, and no client certificate or key logging.

**Tests:** HTTP and HTTPS fake-stream tests, alternate ports, IPv4/IPv6 peer tuple normalization, absent/malformed/mismatched/wrong-port peer metadata, rebinding to a private or different public peer, original-host SNI, exact DNS/IPv4/bracketed-IPv6 Host headers on default/alternate ports, one request line despite multiple legal stream writes, redirect destination untouched, `101` rejection, connect/TLS/read cancellation cleanup, and zero retry sleeps.

**Status:** Complete

### TDD tasks

1. Add failing transport tests in `tldw_Server_API/tests/Security/test_http_hop_transport.py` using HTTPcore's public async backend/stream protocols. The fake stream must record `connect_tcp`, `start_tls`, writes, closes, and `server_addr` without using external network.
2. Run RED for the new file and retain the observed failures in `.superpowers/sdd/progress.md`.
3. Implement the private validated-address backend/stream wrapper in `http_hop.py`:
   - assert HTTPcore requests only the approved origin/port;
   - dial only the already-validated selected IP;
   - verify `server_addr` immediately and after TLS wrapping;
   - reject Unix sockets and any retry sleep path;
   - cap individual network reads to a small fixed chunk; Stage 3 applies the remaining header/wire ceiling plus one once header-boundary state exists;
   - expose no proxy, UDS, client certificate, or externally supplied SSL-context parameters.
4. Build request framing internally. Reject caller `Host`, `Content-Length`, `Transfer-Encoding`, `Proxy-Authorization`, `Connection`, `Proxy-Connection`, `TE`, `Trailer`, `Upgrade`, `Keep-Alive`, `Expect`, and `Accept-Encoding`; cap target bytes, caller header count/bytes, and body bytes before constructing the HTTPcore request. Emit an explicit correctly bracketed Host header, controlled `Connection: close`, controlled `Accept-Encoding: identity` until Stage 3 installs bounded decoders, and generated body framing.
5. Build an environment-independent TLS client context directly with the explicit certifi CA path; do not call `ssl.create_default_context()`, accept an external SSL context, load a client certificate, or enable key logging.
6. Implement the one-use HTTPcore backend and pool call with per-operation timeout extensions and guaranteed pool/response close on success, error, timeout, and cancellation. The one-use backend must fail closed if HTTPcore's separate `ConnectionNotAvailable` reassignment loop attempts another physical dial. Persist the peer verified by the stream wrapper as response evidence; do not add a redundant post-response peer lookup.
7. Run GREEN plus the legacy MCP docs-fetcher tests and focused existing egress/http-client security tests to prove compatibility.

### Commit

`feat(security): bind HTTP hops to validated peers`

## Stage 3: Bound headers, streaming, decompression, and parser input

**Goal:** Enforce all response ceilings during streaming and expose only bounded typed failures.

**Success criteria:** The network wrapper independently bounds the aggregate bytes/count of all informational and final response headers before HTTPcore/h11 parsing; `Content-Length` can reject an oversized encoded entity before body iteration; raw wire counters remain authoritative when length is absent or false; identity, gzip, and zlib-wrapped deflate are decoded incrementally with bounded `decompress()` calls, including a bounded empty-input final drain; stacked/unknown/truncated/concatenated encodings fail closed; parser input is never larger than its explicit ceiling; transport exceptions and timeouts become stable sanitized errors. One total deadline covers DNS through response finalization and close. HTTPcore's raw DEBUG wire traces remain disabled by central logging policy because they include complete response headers and raw parser failures.

**Tests:** Repeated `100`/`103` headers plus a final response, aggregate header byte/count overflow, status-line/reason accounting, underreported/malformed/duplicate length, chunked/no-length streaming, raw wire oversize, decompressed oversize, parser oversize, gzip bomb, invalid/truncated/concatenated gzip, zlib-wrapped deflate, raw-deflate rejection, adversarial chunk boundaries, output produced during the final bounded empty-input drain, unsupported/stacked encodings, malformed protocol, first-byte/idle/total timeout, bounded error strings, HTTPcore success/failure trace redaction, cancellation cleanup, and RFC-compliant `205` zero-content framing with `HEAD` precedence.

**Status:** Complete

### TDD tasks

1. Add failing streaming tests in `tldw_Server_API/tests/Security/test_http_hop_streaming.py`; generate compressed payloads in memory and use event barriers for timeout/cancellation cases.
2. Run RED and record the exact failure count.
3. Implement a narrow raw-stream response guard—not a second HTTP body parser—that scans complete header blocks only far enough to distinguish informational from final status. Count every status line, reason phrase, header line, terminator, and informational block cumulatively; request at most the smaller of the fixed read chunk and the applicable remaining ceiling plus one, and fail before forwarding overflow bytes to HTTPcore. Continue enforcing the raw wire ceiling after the final header terminator, including transfer framing and trailers.
4. Normalize/count the final headers returned by HTTPcore and apply encoded `Content-Length` preflight without logging or returning header values in errors.
5. Iterate `response.aiter_stream()` once for content decoding; do not use that post-transfer stream as the raw wire counter.
6. Implement bounded identity/gzip/zlib-deflate decoders with `zlib.decompressobj(...).decompress(..., max_length=remaining + 1)`, `unconsumed_tail` handling, an empty-input `decompress(b"", max_length=remaining + 1)` final drain, explicit EOF/unused/trailing-data checks, and no `Decompress.flush()` or unbounded `zlib.decompress` call. `Decompress.flush(length)` is forbidden here because `length` controls only the initial output-buffer allocation and does not cap returned output.
7. Enforce `max_decompressed_bytes` and `max_parser_input_bytes` before extending the returned buffer. Return only bounded bytes after the entire accepted stream finishes.
8. Wrap the entire internal hop—DNS, connect, TLS, headers, body, decompression, and close—in one total `asyncio.wait_for()` deadline. Map HTTPcore, DNS, TLS, protocol, decompression, and timeout failures to a small stable `HTTPHopErrorCode` vocabulary. Preserve `CancelledError`; never include URL query, headers, body, filesystem paths, credentials, or upstream exception text.
9. Run GREEN and static checks for all Stage 1-3 files.

Header count means physical field/continuation lines across informational and final blocks. `wire_bytes` means plaintext bytes actually read after the final header terminator, including chunk framing and trailers; bytes a peer never sends or that HTTP/1.1 framing causes the client never to read are unknowable. Trailers are bounded by raw wire bytes and h11's parser, not by `max_response_headers`. The raw guard rejects duplicate `Content-Length` lines before h11 normalizes identical values. A non-`HEAD` `205` may use fixed zero length or the canonical five-byte empty chunk terminator; chunk extensions, trailers, decoded content, and coalesced trailing bytes fail closed. `HEAD` remains header-terminated. `asyncio.wait_for()` may exceed the nominal deadline while it waits for cancellation cleanup to finish.

### Commit

`feat(security): bound one-hop response streaming`

## Stage 4: Prove isolation, concurrency, and integration; finalize

**Goal:** Demonstrate that the primitive's security claims hold under real local HTTP I/O, deterministic TLS transport tests, and concurrent failure paths, then complete TASK-12971.

**Success criteria:** One local HTTP smoke test plus deterministic HTTP/HTTPS transport tests prove direct validated-address dialing, Host/SNI, peer verification, and no redirect follow; ambient proxy/netrc/cookie/auth/client-cert state has no effect; concurrent calls do not share requests, streams, counters, or credentials; cancellation and failures release resources; focused and compatibility suites, lint/format/compile, dependency checks, Bandit, and adversarial review pass.

**Tests:** One raw `asyncio.start_server()` HTTP smoke server bound to `127.0.0.1:0` with test-only public-address classification; deterministic fake TLS streams for SNI/context checks; one combined temporary HOME/netrc plus ambient proxy/auth/cookie/CA/client-cert/keylog-variable isolation test; concurrent event barriers/counters; cancellation and cleanup assertions. IPv6 transition/mapped/NAT64 cases remain pure unit tests so CI does not require IPv6 sockets.

**Status:** Complete

### TDD and verification tasks

1. Add failing integration/isolation/concurrency cases to `test_http_hop_transport.py` and `test_http_hop_streaming.py`. Use test-only monkeypatching for loopback classification; production code must never expose a `block_private=False` escape hatch.
2. In one combined ambient-state test plus public-signature inspection, prove that `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, `NO_PROXY`, `.netrc`, ambient cookies/auth, `SSL_CERT_FILE`, `SSL_CERT_DIR`, `SSLKEYLOGFILE`, and client-certificate variables/configuration cannot alter the route, headers, trust roots, key logging, or TLS client identity. HTTPcore exposes no trust-env, cookie-jar, netrc, auth, or client-cert environment path, so test that absence rather than emulating unused client features. Also prove explicitly supplied route-policy Authorization remains explicit rather than ambient.
3. Run the complete focused suite:

   ```bash
   source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
   python -m pytest -q \
     tldw_Server_API/tests/Security/test_http_hop_contract.py \
     tldw_Server_API/tests/Security/test_http_hop_transport.py \
     tldw_Server_API/tests/Security/test_http_hop_streaming.py \
     tldw_Server_API/tests/Security/test_egress.py \
     tldw_Server_API/tests/Security/test_egress_env_absent_defaults.py \
     tldw_Server_API/tests/Security/test_egress_global_env.py \
     tldw_Server_API/tests/Security/test_dependency_security_floor.py \
     tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_fetcher.py \
     tldw_Server_API/tests/http_client/test_http_client.py \
     tldw_Server_API/tests/http_client/test_http_client_stream_timeouts.py
   ```

4. Run static/security checks:

   ```bash
   source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
   ruff check \
     tldw_Server_API/app/core/Security/http_hop.py \
     tldw_Server_API/app/core/Security/egress.py \
     tldw_Server_API/tests/Security/test_http_hop_contract.py \
     tldw_Server_API/tests/Security/test_http_hop_transport.py \
     tldw_Server_API/tests/Security/test_http_hop_streaming.py \
     tldw_Server_API/tests/Security/test_egress.py \
     tldw_Server_API/tests/Security/test_dependency_security_floor.py
   black --check \
     tldw_Server_API/app/core/Security/http_hop.py \
     tldw_Server_API/app/core/Security/egress.py \
     tldw_Server_API/tests/Security/test_http_hop_contract.py \
     tldw_Server_API/tests/Security/test_http_hop_transport.py \
     tldw_Server_API/tests/Security/test_http_hop_streaming.py \
     tldw_Server_API/tests/Security/test_egress.py \
     tldw_Server_API/tests/Security/test_dependency_security_floor.py
   python -m compileall -q \
     tldw_Server_API/app/core/Security/http_hop.py \
     tldw_Server_API/app/core/Security/egress.py \
     tldw_Server_API/tests/Security/test_http_hop_contract.py \
     tldw_Server_API/tests/Security/test_http_hop_transport.py \
     tldw_Server_API/tests/Security/test_http_hop_streaming.py
   python -m bandit -r \
     tldw_Server_API/app/core/Security/http_hop.py \
     tldw_Server_API/app/core/Security/egress.py \
     -f json -o /tmp/bandit_task_12971.json
   git diff --check
   ```

5. If Python 3.10 and 3.12 interpreters are locally available, run the three focused hop test files under both. Otherwise record the missing local matrix explicitly rather than claiming it ran.
6. Perform two reviews before finalization:
   - correctness/security review against every TASK-12971 acceptance criterion;
   - simplification review for custom parsing, duplicate policy, unnecessary dependency/API surface, and legacy-client coupling.
7. Update the canonical design and the blocked TASK-12968.2 plan with the delivered public import/test paths, update `.superpowers/sdd/progress.md`, and record verification plus known limits in Backlog.md. This supplies the prerequisite; TASK-12968.2 performs the later gateway-consumption proof after this task is complete.
8. Mark every stage complete only from fresh evidence, commit the final review fixes, then mark TASK-12971 Done. Do not begin TASK-12968.2 until this primitive's focused tests are green and its public contract is stable.

### Commit

`test(security): prove one-hop transport isolation`

## Final review checklist

- [x] One call resolves once and sends at most one physical HTTP request.
- [x] Every resolved address is canonical and globally routable; mixed sets fail closed.
- [x] The delegate receives only a validated IP; Host and TLS SNI retain the route hostname.
- [x] Missing/mismatched connected-peer metadata fails closed and closes the stream.
- [x] Redirects and retryable statuses are returned without an internal second hop.
- [x] No ambient proxy, netrc, cookie, authorization, client certificate, cached client, or SDK state exists in the call path.
- [x] Response header, encoded entity, decompressed body, parser input, operation, and total-time bounds are all tested.
- [x] Oversized/compressed-bomb bodies stop before full materialization.
- [x] Errors are typed, generic, bounded, and secret/query/body/path free.
- [x] Existing HTTP-client callers and MCP docs-fetcher behavior remain unchanged.
- [x] Ruff, Black, compileall, focused tests, compatibility tests, `git diff --check`, and Bandit pass.
- [x] TASK-12968.2's blocked plan names the delivered import/test paths and requires later gateway consumption instead of `afetch_json`; the actual integration proof remains in TASK-12968.2 after this prerequisite completes.
