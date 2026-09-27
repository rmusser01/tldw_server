# Buddy follow-up: import credits and lifecycle evidence

Tasks: TASK-13242 (credit repair), TASK-13211 (open lifecycle investigation),
TASK-13227 (open native qualification). Architecture: [ADR-006](../../backlog/decisions/006-buddy-artwork-credit-portability.md), extending ADR-005.

## Findings and repair

On dev `50c1f689575b1bc21ed3e78cdb193b03fe968cdd`, the published Trenchcoat
archive imported successfully through the real authenticated HTTP worker, but
the import discarded its embedded creator/source/license record. Independent
Buddy creation returned 201 without those credits, and a native export returned
200 without them. This is separate from Chatbook's already-merged path-import fix.

The repair stores bounded artwork credits in existing manifest JSON, snapshots
them into independent Buddy attribution, and restores the supported native
carrier on export. It copies no unrelated source context. Export strips the
internal manifest field from a copy because Chatbook strictly validates animation
root keys; credits are explicitly included in the export fingerprint.

PostgreSQL testing also exposed native export passing `datetime` values to JSON.
Only exported pack/asset timestamp fields now normalize those values to ISO text;
SQLite strings remain unchanged. PostgreSQL import-job metadata remains explicitly
unsupported by its existing repository. No migration or import-job backend expansion.

Already-imported missing credits cannot be reconstructed. Re-import the original
credited archive and make a new independent copy. Originals without embedded
credits still require their accompanying notice files.

## Verification

- RED: 10 credit-focused SQLite cases failed before implementation, including
  missing imported credits and invalid metadata being ignored.
- An intermediate focused run passed 111 cases and exposed the PostgreSQL
  timestamp failure. After the fix, the final portability file passed **45 tests**,
  including PostgreSQL snapshot/export/preview. The adjacent independent-Buddy,
  visual-manifest and asset-remapping files supplied **72 passing cases** in the
  earlier focused run. No full suite was run.
- Ruff, Black and Bandit passed on touched code; Bandit reported zero findings.
- Independent review found native export incompatibility and a new rejection of
  legacy credit-free source context. Both received regression coverage and were
  resolved; final review reported no blockers.
- Real HTTP on a disposable SQLite server: preview → worker commit → independent
  Buddy copy → worker export → downloaded archive → worker re-import all completed.
  The exact creator, source URL, license and **11,654 UTF-8 bytes of notices**
  survived. Every original PNG byte hash matched.
- The final exported archive passed Chatbook's actual native importer and draft
  validation: one asset, 18 states, activatable, exact credits, staging cleanup.
  Code came from Chatbook `fc7e0a0c2a3c2f03a2b5c7670a50c680760e9daf`; its complete
  Persona_Visual package has no diff from fetched dev `02374bf66a`.

The [machine-readable receipt](artifacts/buddy-credits-13242/verification.json)
records source file hashes and the exact archive hashes. Runtime source was the
pinned server base plus the recorded repair files. No provider credentials,
physical microphone or external model account were used. Service-level native
import is not a terminal UI walkthrough.

Reproduction command for focused storage verification (using a disposable local
PostgreSQL fixture with its test connection environment):

```sh
source .venv/bin/activate
TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest \
  tldw_Server_API/tests/Persona/test_persona_visual_portability.py -q
```

## Legacy request-loop investigation

A clean frozen-lock WebUI archive of server dev `50c1f68957` ran in webpack
development mode on loopback 18180 against the disposable backend on 18181.
Temporary console probes recorded legacy host mount/cleanup, pack-list effect
inputs, session-list inputs and context publications. Probes were confined to
the disposable frontend, not shipped as a production change.

In the actual in-app browser on `/persona`:

| UTC | Explicit viewport/action | Observed diagnostic |
| --- | --- | --- |
| 14:17:05.212 | 1280 × 800 | Pack list, development cleanup/remount, second pack list; one session list at .216 |
| 14:17:20.447 | 1023 × 800 | Legacy host cleanup; dock removed |
| 14:17:27.347 | 1024 × 800 | Host mount, development cleanup/remount, two pack lists; one session list at .351 |

This establishes breakpoint-driven remounts as one source of paired reloads. It
does **not** reproduce the historical ~250 ms loop or establish its initiating
trigger. No rate-limit failure was observed during these transitions. Later
context-null logs coincide with deliberate backend restarts for credit testing
and do not count as spontaneous loop evidence. The viewport override was reset.

Source review confirms pack-list dependencies are Persona identity, target
availability and refresh nonce. A surface-only change reloads sessions, not
packs. A normal readiness poll has no 250 ms retry. The original incident's
frontend SHA was unavailable locally, so no exact-source historical diagnosis is
claimed. TASK-13211 remains open; do not apply a speculative lifecycle fix.

## Remaining qualification

Native Chrome control returned **Computer Use permissions are not granted**.
Installed-extension and native-terminal fresh/upgraded journeys remain open.
Physical voice qualification also remains open; no capture was started here.
The direct downloaded-pack upload option in independent Buddy management is a
remaining UX opportunity; this repair documents the current supported API route.


## PR #2940 review follow-up

Qodo requested explicit helper contracts and typed/documented credit tests. Those
are now present. A framework-neutral central `PersonaArtworkValidationError`
retains `ValueError` compatibility and stable messages; it is re-exported from
central exceptions without importing FastAPI into the artwork validator. Nine
invalid-record cases first failed when required to raise the domain type, then
all 44 SQLite portability cases passed after the change. Ruff, Black and touched
Bandit checks passed with no findings.

Qodo's task-marker report exposed mixed Backlog implementations: the installed
JavaScript CLI emits `SECTION:NOTES`; the Python MCP/CLI accepts that alias and
emits `SECTION:IMPLEMENTATION_NOTES`. The supported Python editor normalized the
two actually damaged tasks (13211/13227), and every historical notes line was
verified retained. Task13242's original single NOTES section was already valid.

The docs CI failure was a relative link to an API page excluded from the curated
site. The link now uses its existing public GitHub destination. Local exact docs
gate retries hit a macOS multiprocessing semaphore allocation failure before
building pages (`SemLock: ENOSPC`, with 290 GiB disk available), including outside
the sandbox. This is not evidence of a passing strict build; the Linux CI gate
must verify it. The cancelled license-audit run had no executable steps/log, so
no license policy was weakened or changed.

The earlier source-hashed HTTP/native receipt remains tied to its recorded
implementation. These review changes preserve successful data/animation behavior
and tighten exception typing; they do not retroactively replace its source hashes.
