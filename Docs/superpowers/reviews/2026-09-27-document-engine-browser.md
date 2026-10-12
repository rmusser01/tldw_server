# Browser document-engine qualification — E3 / TASK-13593

Date: 2026-10-07. Immutable task base: `bf0965412cb65b6fbfee3b31e5e211dbd2872841`.
Governing decision: [ADR-070](../../ADR/070-portable-note-language-and-negotiated-structured-editing.md).
Status: qualified for pure-engine adoption; worker/editor lifecycle remains E4/I3/I5.

## Candidate acceptance before package adoption

The existing shared ChatMacro/Skills helpers use js-yaml 4.2.0's load/dump APIs.
A real synthetic round trip changed 9007199254740993 to 9007199254740992, -0 to 0,
and quoted escape spelling to plain text. It is not a source editor formatter.
The yaml candidate's generic parse/stringify also rounded the integer, changed
1e400 to .inf and removed the negative-zero/quoted spelling. That path is rejected.
Native JSON.parse objects likewise cannot retain numeric/string source tokens.

Accept jsonc-parser 3.3.1 (MIT, Microsoft) and yaml 2.8.1 (ISC, Eemeli Aro), exact
runtime pins in @tldw/ui, with no runtime transitive dependencies. Before any
package/lock change, their npm tarballs were extracted only beneath
`/private/tmp/tldw-task9-evidence/candidates`; Vitest and bundler aliases pointed
to those extracted distributions. No shared node_modules or environment binding
was changed. The license files were read; MIT and ISC notices remain in their
installed packages. jsonc-parser 3.3.1 ships UMD plus ESM, despite current upstream
main describing an ESM-only distribution. yaml ships browser ESM and Node CJS
conditional entries. The selected actual distributions govern this evidence.

| Distribution | Tarball SHA-256 | License-file SHA-256 |
| --- | --- | --- |
| jsonc-parser-3.3.1.tgz | 4a0315b8671e7463bae7af7c142cdf19e9aa7ba39eb36dc2df383b8648e3cbc9 | c3a24c02d678e5f0711623dcf3ab3f243b273c2633f638a1fa8e35fb7c7f8e4d |
| yaml-2.8.1.tgz | 195759b97d3f2e6085474549c069397ad7af6160be6d7c87442b8c47cb724063 | 5bba27375d93e9119f76c1015f7672cf9ad5f70952296e0842fb2243d6376869 |

Primary references retrieved: [jsonc-parser](https://github.com/microsoft/node-jsonc-parser),
[yaml](https://eemeli.org/yaml/), and [YAML Core resolution](https://yaml.org/spec/1.2.2/#1032-tag-resolution).
The experiments below establish qualification, rather than upstream round-trip claims.

## Owners and preservation checks

The new `apps/packages/ui/src/services/document-editing/engine.ts` exports
`checkDocument`/`formatDocument`, consuming the unchanged E1 request-aware
constructors. The four pure production modules have no UI, filesystem, database,
network or save imports. JSON uses strict visitor options (disallowComments,
allowTrailingComma false, allowEmptyContent false), every error callback and
scanner error; numeric visitor values are discarded. Duplicate decoded keys use
a Set per object, including escaped names and prototype spellings. Formatting
applies candidate edits to the original source, then checks the whole raw token
sequence. JSONL compacts each physical LF/CRLF record separately, preserving its
terminator and the optional final newline; empty source is zero records. Visible
and representation-metadata BOMs are errors for JSONL.

YAML feeds each Lexer lexeme into Parser.next and checks the public collection
stack before Composer receives a document. Composition retains source tokens and
uses exact BigInt integer resolution; no toJS, alias expansion, generic
serialization or supplied custom constructors occurs. Scalar key identities are
tag-aware: integer/boolean/null values remain distinct types, float coefficients
and exponents normalize directly from source strings with BigInt exponent
arithmetic, and strings use decoded scalar source. Binary floating-point values
never determine float-key equality or formatting. YAML 1.2 signs apply only to
decimal integers; signed hex/octal spellings are strings and collide with their
quoted spellings, not decimal keys. Explicit 1.1 keeps its separate resolution.

Formatting edits only existing block indentation and separator spaces. The
complete reparsed CST shape, raw scalar bytes, comments, line breaks, directives,
markers, tags, anchors and aliases must match after excluding indentation/space
trivia. Failure returns no replacement and a source-free message. Formatting is
idempotent; unchanged returns no edit. Alias references are checked against
preceding anchors without traversing/expanding their targets. Explicit merge
overrides do not become ordinary duplicate declarations.

## Supported and refused YAML

Supported: empty streams/comments; single/multiple documents and document
markers; explicit 1.1/1.2 directives; Core scalars, qualified standard scalar and
collection tags; block mappings/sequences and existing flow layouts; anchors,
aliases and symbolic cycles; qualified mapping merges; quoted escapes; large and
high-precision numeric tokens; literal/folded/chomped scalar bytes whose header
indentation is unchanged. Positive tests actually reindent a commented, anchored,
merged multi-document stream, a nested sequence, and CRLF source while preserving
block content and exact number/string spellings. Flow whitespace is retained.

Refused as unsupported: custom/executable and unqualified standard tags or
node/value/tag combinations; complex or alias keys; unqualified merge targets;
mixed versions; inherited 1.1 interpretation for an undirected next document;
indentless sequences; moving a multiline scalar header that would require token
changes; implicit YAML 1.1 timestamps and sexagesimal floats. Undefined aliases
and ordinary duplicate keys are errors. No all-YAML refusal or object dump
fallback is used.

A self-review control exposed yaml 2.8.1's 1.1 directive inheritance: the next
undirected document remained 1.1 in Composer. Reviewed Python resets each
DocumentStartEvent to its own version or 1.2. The new source-directive ledger
refuses the mixed/inherited case before key equality, with one failing control
then a passing whole-YAML run, plus explicit-1.2/undirected-next formatting and
idempotence. Frozen fixtures were not changed.

A late self-review probe found candidate 1.1 finite-float matches without a real
mantissa digit: `.`, `+.`, `-.`, `._`, `e3`, `.e3`. Reviewed Python resolves bare
`.` as string; the browser now explicitly refuses that candidate branch without
inventing a type or duplicate. This is a narrower browser admission profile;
default 1.2 and quoted dots remain supported strings. Six failing controls became
69 passing whole-YAML controls. Scalar keys are validated once so one unsupported
key produces one diagnostic. The earlier 133-case/license/ADR/bundler qualification
preceded package insertion; this late correction and ADR clarification preceded
final acceptance/commit. No production application imported the engine.

## Resource and loading limits

The pure engine enforces 1 MiB UTF-8 source/replacement size, collection depth 100
while parsing, 100 displayed diagnostics (truncation means incomplete), and
cooperative two-second monotonic checkpoints. A deterministic advancing-clock
control exercises the deadline. It has **no hard termination or heap/RSS ceiling**.
Lexing one long token, composition, encoding, cloning, exact scalar conversion,
CST signature creation and cleanup may execute between checkpoints. A separate
terminable worker must enforce E4's deadline and own scheduling/lifecycle. No
editor/save limit, authority, recovery or 200,000-character File Notes rule changes.

Fresh Node 26.0.0 child observations below use the final browser-targeted engine
bundle and dedicated synthetic data. RSS is current whole-process resident bytes,
not a measured peak; no browser responsiveness or worker IPC guarantee follows.
`yaml-deadline` is a large adversarial probe name: it completed within the budget,
not a demonstrated natural timeout. A 738,890-byte YAML mapping reached about
318 MB RSS. Import timings are for the bundled pure module, including both parser
engines; no parser is imported by an application startup owner in this task.

```text
{"name":"import","importMs":2.494791000000003,"bytes":0,"encodeMs":0.011874999999999858,"elapsedMs":0.35837500000000233,"state":"checked","rss":50593792,"heap":5042984}
exit=0
{"name":"json-large-string","importMs":2.5596250000000005,"bytes":1048576,"encodeMs":0.15154199999999918,"elapsedMs":8.608834000000002,"state":"checked","rss":69779456,"heap":14972088}
exit=0
{"name":"json-dense","importMs":2.5315830000000012,"bytes":1048575,"encodeMs":0.18091700000000088,"elapsedMs":72.765,"state":"checked","rss":101695488,"heap":35016928}
exit=0
{"name":"json-depth","importMs":2.6051670000000016,"bytes":203,"encodeMs":0.013124999999998721,"elapsedMs":0.46858299999999886,"state":"resource_limited","rss":50741248,"heap":5095256}
exit=0
{"name":"utf8-size","importMs":2.47325,"bytes":1048580,"encodeMs":0.5435000000000016,"elapsedMs":4.743042000000003,"state":"resource_limited","rss":67960832,"heap":15909320}
exit=0
{"name":"yaml-large-scalar","importMs":2.732333999999998,"bytes":1048576,"encodeMs":0.18833300000000008,"elapsedMs":8.366833,"state":"checked","rss":70287360,"heap":15140384}
exit=0
{"name":"yaml-wide","importMs":2.470875000000003,"bytes":238890,"encodeMs":0.048332999999999515,"elapsedMs":161.112583,"state":"checked","rss":187744256,"heap":62035544}
exit=0
{"name":"yaml-deadline","importMs":2.541708,"bytes":738890,"encodeMs":0.12425000000000352,"elapsedMs":435.13550000000004,"state":"checked","rss":318078976,"heap":169290352}
exit=0
{"name":"yaml-aliases","importMs":2.713707999999997,"bytes":120016,"encodeMs":0.038584000000000174,"elapsedMs":124.58712500000001,"state":"checked","rss":203522048,"heap":83148184}
exit=0
{"name":"yaml-format","importMs":2.8473749999999995,"bytes":138893,"encodeMs":0.05137500000000017,"elapsedMs":170.93908299999998,"state":"formatted","rss":175472640,"heap":57886728}
exit=0
```

Node 26.0.0, Bun 1.3.2, Vitest 4.0.18 and TypeScript 5.9.3 were used. Node's
`--no-experimental-webstorage` disables its newly introduced global storage while
Vitest's existing jsdom/browser fixture owns storage. The final WXT graph contains
88 modules, including 73 yaml and 6 jsonc-parser modules; no Node builtin or
runtime Prettier import. A separate esbuild browser graph has 85 inputs, zero
external imports and 260,687 bytes unminified. These are module-loading observations,
not an application bundle-size promise.

Actual Next 16.1.4 webpack client/SSR compilation passed before adoption using
isolated aliases. After adoption, default Next Turbopack with the unmodified real
configuration loaded the actual installed packages. WXT 0.20.13 used its real
configuration and Vite 5.4.21 production pipeline with a filtered temporary
entrypoint, first with isolated aliases and then normal installed resolution.
Its final engine/probe chunk is 119,882 bytes; the output totals 15.43 MB because
WXT copies the application's existing public assets. Both emitted engines ran in
fresh Chromium 145.0.7632.6 contexts and produced exact JSON precision/negative-zero,
YAML reindent, Core signed-hex duplicate, inherited-version and mantissa-refusal
results, with HTTP 200 and zero page errors. Source was synthetic only.

The WXT bundle was loaded over loopback under `script-src 'self'; object-src 'self'`,
not installed as a live browser extension. No installed-extension lifecycle,
service-worker termination, browser-worker creation, editor history or mounted UI
budget is claimed. Those remain E4/I3/I5. The temporary Next page, WXT entrypoint,
and dedicated Next build directories were moved out of the checkout after checks.

Inherited build notices: WXT reports existing duplicate auto-import names
MediaNavigationFormat and estimateStorageCost; filtered unrelated entrypoints are
listed as skipped. Next reports nine-month-old caniuse-lite data. Those owning
files are unchanged from the immutable base. No notice was suppressed in the
captured logs. Initial harness attempts also showed missing API-origin configuration,
sandbox loopback EPERM, its resulting stale Next lock, and default Turbopack
ignoring webpack-only candidate aliases. The corrected harness uses explicit
webpack for isolated pre-adoption aliases, then normal default Turbopack after
adoption. The app's networking configuration was not changed.

Final emitted artifact provenance (Next's directory was subsequently moved to
`/private/tmp/tldw-task9-evidence/next-turbopack-artifacts`):

```json
[
  {
    "path": "/private/tmp/tldw-server-structured-document-editing/apps/tldw-frontend/.next-live-tier-task9-adopted/dev/static/chunks/[root-of-the-server]__d47e2861._.js",
    "bytes": 52719,
    "sha256": "377a565cf2f3f67c7b45b9504b439ce5fcc70b796328e8df8e62581c6f2ef6c9"
  },
  {
    "path": "/private/tmp/tldw-task9-evidence/wxt-adopted-output/chrome-mv3/chunks/task9-document-engine-qualification-DodxeOlg.js",
    "bytes": 119882,
    "sha256": "bb4cf79bd85b7d34e223e1480b8ad6022e5b0a67e40db2c3fad024e8464dcf62"
  }
]
```

## Verification and comparison

Final explicit contract/JSON/YAML/position selection: **142 passed**, exit 0.
Meaningful RED: JSON behavior 43 failed/5 passed after the missing-export check;
YAML 41 failed/8 passed; inherited-version 1 failed/60 passed; mantissa spelling
6 failed/63 passed; duplicate diagnostic 6 failed/63 passed. Final strict scoped
TypeScript, whole-web typecheck, scoped ESLint with zero allowed warnings,
Prettier 3.2.4 and diff checks pass. Whole-web base and final both pass with the
repository's prescribed 8 GiB Node heap; an initial default-heap run exhausted
memory. New source is checked independently under strict mode, not hidden by the
web app's looser tsconfig or Next's ignoreBuildErrors. No legacy source changed;
there is no introduced legacy static debt. Bandit is not applicable: no Python
production source was changed. No full suite was run.

```sh
# S=/private/tmp/tldw-server-structured-document-editing
NODE_OPTIONS=--no-experimental-webstorage bun run --cwd apps/packages/ui test src/services/document-editing/__tests__/contract.test.ts src/services/document-editing/__tests__/json-document.test.ts src/services/document-editing/__tests__/yaml-document.test.ts src/services/document-editing/__tests__/source-positions.test.ts
# From apps/tldw-frontend (same command at immutable baseline and final):
NODE_OPTIONS=--max-old-space-size=8192 node node_modules/typescript/bin/tsc --noEmit --incremental false --pretty false
# From S, strict scoped config is retained with the full evidence:
node apps/packages/ui/node_modules/typescript/bin/tsc -p /private/tmp/tldw-task9-evidence/tsconfig-adopted.json
BUN_INSTALL_CACHE_DIR=/private/tmp/tldw-task9-evidence/bun-cache bun install --frozen-lockfile --ignore-scripts --cwd apps
```

The freeze check reports 1,891 installs across 1,920 packages with no changes.
Only the two package declarations and their two lock entries were added; their
installed license hashes equal the isolated candidate hashes above. Full exact
commands, complete captured outputs and process exits are in Chatbook's plan
workspace `task-9-report.md`; raw logs/probe scripts/graphs/artifacts are in
`/private/tmp/tldw-task9-evidence`. Task 13593 remains In Progress for independent
controller review.

Comparison record: reviewed Python source `ba744f9059d30e26b1a025561f9dead3696a030c`,
metadata closure `3ffaf527f304644e005b3c2ae8f529e274987417`, durable Python report's
37 actual outcomes. No unchanged Python suite was rerun to regenerate it. A tiny
new resolver-only probe characterized the newly discovered 1.1 mantissa boundary.
Both engines agree on all shared validity/category sets and format admission;
parser-specific error multiplicity and token-width ranges may differ. All spans
below are zero-based Unicode-scalar offsets, end exclusive. The repeated JSON key
and Unicode/CRLF/EOF controls assert exact relevant ranges. YAML complex/custom
and mixed-version rows explicitly remain unsupported rather than false success.

Fixture v1 SHA-256: `fcf516e4c0bb8ff7a19748a4d5f7deb8a30ad0078ea0ce2af0e1e81af50fb0d7`.

| Fixture | Check | Codes | Scalar spans | Format allowed | Format result |
| --- | --- | --- | --- | --- | --- |
| json-large-int | checked | — | [] | true | formatted |
| json-nan | checked | syntax, syntax | [[5,8],[8,9]] | false | blocked |
| json-duplicate | checked | duplicate_key | [[7,10]] | false | blocked |
| jsonl-blank | checked | blank_record | [[3,3]] | false | blocked |
| yaml-stream | checked | — | [] | true | unchanged |
| json-escaped-duplicate | checked | duplicate_key | [[7,15]] | false | blocked |
| json-sibling-keys | checked | — | [] | true | formatted |
| json-comment | checked | syntax | [[7,17]] | false | blocked |
| json-trailing-comma | checked | syntax, syntax | [[7,8],[7,8]] | false | blocked |
| json-infinity | checked | syntax, syntax | [[0,8],[8,8]] | false | blocked |
| json-empty | checked | syntax | [[0,0]] | false | blocked |
| json-scalar | checked | — | [] | true | unchanged |
| json-overflow | checked | — | [] | true | unchanged |
| json-negative-zero | checked | — | [] | true | unchanged |
| json-precision | checked | — | [] | true | unchanged |
| json-escaped-string | checked | — | [] | true | unchanged |
| jsonl-empty | checked | — | [] | true | unchanged |
| jsonl-final-newline | checked | — | [] | true | unchanged |
| jsonl-no-final-newline | checked | — | [] | true | unchanged |
| jsonl-crlf | checked | — | [] | true | unchanged |
| jsonl-scalars | checked | — | [] | true | unchanged |
| jsonl-whitespace-record | checked | blank_record | [[3,5]] | false | blocked |
| jsonl-multiline-record | checked | syntax, syntax, syntax | [[1,1],[5,6],[8,9]] | false | blocked |
| yaml-empty | checked | — | [] | true | unchanged |
| yaml-comments | checked | — | [] | true | unchanged |
| yaml-anchors-aliases-merges | checked | — | [] | true | unchanged |
| yaml-directive-1.1 | checked | — | [] | true | unchanged |
| yaml-directive-1.2 | checked | — | [] | true | unchanged |
| yaml-mixed-versions | unsupported | unsupported | [[35,35]] | false | blocked |
| yaml-complex-key | unsupported | unsupported | [[2,8]] | false | blocked |
| yaml-custom-tag | unsupported | unsupported, unsupported | [[7,14],[15,21]] | false | blocked |
| yaml-standard-tag | checked | — | [] | true | unchanged |
| yaml-literal | checked | — | [] | true | unchanged |
| yaml-folded | checked | — | [] | true | unchanged |
| yaml-chomping | checked | — | [] | true | unchanged |
| yaml-duplicate | checked | duplicate_key | [[5,6]] | false | blocked |
| yaml-malformed | checked | syntax | [[9,9]] | false | blocked |

## Review fix round 1 — merge admission and key-property indentation

Fix base: `9d50710a37cab6bd759b0b55b6762a4c5f64c7d2`. Independent review found
that merge declarations skipped per-map duplicate tracking and explicit merge
scalar tags skipped value qualification. Both shared admission branches are
corrected: merge identities enter the same per-map key set, independently of
symbolic merge-target validation; explicit merge tags use the existing schema
source matcher (`^<<$`), and an unqualified scalar cannot enter merge handling.
Single mapping/alias/sequence merges and explicit imported-member overrides stay
qualified, sibling-map scopes stay separate, and quoted string `"<<"` is distinct.

Adjacent positive controls exposed a pre-existing layout failure: CST mapping
key offsets follow tag/anchor properties, so the formatter left their line's
indentation while moving sibling keys. The base bundle safely returned failed/no
replacement. Controller Ruling 38 authorized a narrow existing-AC3 correction:
the same leading-whitespace helper now also visits tag/anchor tokens in the start
of block-map keys. Exact token spellings, both property orders, inline key/value
properties, flow layouts, block-scalar contents, no-op and idempotence are covered.
The original three explicit-merge reindent expectations were retained.

Whole-YAML RED/GREEN: I1 2 failed/74 passed → 76 passed; I2 7 failed/77 passed
(including three newly exposed layout failures) → 3 failed/81 passed after
admission correction; expanded layout RED 5 failed/86 passed → final **91 passed**.
Final scoped strict types, whole-W 8 GiB typecheck, changed-source ESLint/Prettier
and diff checks pass. No unchanged JSON/position/contract suite, benchmark,
Next/WXT build or browser-lifecycle probe was rerun. Those original observations
above remain evidence for the original head, not the amended source.

The refreshed browser-target engine (esbuild 0.27.2, 85 modules, no externals)
passes 15 focused bundle cases including the exact reproduced holes and adjacent
positive/layout controls. Artifact: `/private/tmp/tldw-task9-evidence/fix1/engine-fix1.mjs`,
also copied to the review probe's `/private/tmp/tldw-task9-evidence/engine-final.mjs`;
260,791 bytes, SHA-256 `2ebc823bdccb659abe3b00d003733a9539ff594d18c83d0a1f49254ae546c1c2`.
Original 260,687-byte bundle is preserved as `fix1/engine-base-9d50710a37.mjs`.
Full commands, outputs, exit JSON and source/artifact hashes live under
`/private/tmp/tldw-task9-evidence/fix1`; the bounded task report has the evidence
index. Shared fixture SHA remains unchanged. Pure-engine resource/lifecycle limits
and deferred inherited tooling warning M1 remain as previously disclosed.
ADR070 already governs this correction; no dependency or ownership boundary changed.
