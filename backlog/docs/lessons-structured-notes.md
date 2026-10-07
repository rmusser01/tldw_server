# Structured Notes lessons

## TASK-13517: execute the carrier set admitted by preflight

During Task6 independent review, a root `tldw.note.v1` envelope containing a wrapper-shaped extra field passed portable preflight, but wrapper-first extraction later executed nested `tldw.note.v99` records. The same bytes therefore selected different records at validation and mutation. Fix round1 reproduced eight real API failures across `notes`, `data`, `items` and `results` with both `json` and `portable_note`, each preceded by a valid legacy batch item. After the shared extractor gave declared root envelopes precedence and execution reused retained preflight rows, all eight controls passed and canonical SQLite notes contained only the admitted root and earlier valid record. Browser import-preview controls passed for all four wrappers.

Consume the exact carrier set admitted by preflight through execution. Give declared root envelopes consistent precedence in shared extraction and preview; preserve existing unversioned container behavior. Do not validate one selection and then independently extract another before writes. Evidence: `/private/tmp/f6-fix1-s-red.log`, `/private/tmp/f6-fix1-s-green.log`, `/private/tmp/f6-fix1-s-covering.log`; full Task6 fix report records the immutable reviewed base and final covering results.

## TASK-13517: opaque modes must not consult object prototypes

The source filename helper indexed a plain JavaScript extension table with the persisted opaque language string. Review found that `constructor`, `toString` and `__proto__` resolved inherited properties and produced source extensions outside the known-mode contract. Three parameterized cases failed before the fix and passed after `Object.hasOwn` guarded the existing lookup. They also assert that portable output retains each exact language string and source text remains exact.

Check own known-mode membership before using an object as an extension table. Every unknown opaque mode uses the `.txt` source fallback; portable records preserve the raw mode. Evidence: `/private/tmp/f6-fix1-u-red.log` (three extension failures plus four preview failures), `/private/tmp/f6-fix1-u-green.log`, `/private/tmp/f6-fix1-u-covering.log` (38 tests passed).

## TASK-13518: source directives and decimal spelling outrank candidate resolution

Browser engine qualification found that yaml 2.8.1 inherits explicit YAML 1.1 into
an undirected next document, while reviewed Python resets each document to its
own directive or 1.2. A regression failed before the browser tracked source
per-document directives and declined the mixed/inherited interpretation before
key comparison. A second probe found that the candidate resolves bare 1.1 `.` as
float NaN, whereas Python resolves it as a string; a naive exact-decimal normalizer
then treated the missing mantissa as zero. Six controls failed before the browser
explicitly refused candidate finite floats without a mantissa digit, then the
whole YAML selection passed 69 tests. Default 1.2 and quoted dots remain strings.

Do not use inherited parser state or a tolerant numeric match to establish
cross-runtime semantics. Keep per-document source metadata, require a real
mantissa for finite numeric identities, and decline unqualified interpretations
without modifying the draft. Evidence: the E3 browser qualification report and
`/private/tmp/tldw-task9-evidence/{version,mantissa}-{red,green}.log`.

### E3 merge review and property-prefix indentation (2026-10-07)

Task 13518 review found that exempting merge processing from the ordinary key set
also admitted repeated `<<` declarations; imported-member overrides do not exempt
the merge declaration itself. The same review found `!!merge x` bypassed the tag's
source matcher. Adjacent real formatting controls then showed yaml 2.8.1 CST key
offsets start after tags/anchors: reindenting only the scalar offset left that
line behind and failed the source-preservation gate. Track merge declarations in
the map-local set, qualify tag values before merge handling, and use retained CST
property tokens when adjusting leading key whitespace. The 91-case YAML selection
and 15 amended bundle controls cover these incidents; generic dumping is no remedy.
