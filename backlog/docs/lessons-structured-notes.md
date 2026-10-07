# Structured Notes lessons

## TASK-13517: execute the carrier set admitted by preflight

During Task6 independent review, a root `tldw.note.v1` envelope containing a wrapper-shaped extra field passed portable preflight, but wrapper-first extraction later executed nested `tldw.note.v99` records. The same bytes therefore selected different records at validation and mutation. Fix round1 reproduced eight real API failures across `notes`, `data`, `items` and `results` with both `json` and `portable_note`, each preceded by a valid legacy batch item. After the shared extractor gave declared root envelopes precedence and execution reused retained preflight rows, all eight controls passed and canonical SQLite notes contained only the admitted root and earlier valid record. Browser import-preview controls passed for all four wrappers.

Consume the exact carrier set admitted by preflight through execution. Give declared root envelopes consistent precedence in shared extraction and preview; preserve existing unversioned container behavior. Do not validate one selection and then independently extract another before writes. Evidence: `/private/tmp/f6-fix1-s-red.log`, `/private/tmp/f6-fix1-s-green.log`, `/private/tmp/f6-fix1-s-covering.log`; full Task6 fix report records the immutable reviewed base and final covering results.

## TASK-13517: opaque modes must not consult object prototypes

The source filename helper indexed a plain JavaScript extension table with the persisted opaque language string. Review found that `constructor`, `toString` and `__proto__` resolved inherited properties and produced source extensions outside the known-mode contract. Three parameterized cases failed before the fix and passed after `Object.hasOwn` guarded the existing lookup. They also assert that portable output retains each exact language string and source text remains exact.

Check own known-mode membership before using an object as an extension table. Every unknown opaque mode uses the `.txt` source fallback; portable records preserve the raw mode. Evidence: `/private/tmp/f6-fix1-u-red.log` (three extension failures plus four preview failures), `/private/tmp/f6-fix1-u-green.log`, `/private/tmp/f6-fix1-u-covering.log` (38 tests passed).
