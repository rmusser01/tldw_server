# UAT261 pre-correction freeze record

This record preserves the prior diagnostic packet identity before the
independent cleanup correction.

| Artifact | Prior SHA-256 |
| --- | --- |
| `uat261_capture.py` | `2fc4903f398201fd16b652c33465628a688b566f0770a6da9d45d933a11224b0` |
| `test_live_capture.py` | `775449099956c773bd0f8fdad944c2689174b6ecf31cb23464bb4714b34a8c34` |
| `RUN.md` | `85cf572d37fad15836dbab2d1ba23ae99adbcce49725e598bc5ca666f2cc2114` |
| `REPORT.md` | `64456f4e11c9b8d70ed1b47ecd5e2e77e7951cdf1662198c9775a3feee955c90` |
| `bandit-source.json` | `76926e68283ff9af41a815d15e251946930a6a2b9669cda6c50041dcba0bfd70` |
| `bandit-all.json` | `6e6afb7f2da42b4fc02a354a947d52edba655959d271993aac38478ec4be26a9` |

The prior report covered a two-call, source-external diagnostic wrapper with a
safe input/terminal projection and intentionally null final-answer field. It
recorded 9 synthetic local tests, Ruff/compile success, source Bandit 0, and
test-only B101 findings in the full diagnostic-directory Bandit output. It did
not make a live provider call.

Independent review then demonstrated a cleanup defect: closing an unstarted
wrapper generator did not reach the underlying provider stream. The corrected
packet adds only explicit sync/async iterator cleanup and the separate
malformed-observation-state marker; it does not alter product source or the
live-call protocol.
