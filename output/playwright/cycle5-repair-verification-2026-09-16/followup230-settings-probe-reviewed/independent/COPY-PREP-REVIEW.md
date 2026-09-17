# Copy-only preparation follow-up

**Prior finding closed.** `prepare-copies.mjs` now compares the sorted full `.pth` filename/content-hash inventory against `reviewed-pth-inventory.json` before copying the Python environment. This catches both additional filenames and changed existing hooks, including project-source paths that do not begin with `__editable__`.

The pinned **seven** entries exactly match the currently installed files. The reviewed contents comprise the two known editable project hooks that will be renamed only in the copy, plus `_virtualenv`, `coloredlogs`, `distutils-precedence`, `google_auth` namespace setup, and `pytest-cov`. No extra project hook is present. My earlier quick message incorrectly said eight; the exact verified count is seven.

The original bounded source review remains clear: exact released-commit archive, fresh dependency/cell destinations, preserved relative dependency geometry, copied absolute self-link replaced locally, only copied caches removed, and no original donor mutation. Source manifests are generated before copying dependencies. This is source/read-only metadata review, not proof of clone success, relocation, imports or application startup. The documented post-copy origin/symlink checks remain required.

`copy-script-review-closure.json` binds the exact script/inventory hashes and records the absent release gate. No preparation script, archive, dependency copy, profile, browser, runtime, or database action was executed by this review.
