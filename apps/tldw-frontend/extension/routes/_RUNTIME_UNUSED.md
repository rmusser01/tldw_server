# Runtime-unused, but parity-maintained — edit `packages/ui/src/routes/` instead

See `apps/FRONTEND_AUDIT.md` (§6b) and backlog **TASK-12103**.

At **runtime** the Next.js web build does not render these files — the `@/` alias resolves to
`../../packages/ui/src`, so pages mount `packages/ui/src/routes/*`. **Editing a component here has
no effect on the running app;** change the `packages/ui/src/routes/*` version instead.

**Do NOT delete this directory.** Unlike truly-dead code, it is **actively referenced by ~22 tests**:
a few import these modules directly, and ~19 `readFileSync` *parity-guard* tests assert this copy
stays byte-in-sync with `packages/ui/src/routes/*`. Deleting it would cascade-break those suites.

If this copy is ever to be removed, it needs a deliberate follow-up: migrate/retire those parity
tests to target `packages/ui/src/routes/*` first. Tracked in TASK-12103.

(The sibling `../shims/` directory **is** live — do not confuse the two.)
