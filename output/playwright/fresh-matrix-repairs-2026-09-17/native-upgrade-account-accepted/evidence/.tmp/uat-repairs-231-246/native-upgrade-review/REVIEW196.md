# TASK13260.196 independent review

**Verdict: clear for the parent's separately authorized targeted-upgrade preparation/start. No actionable findings.** This review does not execute that preparation/start or certify native acceptance.

## Exact target

Owned manifest SHA256: `989b46b66e5fed60854dd2132852376e35f9dfb4fb700d5c0b42d49928155cd9`.
Helper SHA256: `1d3254bc0c7403770348132974b42ff28568051fc5e3a06a829f920b410c01e6`.
All three owned files match; see hashes.json. Removing exactly three new export keywords makes matrix-launcher.mjs byte-identical to its retained baseline.

## Verified behavior

- Upgrade supports only backend/frontend and PG single/multi, with distinct original/copy run identities. It calls requireInitialized(original) and runtimeEnv(original), preserving original initialization, credentials, config, databases, runtime role, ports and API cwd. It does not invoke provisioning, initialization, reset, holder release, or process cleanup.
- Original source fingerprints and original profile/init/holder hashes remain distinct from the new source proof. A changed immutable binding refuses reuse. Binding also records exact launcher/helper bytes; spawn receipts truthfully distinguish starting, started, failure, interruption and exit, without claiming readiness.
- Released targeted-upgrade policy, cell, revision, copy completion, archive hash, tracked file hashes/links and reused Python identity are checked before spawning. The existing origin inspector supplies new Python roots and Next CLI/cwd; frontend environment masks archive dotenv values and uses a separately owned build directory. Existing API/browser origin remains unchanged.
- Original PG holder checks remain intact, followed by the existing live role validator for both auth/content databases. Restricted direct-login role requirements are unchanged. No provisioning identity is forwarded to the app.
- Known archive-relative data blocks relocation rather than silently moving or discarding it. Symlinked storage parents and dangling/nonempty locks fail closed. Existing ports and unowned build state are rejected; signal forwarding targets only the newly spawned child and removes only its own listeners.

## Independent checks

From repository root:

```sh
node --experimental-vm-modules --test .tmp/uat-repairs-231-246/native-upgrade-preparation/matrix-upgrade.test.mjs .tmp/uat-next-matrix-20260916/matrix-launcher.test.mjs .tmp/uat-matrix-browser-20260917/browser-wrapper.test.mjs
```

**124 passed, zero failures/skips, exit0** (42 new upgrade controls plus82 existing launcher/browser controls), retained in combined.log. Tests use synthetic filesystem records and fake inspection, role, socket, child or browser CLI boundaries. None invoked the actual inspector, provisioner, launcher CLI, browser or database.

Independent Node syntax checks: all3 pass. Independent installed ESLint/root-config lintText: all3 parsed,0errors/0warnings; exact export-only attribution passes (static.json). Author's scoped Bandit receipt correctly discloses3 JavaScript parse limitations and provides no JavaScript security certification.

## Boundaries and documentation

The copy gate/manifest remain trusted operator-produced provenance, not signed attestations; this helper validates recorded bytes rather than independently deriving a complete tree from Git. The storage inventory is explicitly bounded. Existing native record readback and health/auth proof remain the parent's work after approved launch; this is upgrade acceptance, not fresh initialization or a full matrix run.

At first read, IMPLEMENTATION196.md still listed123 tests and referenced a not-yet-written commands.json. Author confirmed packaging was incomplete, supplied the exact frozen124 command, and corrected prose without changing source/test hashes. The independent result and frozen source hashes above govern this review.

No source/test/task/profile/runtime/database/archive/browser/Git changes were made by this reviewer. Only this review packet and synthetic temporary test fixtures were written.
