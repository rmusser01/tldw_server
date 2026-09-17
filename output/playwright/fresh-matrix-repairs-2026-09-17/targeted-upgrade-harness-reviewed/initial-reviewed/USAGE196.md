# Targeted upgrade helper — TASK13260.196

Actual preparation/startup remains parent-gated. The frozen product candidate is `a7d3155a567afb25982eb360ea24b973cc3249c9`. No copying, startup, DB access or browser action was performed during implementation.

The original profile run remains `repairs231-250-targeted-20260917`. Use a distinct parent-selected **copy run ID** with the existing copy-only helper. Its released gate must retain `purpose: targeted-acceptance`, both PG cells, that new run ID/revision, and these additional annotations:

```json
{
  "dataPolicy": "existing-profile-upgrade",
  "originalRunId": "repairs231-250-targeted-20260917"
}
```

After the parent prepares new archives and stops only the exact old app processes, these are the later launch shapes; placeholders are not prepared values:

```sh
node .tmp/uat-next-matrix-20260916/matrix-upgrade.mjs backend pg-single repairs231-250-targeted-20260917 '<approved-new-copy-run>'
node .tmp/uat-next-matrix-20260916/matrix-upgrade.mjs frontend pg-single repairs231-250-targeted-20260917 '<approved-new-copy-run>'
```

Use `pg-multi` for the other existing profile, serially under parent ownership. The helper has no prepare, initialize, bootstrap, reset, holder or cleanup action. Keep original PG holders alive. It retains old API cwd/config/auth/storage/ports; only Python source roots and frontend executable/cwd/build change. The old optional ACP configuration remains unexercised.

New private receipts/logs appear under `targeted-upgrades/<copy-run>/<cell>/`. A started process is not a readiness or native acceptance claim. The parent must retain explicit current-source receipts with browser evidence, verify actual health/auth, and read back original Media 1, Notes/decks/cards/accounts/canonical chats before targeted acceptance. The original browser wrapper and credentials remain unchanged.

If any known archive-local data exists, proofs change, a role is elevated/released, a port is occupied or an unowned build exists, the helper refuses to launch. Preserve the failed attempt and inspect it; do not edit old proofs to recover. Schema/bootstrap or root-local data transfer needs a separately reviewed action, not an automatic retry/reset. Fresh full-matrix acceptance remains separate.
