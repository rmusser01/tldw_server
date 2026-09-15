# Targeted private Quick Ingest verification

Existing isolated multi-user browser and API; repair af1e7bb08b, checkout c1b2cef1c2 by the end. Production was held stable throughout. This predates the pending dev-baseline integration and is not a fresh UAT.

- Bob opened an empty wizard, selected the repository synthetic onboarding-uat-note.md through the native file chooser, chose Quick (Extract, no AI analysis), reviewed and started processing. Results show1 succeeded/0failed and an Open in Media source action.
- Bob closed and reopened the wizard: exact completed filename, result and source actions remained. Capture20:03:49UTC.
- Bob used normal Settings Logout. The signed-out login form is retained. The runner entered the isolated admin credentials with the redacting helper and clicked Login; private values are not retained.
- Settings exposes no Quick Ingest button, so the runner returned to the main app before checking it. The post-login Quick Ingest20:08:41 shows the empty Add step/Configure0, no Bob filename and no previous source action. The admin identity is runner-reported from the selected isolated credential helper; this bundle contains no independent post-login principal/role capture.

This verifies the original completed-result disclosure scenario and same-account close/reopen. No stale result action was clicked under admin. Controlled account/target/lifetime/rotation regressions and independent review remain separate evidence; this does not certify real extension popup destruction or another full matrix.

Two automation waits used incorrect control assumptions: visible Start Processing has accessible name Start processing, and Settings has no Quick Ingest button. The runner inspected fresh snapshots and used the correct controls. Those timeouts are not ingest failures. Captures have trailing blank lines normalized; otherwise unchanged.
