# Fresh matrix browser wrapper

TASK13260.166 prepares isolated browser execution for the reviewed matrix launcher. This is harness preparation; no matrix browser, source archive, dependency copy, application profile, account or database has been created by these checks. Full execution waits for the repair and acceptance gate.

After ordinary preparation, successful initialization and application startup:

```sh
node .tmp/uat-next-matrix-20260916/matrix-browser.mjs RUN CELL snapshot
```

Use the selected profile's recorded session automatically. Do not pass session flags. Each cell/run has its own source directory, credentials and browser evidence. Use ordinary setup/login/provider UI; do not seed browser authentication or extract stored tokens.

The wrapper redacts profile API keys, JWT/hash secrets, all planned account passwords and the selected PostgreSQL runtime password. Before entering a provider key in the UI, add that key privately to the optional `providerSecrets` array in that profile's mode0600 credentials file. Do not print the credentials file. It also redacts known URL/JSON encodings and JWT-shaped text. This is known-secret redaction, not a guarantee that arbitrary unknown secrets or screenshots can be published safely.

Raw stdout/stderr stays in the selected profile's `browser-evidence` directory as an exclusive mode0600 `.private.txt` file. Displayed text and copied snapshots are redacted. Snapshot references are replaced by paths to redacted copies; paths escaping the owned archive are rejected. Subprocess infrastructure errors suppress partial output. Ordinary CLI failures preserve their exit code. Screenshots still need visual review before sharing.

Verification uses temporary synthetic profiles and a harmless Node subprocess, never a real browser. Initial module-absence RED and one provider-redaction RED were retained. The first green attempt failed because macOS resolves `/var` to `/private/var`; the test fixture now canonicalizes its temporary root and negative cases first prove a valid baseline. Production path checks were preserved. Final targeted run:16 passed,0 skipped. Syntax checks cover wrapper and test. Bandit is a Python analyzer and cannot parse these two JavaScript files; its result is retained without claiming JavaScript security coverage. Manual review covers path confinement, credential handling, process failure output and private evidence.

Existing14 reviewed launcher files and their snapshots must remain identical to the role125 source manifest. The new wrapper adds no automatic matrix gate decision; the operator must satisfy the existing gate before invoking it against a real profile.
