# Skills Live Integration Certification

Run the explicit local gate from the frontend package:

```sh
cd apps/tldw-frontend
bun run e2e:skills:certify
```

Use an active local project `.venv`, installed Bun workspace dependencies, and installed Playwright Chromium. The runner performs tracked Chromium probes from both the frontend and extension packages; a missing browser fails preflight rather than skipping.

The gate starts one disposable backend, builds the production Chrome extension, and retains sanitized evidence at `test-results/skills-certification/<run-id>`. It requires zero skipped tests and successful cleanup for a passing exit status.

The certified lifecycle uses dry render only; it does not execute real models or tools. This command is an operator-run release gate and is not part of default PR CI.
