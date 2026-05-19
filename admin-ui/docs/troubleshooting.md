# Troubleshooting

Common issues and solutions when developing or deploying the admin UI.

---

## Authentication Issues

### 401 Unauthorized on every request

1. **Check the API base URL.** Ensure `NEXT_PUBLIC_API_URL` points to the correct backend host and port.
2. **Verify auth mode.** If the backend runs in `single_user` mode, confirm `SINGLE_USER_API_KEY` is set and the UI sends it as `X-API-Key`. In `multi_user` mode, verify the JWT flow.
3. **Check cookie domain.** In production, ensure the auth cookie domain matches the admin UI domain. Cross-origin cookies require `SameSite=None; Secure`.
4. **Inspect the token.** Decode the JWT at [jwt.io](https://jwt.io) and verify `exp` is in the future, `iss` matches the backend, and required claims (`sub`, `role`) are present.

### JWT mismatch after backend restart

The backend generates a new JWT signing key on first start if none is configured. If you restart the backend without persisting `JWT_SECRET`, all existing tokens become invalid.

**Fix:** Set `JWT_SECRET` in the backend `.env` file so it persists across restarts.

### Cookie not sent in development

Browsers block cross-origin cookies when `SameSite` is not explicitly set. In local development:

- Run the admin UI and backend on the same hostname (e.g. both on `localhost`).
- Or configure the backend to set `SameSite=None; Secure` and access the UI over HTTPS (use `mkcert` for local certs).

### MFA prompt loops

If the MFA verification page reloads endlessly:

1. Clear the auth cookie manually in browser DevTools.
2. Log in again.
3. If the issue persists, check that the backend MFA verification endpoint returns the correct session cookie.

---

## Proxy and Network Errors

### 502 Bad Gateway

The admin UI's backend proxy (`/api/proxy/...`) could not reach the backend.

1. Confirm the backend is running and listening on the expected port.
2. Check `NEXT_PUBLIC_API_URL` and any reverse proxy configuration.
3. Inspect the Next.js server logs for the upstream error.

### 504 Gateway Timeout

The backend took too long to respond.

1. Check backend logs for slow queries or blocked threads.
2. Increase the proxy timeout in `next.config.js` if the operation is legitimately slow (e.g. large exports).
3. Consider moving long-running operations to background jobs.

### Backend unreachable after deploy

If the admin UI loads but all API calls fail:

1. Verify DNS resolution from the admin UI container/host.
2. Check firewall rules between the UI and backend.
3. Confirm the backend health endpoint (`/api/v1/config/quickstart`) returns 200.

---

## Missing Environment Variables

### Startup validation failures

The admin UI validates required env vars at build and runtime. If a required variable is missing, the build or server start will fail with a descriptive error.

**Required variables:**

| Variable               | Purpose                              |
|------------------------|--------------------------------------|
| `NEXT_PUBLIC_API_URL`  | Backend API base URL.                |
| `NEXTAUTH_SECRET`      | Session encryption secret.           |
| `NEXTAUTH_URL`         | Canonical URL of the admin UI.       |

**Optional but recommended:**

| Variable                        | Purpose                                  |
|---------------------------------|------------------------------------------|
| `SENTRY_DSN`                    | Error reporting.                         |
| `NEXT_PUBLIC_ENABLE_BILLING`    | Toggle billing UI features.              |
| `NEXT_PUBLIC_ENABLE_COMPLIANCE` | Toggle compliance dashboard.             |

### Verifying env vars

```bash
# Print resolved env vars (redacts secrets)
bun run env:check
```

If the script is not available, manually verify:

```bash
echo $NEXT_PUBLIC_API_URL
echo $NEXTAUTH_URL
```

---

## Test Failures

### Common mock issues

**`vi.fn()` not being called:**

- Ensure the mock is set up in `beforeEach`, not at module scope (mocks reset between tests).
- Verify the import path in `vi.mock()` matches the exact alias used in the component (e.g. `@/lib/api-client`).

**`screen.getByRole` not finding an element:**

- Check that the element is rendered asynchronously. Use `await screen.findByRole(...)` instead.
- Verify the component has the correct `role`, `aria-label`, or text content.

**Stale closure in mocked return values:**

- Use `mockReturnValue` (not `mockReturnValueOnce`) if the mock is called multiple times during a single test.

### Snapshot updates

When UI changes are intentional:

```bash
bunx vitest run --update
```

Review the diff carefully. Only commit snapshot changes that match the intended UI update.

### Test timeouts

If tests hang or time out:

1. Check for missing `await` on async operations.
2. Ensure `cleanup()` is called in `afterEach` to prevent leaked state.
3. Look for unresolved promises in mocks (e.g. `mockResolvedValue` vs. `mockImplementation` with missing `async`).

---

## Build Errors

### Standalone output issues

The admin UI uses `output: 'standalone'` in `next.config.js` for container deployments. Common issues:

**Missing files in standalone output:**

- Files outside `public/` and `.next/` are not included automatically. Copy additional assets in your Dockerfile.

**Module not found in standalone:**

- Standalone mode tree-shakes aggressively. If a dynamic import fails, ensure the module is referenced statically somewhere or add it to `serverExternalPackages` in `next.config.js`.

### Sentry wrapping errors

If the build fails with Sentry-related errors:

1. Verify `SENTRY_DSN` is set (or remove Sentry config files if not using Sentry).
2. Check that `sentry.client.config.ts`, `sentry.server.config.ts`, and `sentry.edge.config.ts` are valid.
3. Ensure `@sentry/nextjs` version matches the Next.js version.

### TypeScript errors after dependency update

```bash
bun run typecheck
```

If types are broken after a dependency update:

1. Delete `node_modules` and `bun.lock`, then reinstall: `bun install`.
2. Check for breaking changes in the dependency changelog.
3. Update type imports if the dependency renamed or moved types.

---

## Getting Help

If none of the above resolves your issue:

1. Check the [admin-ui README](../README.md) for setup instructions.
2. Search existing issues in the repository.
3. Open a new issue with:
   - Steps to reproduce.
   - Relevant error messages and logs.
   - Environment details (OS, Node/Bun version, browser).
