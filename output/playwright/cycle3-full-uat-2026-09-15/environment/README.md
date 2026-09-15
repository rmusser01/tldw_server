# Cycle3 UAT environment cache repair

The application behavior under test remains d40e17dc81. This is a configuration-only transition during the multi-user run, not a complete fresh acceptance pass or a product repair.

Repeated disk exhaustion grew the isolated frontend dist to3.4GiB, including2.6GiB of development filesystem cache. Both browser runners paused with no inference/save pending. Parent verified frontend43042/43043, stopped them, confirmed port18281 closed and removed only .next-live-tier-cycle3-multi-20260915. API42500, databases, model configuration, credentials and both persistent browser profiles were preserved.

The existing isolated UAT directory selector now disables only Next16.1.4's supported development filesystem cache. Normal development preserves its default. Independent review found no issue; syntax and scoped ESLint checks passed. Actual config comparison below verifies every other normalized config value plus executed headers, redirects, rewrites and the webpack function are equal. This is a config contract check, not application UAT. No Python was touched; Bandit is inapplicable.

Frontend48482/session52335 restarted through the same launcher. Login18281 returned200. After resumed route loading, dev/cache was8KB and about2.9GiB remained free. Application findings and outstanding workflows remain in the running tracker.

- [Config comparison script](next-cache-config-check.cjs)
- [Actual comparison result](next-cache-config-result.json)
- [Configuration diff](next-cache-config.diff)
- [Running tracker](../../../../Docs/Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md)

The first comparison attempt omitted the advanced-mode browser API URL and failed validation before testing; the corrected isolated environment passed. One earlier stop command failed before execution because the full disk could not create a heredoc temporary file; a verified direct Node command succeeded. These harness failures are not omitted or classified as product findings.
