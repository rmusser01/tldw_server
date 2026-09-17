# Targeted original-scenario PostgreSQL acceptance

Run repairs231-250-targeted-20260917; source86458ab88ce3fa62e6518c9d813c3860254ddb2c. Full matrix remains held. Existing dependency installations are copied/reused; this is fresh application configuration, databases and browser state, not a clean OS/dependency install.

New official fixture holders: PGsingle PID18859/session68291, PGmulti PID18878/session2712; original holders29823/96865 remain untouched. Restricted roles enforce RLS; no superuser, BYPASSRLS, inherited grants or memberships. New targets18702/18782 and18703/18783. Normal initialization started; no browser acceptance yet.

Normal initialization exits0 both profiles; administrator bootstrap exits0. Running single backend19326/session93977, frontend19332/session75746; multi backend19432/session84837, frontend19442/session79589. Final release evidence independently reviewed clear (final-release-review/REVIEW.md SHAeaf305faede197127cff6beafbe2d2e38783b5e9129595128981b5560c525e4c).

Observer preparation gap: initial installation failed with ReferenceError URL is not defined in the Playwright CLI execution sandbox, before CDP creation or any TestBot request. Original helper/failure retained. Runtime-compatible bounded origin/path parsing is separately reviewed before use; no transport or product behavior changes.

Multi provider restart: old19432 completed application teardown at20:08:32.544UTC (2733ms teardown); launcher result SIGTERM1 is interrupted-process convention. Replacement25856/session79564 healthy200 at20:09:38.999. Original single remains19326/19332; multi frontend19442 unchanged. All UI/accounts intact.
