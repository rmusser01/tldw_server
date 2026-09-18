# Reviewed capability authentication readiness repair (UAT258)

Public OpenAPI and docs-info discovery remain available before connection setup. Optional protected ingestion-source capabilities wait for a usable configured authority, including manual keys, JWT, runtime override, hosted credentials and an active exact-origin single-user cookie session. Cache identity follows the effective authority.

Original causal failures and both review rounds are retained. The first review found foreign-origin and wrong-mode cookie markers could still dispatch requests; the final code reuses the actual client cookie predicate. Three maintained real-client cases cover protected dispatch counts 0/0/1 while preserving public discovery. Independent final review passed25 tests; author adjacent run passed120.

Production semantic review cleared in round1; round2 closes the maintained-test gap. Compiler comparison retained90 baseline/current diagnostics with zero added or removed. Scoped ESLint reports no errors with existing warnings. Bandit cannot parse TypeScript; its zero findings are not TypeScript security assurance.

Native fresh-unconfigured and authenticated positive acceptance remain pending explicit source upgrade. No full matrix acceptance is claimed. Payloads preserve original bytes, gzip where needed; known credential variants and JWT patterns are scanned. Raw private profiles, credentials and runtime logs are excluded. Review input references outside this packet remain local hash-only references, not a standalone replay guarantee.
