# In-flight owner boundary RED

A real dependency-wired HTTP handler launches an asyncio.to_thread database read. The probe pauses inside PostgreSQLBackend.execute after the request checkout is captured, then finishes or cancels the request. A separate deterministic controller releases the worker after finalization is attempted. Actual pool return is observed, not mocked. Both cases fail because return occurs before the real query finishes: expected [False], actual [True].

Official required-PG run `uat181-http-inflight-red`: 2 failures, 35 deselected, 0 skips, 12.05s. No production repair has been applied for this newly proved gap. Exact pre-repair source is under inflight-red-source. Original native-lifetime RED remains under red-source and is unchanged.

Proposed correction: close owner immediately, reject new uses, defer only captured owned checkout return until active wrapper/DB command or transaction use finishes. Never wait on worker completion on the ASGI loop, auto-commit, or settle borrowed connections. Full DB execute_query/execute_many scope must include read-only transaction exit and commit=True semantics; direct wrapper and explicit transaction entry/exit also need guards. Raw private-driver access remains an explicit caller-owned escape, outside the safe wrapper guarantee.
