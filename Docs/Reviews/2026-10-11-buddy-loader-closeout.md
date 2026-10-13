# Buddy loader investigation closeout

TASK-13211 closes under the requester-approved criterion revision. The former criterion asked for the September 6 initiating trigger to be identified. The requester approved closing against the verified repair and accepted UAT with historical attribution explicitly bounded.

The original incident receipt records approximately 250 ms pack-list/detail/session request families followed by HTTP 429, but no browser initiator, refresh or effect-lifetime metadata. Its cause remains Not established. Historical frontend `73640bbb89aed7d878d254bd622ca68f79923ad8` contains no 250 ms retry loop, and the recorded 24-update historical replay remained bounded. These observations cannot reconstruct the original application event.

A separate controlled Fast Refresh experiment established paired reads without an identity change. PR #3091 repaired that reproducible effect-replay mechanism, including pending explicit session reloads, while retaining existing scope and stale-response guards. The actual merge is `005802bdb070fd68e087c4db3f831c33bef07c39`. Earlier browser artwork and human voice acceptance remain source-attributed and accepted. The former remount-only conclusion is explicitly superseded in artwork-investigation.md.

On server dev `39b53a876f2cfefcede02c92ae7dacf88f7a98ec`, the three focused loader/host/route files passed 103/103 cases, with zero failures or pending cases. Existing installed dependencies were used; Node emitted its experimental localStorage warning. The first launcher invocation used the repository root and ran no tests; the package-owned invocation completed. The September 30/October 3 private receipt directories are no longer present, so their recorded historical results are not claimed freshly inspected. The original checked-in incident receipt is unchanged.

No product source changed, accepted UAT repeated, or provider/audio call ran. Bandit does not apply to this Markdown-only closeout. Existing ADR-005 and ADR-046 apply; no new ADR. Full application-configured human OpenAI realtime and the original September 6 trigger are not certified by this task closeout.

## Current verification evidence

The [retained verification receipt](assets/buddy-closeout-2026-10-11/loader-verification.json) records the exact three test paths, command, source hashes, result/log hashes, warning and 103 passing cases. The command ran from `apps/packages/ui`; raw result and log remain `/private/tmp/buddy-loader-closeout-20261011.{json,log}`. No pending or skipped case is counted as a pass.
