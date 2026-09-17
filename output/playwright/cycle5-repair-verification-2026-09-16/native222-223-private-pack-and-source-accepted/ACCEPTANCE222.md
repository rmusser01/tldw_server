# UAT222 independent native acceptance — PASS

TASK13260.160's original same-pack authenticated native criterion is satisfied. Combined with the separately clear source review (independent63 required-PG/SQLite tests, zero skips; Ruff/Bandit0; exact frozen hashes), the bounded task criteria support closure. Root owns task status and retention. This is not a full application isolation audit.

## Evidence

Six exact inputs are bound by input-manifest222.json; receipt222.json records the independently checked assertions. The original failed15-request result remains retained: Bob3 received pack2 with Alice2 private evidence at08:44:04, while his job5 request correctly returned404. The previous bounded continuation excluded that known failure and is not reinterpreted as passing222.

The new20-request receipt runs09:23:01.745–09:23:02.298UTC. Fresh ordinary API login/auth-me responses establish Alice id2 and Bob id3; both logouts returned200. Authentication tokens stay in script memory and login response bodies are not retained. Identity is taken from auth/me: the script's top-level aliceId actually comes from its pack variable, so it is not used as identity evidence.

- Alice: job5 completed200; original pack2/deck10 owner2/version1; original three generated cards remain owner2/version2/repetitions0. The job, pack and complete three-card list match the original failed receipt's Alice bodies exactly. A second job read at the end equals the first.
- Every generated card's assistant GET200 returns its original citation and primary citation, pack2 and context. Citation IDs remain3,2,1 for cards9ac6501e…,04843428…,a33c73dc… respectively. Those fields and the existing empty threads4,5,6 match the prior receipt exactly.
- Bob: job5→404 at09:23:02.154; pack2→404 at09:23:02.172; sampled original generated-card assistant→404; deck10 card list200 with zero items/total. This is the original foreign-pack scenario on the original preserved objects.
- Source note b83dca90-fab0-4c6f-8c0f-6f1e93dfffc8 is owner2/version1. Original Citrine card37b10bd7-edf4-4f35-83c1-1490115d8c55 remains owner2/version2/repetitions1. No generation, rating, source edit or destructive operation appears in this receipt.

## Runtime/source attribution

Before-restart09:21:37.733 and after-health09:22:36.272 source manifests match all3668 entries, revision598d377df25fd78e20b895c3cc5c303d19223b7c. The ChaCha hash is63b3ccd15b9ce19bd1d964821eb0d33b4f191ff1a329cd83356e8481a6423d97, exactly the reviewed222 source. The safe runtime receipt records newPID89545 started09:22:20.617, alive with health200 at09:22:36.185 while old82583 was still draining. It does not claim the old process was already gone.

## Scope and limits

Native role remains the qualified privileged/BYPASSRLS configuration; observed protection is application owner enforcement. Restricted-role application reads and foreign mutations/rollback/version/SQLite device semantics are covered by isolated automated tests, not native foreign writes. No raw-SQL/RLS or broad domain claim is made. Native calls do not exercise regeneration or malformed historical child rows.

The non-auth requests are GETs, but assistant GET may create an empty thread by design; no blanket zero-storage-side-effects claim is made. In this readback the three retained thread IDs/timestamps already match the original receipt. This auditor performed no credential/private-session reads, live API or runtime actions, or data mutations. UAT223 source-navigation acceptance remains separate until its actual click and resulting UI/network receipts are audited.
