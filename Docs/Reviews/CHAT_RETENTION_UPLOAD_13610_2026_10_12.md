# Reviewed retention upload preparation — TASK-13610

The private producer prepares one server-allocated candidate through existing seams: strict frozen request validation, typed reference/source/lease admission, original AuthNZ quota authority, another source admission, the house upload sink/validator, and final source admission. No quota or upload await spans a chat transaction. Wrong quota scope and nested calls refuse before candidate creation. Hash and representation remain derived from the reviewed reference; staging size/MIME/paths remain immutable.

Prepared retry initially failed because the generic intent writer correctly accepts only reserved candidates. Shared quota reconciliation now treats prepared candidates as read-only recovery: require the original confirmed intent, no release intent, exact frozen account/scope attribution, and a typed active receipt with the original key and byte count. It never allocates missing authority. Both external quota boundaries handle a second original producer completing preparation. If an initial lookup returns None and the candidate has become prepared, one bounded read-only refresh occurs outside the chat transaction before exact revalidation.

Review also identified the classifier's candidate-before-workspace order. Fresh live-attempt/workspace admission now precedes candidate locking in the same transaction. The real guard/candidate control asserts the approved call order; it does not demonstrate physical PostgreSQL locks.

Failures retain original staging/accounting fences. Tests expire the original lease explicitly, then run actual guarded reclamation and real quota release. This does not qualify automatic housekeeping or retention claim/reference publication.

Evidence before final stable-source checks:
- Preliminary seven subprocess failures/seven official AuthNZ PostgreSQL skips, 12 warnings, 317.52s withheld causes; these are not causal proof. Direct missing-producer control: one failed/13 deselected, 12 warnings, 4.42s; the temporary presence diagnostic was removed.
- First real success/replay failed in10.60s; safe exception/source diagnostics failed in31.47s. Shared prepared recovery then passed seven SQLite cases in76.30s.
- Expanded matrix initially passed11 in127.78s but supplied extra freshness in one fixture. Preserving original None reproduced a false refusal: one failed/21 deselected in32.44s. Bounded refresh passed three targeted cases/19 deselected in15.29s.
- Producer11 passed/11 deselected in159.37s; original quota failure/recovery cases five passed/45 deselected in89.05s.
- Connected store/upload batch161 passed,163 skipped,one failed,16 warnings in508.56s. Its existing cross-process final-death control missed the first15-second checkpoint, which combined initialization and publication. Seven physical IO method bodies match HEAD AST. The harness now reports actual child readiness with a separate120-second initialization bound, preserving15-second publication, five-second claim, eight-second drain/death and all quota/byte assertions.
- All four actual SQLite process publication/death controls passed; four official PostgreSQL skips and109 deselected,12 warnings,23.22s.
- The first lock-order fixture selected the wrong branch and passed; excluded as proof. Corrected control failed at real allocation ordering: one failed/23 deselected,12 warnings,48.38s. Final fresh guard passed three targeted controls/21 deselected in29.51s.
- Four-file Ruff zero; all three pre-existing changed-file HEAD baselines zero. Actual two-production-file Bandit zero findings/errors with coverage checked. Compile/diff and AST-preserving scoped Black at the repository's120 columns pass.
- Both final independent source reviews return bounded CLEAR.

ADR required: no. Existing ADR049/050 and proposed070/071 govern lifecycle, original quota authority and source incarnation. No persistence or public protocol change.

No runtime, browser, profile database, send, claim, public endpoint or capability change. The producer's runtime matrix uses SQLite chat/physical storage and real AuthNZ accounting; physical PostgreSQL remains unqualified. Public multipart identity allocation/mapping, active Sync admission, native-source bridge, atomic claim/reference publication and broad NN/G/H2 acceptance remain open. Private logs/proofs are under /private/tmp/chat-retention-upload-13610-*; raw startup data is not published.

Final stable-source verification: producer12 passed/12 deselected,12 warnings122.31s; connected original quota recovery five passed/45 deselected,12 warnings56.27s; complete physical IO58 passed/59 official/backend-specific skips,12 warnings137.19s. These final suites qualify the changed boundaries; the earlier failed connected batch is not called passing. Physical AuthNZ/PostgreSQL parameters remain unqualified; the initial official fixture reported unavailable. Full canonical fingerprint/path checks pass unchanged:2129 paths/3301 schemas, SHA256 bd7997199762d42481de4c4b32536f3c82f279d695c86ff44afa427b36771212. No frontend source or wire changes, so no frontend rebuild/type check was added.

Normal exact-owned eight-file hooks pass without bypass. House Ruff/Black filters skip these native paths; the meaningful manual checks above pass. The completed owned plan is retired to /private/tmp/IMPLEMENTATION_PLAN_chat_retention_upload_13610_completed.md. Freshly fetched dev2a8ffa78a839a53171a27bc1c5f3ce811bb892b3 is included. Only this bounded task is Done; master TASK13458 and H2 parent13261.4 remain In Progress. Exact-file local commit follows.
