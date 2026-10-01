# PR2979 native post-atexit observation — UAT569

Task: TASK-13260.278.18.83.58. Separate diagnostic branch; immutable tested source b38c9f725748c5addd5ed44f7526bea55ee7621a. This is observation, not corrective work or final PR acceptance.

## Stage 1: Record the missing native phase
**Goal**: Preserve UAT568 evidence and reuse the clean isolated diagnostic worktree.
**Success Criteria**: Official duplicate search and task precede tracked edits; prior d86 branch and recovery remain available.
**Tests**: Read-only source/ref/evidence verification.
**Status**: Complete

UAT568 run36847712224/job110321743417 exited naturally0 with1190cases/1095passes/95originalskips. Its first unconfigure took113.929286s with actual pytest GC stacks. No native samples cover1086.160934s after its final early-atexit marker. Those late seconds and retained graph/type/referrer ownership remain unexplained. Immutable evidence: /private/tmp/pr2979-uat568-native-observation-1052.json (SHA0eabb283afc10cd82f65fba771fbdbcda6d60b1aa240cdd6c29058dd7dc3d543).

## Stage 2: Extend only parent-owned sampling
**Goal**: Reuse the existing filtered native sampler after the owned child's early-atexit marker.
**Success Criteria**: Remove all timed Python signal requests, including marker-read races; retain child phase snapshots; schedule native-only samples at15/30/60/90/300/600/900s after that parent-observed marker. Distinct artifact names preserve original early samples. Missing late markers remain explicitly unavailable.
**Tests**: Real synthetic child with a23s interpreter-finalization hold and default SIGUSR1 disposition must exit naturally; late frames must occur after its actual atexit marker, without late Python signals or private values. Original six synthetic contracts remain.
**Status**: Complete

## Stage 3: Qualify and independently review
**Goal**: Verify the minimal observer delta before publication.
**Success Criteria**: Causal red/green, original synthetic controls, compile/Ruff/diff and Bandit dispositions; immutable patch/evidence review clear. All original workflow/action/config/source bytes, cleanup/GC/exit/warnings and300s case/60m job limits unchanged. No whole local Prompt or PostgreSQL run.
**Tests**: Existing synthetic module plus the new late-phase behavior, static checks and independent source/artifact review.
**Status**: Complete

## Stage 4: Obtain actual native late-phase evidence
**Goal**: Publish only the separate diagnostic branch and dispatch the frozen-source observer.
**Success Criteria**: Actual branch/head/run/job verified; eventual source/version/case/skip parity plus late native frames or explicit absence, natural exit or original maximum retained. No diagnostic result substitutes final uninstrumented PR gates.
**Tests**: Bounded direct metadata and terminal logs/artifacts only.
**Status**: Not Started

Retained-object graph/type/referrer ownership still requires actual evidence before any corrective edit. This extension first identifies the missing native phase; it does not add intrusive Python graph traversal or infer the cause from previous early GC samples. No new ADR.

Causal synthetic red against d86: one failure16.21s, child-30/parent158 after parent sent SIGUSR1 at15s during the real post-atexit destructor hold. Native-only sampling removes this diagnostic hazard; it does not attribute hosted delay. Original early checkpoint fields are renamed to sessionfinish_native_checkpoints to reflect the actual observer behavior.

Local qualification:7synthetic passes/zero skips43.40s/natural0. Real early and late native samples exit0 with filtered frames and natural child0; late child uses default SIGUSR1, no timed signals sent. Parent late interval22.800051s is independently labeled as observed. Compile/Ruff/diff0; Actionlint0 unchanged workflow with shellcheck disabled. Bandit1/35LOW(27B101/2B404/6B603),0errors/HIGH/MEDIUM. B101 are executable synthetic assertions; subprocess calls use fixed shell-free argv with trusted interpreter/helper or sole-owned integerPID. Original first green attempt5pass/2fail is retained: native profiler unavailable in default sandbox, and invalid23s parent-observed timing expectation corrected to scheduled15s bound. Independent review pending.

Independent immutable review CLEAR: /private/tmp/pr2979-uat569-independent-review.json SHA9205e994eb1dd65a4c098c4c352f28cadbdf123f90161c78c7d925e61c612e23. All65 evidence artifacts/three sources/patch verified;13 own narrow stdlib ownership/scheduling/status/sampler/privacy checks pass. No own whole pytest/app/PG run. Local AC1/2 checked; native observational AC3 and DoD1 remain open.
