# Latest-dev chat performance integration — TASK-13570

The user requires all chat repairs and enhancements against latest dev. Refreshed origin/dev is4d6a85c4c693c4c83480d0f7c8046cfda0634c55. Integrate it into the existing isolated remediation branch after the bounded actor/default acknowledgement commit3b27865375. Leave the dirty primary checkout and runtime artifacts untouched.

ADR assessment: no new ADR. Preserve ADR049 ownership/publication, the approved private D7 settings and edit contracts, and ADR059 task tracking. This integration adds no authority, public protocol, queue, background replay or dependency.

Independent pre-merge source review finds six concrete overlap hazards: cached row callbacks discard edit/delete acknowledgements and editor predicates; latest-state routing can adopt a replacement instead of the rendered action source; raw Stop bypasses the focus handoff; research cache/comparator omit builder/policy/callback identity; deferred legacy text flushes omit available flush-time guards; grouped selectors omit draft-only subscriptions used by rendered settings leases. These are incoming integration hazards, not current-branch or executed-test claims.

## Stage 1: Integrate and inspect latest dev
**Goal**: Bring the accepted dev changes into the isolated branch and inspect every overlap.
**Success Criteria**: Resolve conflicts while retaining source ownership, acknowledged actions and exact model preference contracts. Preserve independent safe performance changes; do not commit an unverified merge.
**Tests**: Diff and conflict inventory against both parents; current source/target and model setter call-site traces. Run causal checks before correcting connected regressions.
**Status**: In Progress

## Stage 2: Preserve rendered row actions
**Goal**: Keep full and compact message actions bound to their original owner/editor/target, with truthful completion and focus.
**Success Criteria**: Edit forwards both editor predicates and returns its Promise; Delete returns its result. Retained confirmations and regeneration cannot adopt another source. Explicit Stop retains its UI handoff. Research policy/builder/callback changes reach actual rows. Use existing render-bound closures or existing leases rather than a new action registry.
**Tests**: Actual full/compact row retained-confirmation and edit-ACK controls; Stop event/focus; same-metadata research policy and parent-fence changes; existing shared Message and H1 action suites.
**Status**: Not Started

## Stage 3: Preserve streaming and source subscriptions
**Goal**: Make deferred legacy text publication check the initiating source and keep fresh current actions reactive to draft-only transitions.
**Success Criteria**: Existing scheduler shouldFlush checks captured source/account/controller boundaries. Terminal flush and error cancellation preserve final plain text and variants. Native send remains on its existing sender. Draft address/revision changes render fresh callbacks; selected-model exact ACK/retry remains unchanged.
**Tests**: Token then held iterator with source/account/Stop retirement before timer flush; positive terminal text/persistence controls; address-only replacement with unchanged IDs/rows; existing normal/native/message/model adapter suites and scheduler unit controls.
**Status**: Not Started

## Stage 4: Verify and record the integration
**Goal**: Commit only the reviewed verified merge and record exact acceptance limits.
**Success Criteria**: Latest dev is an ancestor; affected gates, both client types, authored static/format/security checks, canonical tasks and two independent source reviews are recorded. Distinguish inherited failures and fixture preparation from valid causal evidence. Continue qualified actor namespaces/read receipts/projection/first-bind and the remaining master work.
**Tests**: Scoped actual integration and imported performance controls; typecheck/extension compile; lint delta against both appropriate parents, formatter runtime-equivalence where used, diff and normal hooks. Bandit for authored Python; N/A for TS-only changes. No current preview/native/packaged/physical/AT/participant claim from source tests.
**Status**: Not Started
