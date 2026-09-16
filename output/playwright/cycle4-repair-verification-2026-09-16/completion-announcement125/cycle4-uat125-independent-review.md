# UAT125 independent review

**Final outcome: clear.** The active-Retry finding below is resolved by the final settled-state render guard. No remaining actionable finding within UAT125.

## Finding: P2 — suppress stale completion during a new active attempt

The proposed `streamingComplete && !errorPayload` condition fixes the observed settled error, but the completion state remains true after a failure. If that same message slot transitions to a new non-error streaming/processing stub within the two-second completion window, the error mask disappears and **Response complete** is rendered during the active request. The effect at `Message.tsx:435–450` does not clear completion on active start; its cleanup also cancels the old timer. The main parent (`PlaygroundChat.tsx:1372`) keys normal rows by block index, so same-slot replacement can retain this component state rather than always resetting by assistant ID.

Private mounted regression (no repository edit): `/private/tmp/cycle4-uat125-retry-probe.config.mts` injects one focused case into the existing integration test through a Vite read transform. Transition: active stream → decoded error (completion hidden) → active non-error stub. `/private/tmp/cycle4-uat125-retry-probe.log`: **1 failure**, because the completion span reappears on the last transition. This is component-level automated evidence, not a new native observation.

Narrow correction: require the message to be settled (`!props.isStreaming && !props.isProcessing`) as well as lacking an error payload, or equivalently reset completion state for the new lifetime without introducing a separate state system. Add both active flags to regression coverage; preserve the successful settled announcements and current assertive error alert.

## Verified current scope

- Production diff is one render predicate in `Message.tsx`; no action, transport or persistence changes.
- The supplied RED log has exactly3 failures/15 passes: streaming failure, processing failure, and later error replacing success.
- Independent focused run: **19 tests/2 suites passed**, `/private/tmp/cycle4-uat125-independent-green.log` (error-recovery integration18 + source guard1).
- Translation fixture now supports the actual `t(key, {defaultValue})` signature; the new assertions inspect the actual mounted polite span/alert, though decoder and child components remain controlled.
- UAT124's resolved assistant-parent persistence work is separate. It can preserve/group failed variants on reload; it does not reset this presentation state or make an error a successful response.

## Limits

No repository edits, runtime/browser/inference, full compiler, lint, or tracker/task changes. The exact native failure was corroborated in `/private/tmp/cycle4-native122-final-image-only-identity.json` (error article plus Response complete); the later snapshot retains the error alert but not the transient completion. Native post-fix acceptance remains with root. Review remains pending the bounded active-state correction above.

## Final correction verification

Root added permanent streaming/processing active-start regressions (RED2 fail/18 pass) and now requires streamingComplete, no decoded error, and neither active flag. The original private Vite probe is unchanged and now passes1/1: /private/tmp/cycle4-uat125-retry-probe-green.log. Independently rerun final affected suites: **21 tests /2 suites PASS**, /private/tmp/cycle4-uat125-independent-final-green.log. Both successful completion controls remain green; error arrival and immediate active replacement suppress completion. Final reviewed source/test hashes are in /private/tmp/cycle4-uat125-independent-manifest.json. No lifecycle refactor or cross-task edit was needed. Native post-fix verification remains with root.
