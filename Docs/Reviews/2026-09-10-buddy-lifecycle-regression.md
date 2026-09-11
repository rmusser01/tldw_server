# Buddy route lifecycle regression — 2026-09-10

TASK-13211 remains an open investigation. This check covers its bounded-loading
criterion; it does not establish the historical initiating trigger.

## Executable evidence

The shared-UI test `sidepanel-persona.buddy-lifecycle.test.tsx` mounts the real
Persona route, Buddy context, Buddy host and live-control hook together. HTTP
responses and online/connection/capability hook outputs are controlled; selected
voice-controller output fields are overridden. The host uses `root="sidepanel"`
without WebLayout, ServerReadinessGate or Fast Refresh. It verifies that all
five voice states and the tool signal actually reach the published route context.

Across 24 state transitions at controlled 250 ms intervals, plus transcript updates and
runtime diagnostic changes, requests remain at one pack list, one pack detail and
one session list. The host remains present and its pack remains loaded. The
positive control explicitly takes the route offline and back: the host disappears,
then exactly one new request set loads. This distinguishes ordinary state updates
from an actual availability-driven remount.

```sh
cd apps/packages/ui
bun run test src/routes/__tests__/sidepanel-persona.buddy-lifecycle.test.tsx --maxWorkers=1
```

The source is dev `50c1f68957`; the relevant frontend files are unchanged in the
credit-repair merge `f45bdca7b0`. Execution used a disposable frozen-lock frontend
installation. No physical microphone, provider, installed extension or native
terminal acceptance is claimed. DOM host/pack assertions do not prove rendered
image pixels or real audio playback.

The initial focused run passed one test in 8.20 seconds; the corrected historical
replay passed in 7.98 seconds, and all four current source files were restored
byte-for-byte. Qodo review then replaced the repeated-update loop's real sleeps
with a fixed Vitest clock and explicit timer advancement. The check asserts exactly
six seconds of simulated updates and restores real timers before reconnect checks
and in failure cleanup. The revised focused test passed in 0.97 seconds.
Review corrected the WebSocket fixture to use the real URL/protocol envelope and
ready-state transitions, plus cleanup on failure. ESLint reported no code findings;
its shared-package invocation emitted the existing Next pages-directory diagnostic.
Node emitted its localStorage experimental warning. No Python production code
changed, so Bandit is not applicable to this test/documentation change.

## Historical source recovered

The original frontend commit `73640bbb89aed7d878d254bd622ca68f79923ad8` was found
in a separate local checkout after the first clone and GitHub lookup failed.
Replaying its route, Buddy host/context and live-control hook in the same test
harness also held all three request counts at one across the 24 updates. That is
a controlled four-file replay, not a complete historical application build.

Context, live-control, WebLayout, readiness, connection, media-query, capability
and Next configuration files are identical between the two revisions. The route
now pins connected-session identity and the WebUI passes its explicit shell;
neither difference supplies a recurring 250 ms trigger. Connected health polling
preserves its interactive state while checking; normal poll/retry intervals are
30 seconds, 5 seconds and 2 seconds.

Development Fast Refresh can rerun both effects while ignoring unchanged dependency
arrays. It remains a possible mechanism, not a proven cause. The earlier UAT log
does not record refresh/effect lifetime events at the failing timestamps. A future
reproduction must capture refresh events, effect setup/cleanup, route availability
and Persona identity together with request timing. No speculative retry, debounce
or lifecycle change is included.

## Related completion

PR #2940 merged the separate artwork-credit repair after all CI checks passed and
Qodo resolved its six findings. TASK-13242 is complete. TASK-13211's initiating
trigger and real-browser failure reproduction remain open; native/physical
qualification continues in the existing dedicated tasks.
