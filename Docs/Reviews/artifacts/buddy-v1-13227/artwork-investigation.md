# Buddy v1 legacy-load loop investigation

## Scope and state

- Read-only source/test investigation at `1fc19c7c8384f38eca2becdf53fb8bbcd205be7a` on `codex/buddy-v1-live-qualification`.
- No production, test, task, or repository-document edits were made. The untracked live-build directory and root-owned plan/task files were left untouched.
- Historical symptom recorded in TASK-13211 on 2026-09-06: roughly 250 ms repetitions of the legacy Persona visual pack list/detail requests together with the Persona live-session list, eventually receiving 429, then disappearing after rebase/HMR reload and reconnect.
- Fresh 2026-09-08 WebUI observation at loopback ports 18180/18181: unaffiliated `/api/v1/buddies` plus attachment reads occurred once per five-second independent-host poll, with no attachment, no 429, and no reproduced legacy loop at the reporting boundary.

## Source-backed conclusions

The paired requests identify a narrower trigger than an ordinary re-render.

1. `BuddyShellHostInner` starts both request families when it mounts with an active Persona. Its visual effect calls `listPersonaVisualPacks(activePersonaId)` and, when the active summary lacks `assets_by_id`, `getPersonaVisualPack(...)`. That effect depends only on the normalized active Persona selection and `visualPackRefreshNonce`. The same inner component calls `usePersonaLiveControl`, whose auto-load effect invokes `listPersonaLiveSessions(...)` and depends on `autoLoad` plus a `reload` callback memoized from normalized Persona ID and surface strings.
2. Neither service function nor either effect contains a retry/poll loop. The host's one-second timer only clears expired visual overrides. A visual-pack activation event can repeat pack loading, but cannot repeat the live-session list. Voice/tool context changes are semantic context updates and do not change the live-control reload callback's primitive dependencies.
3. Therefore repeated *paired* pack list/detail and live-session list requests require one of these events:
   - `BuddyShellHostInner` is repeatedly unmounted/remounted; or
   - the active Persona ID / `surface_id` toggles between values in a way that restarts both effects.
   Ordinary object recreation or animation ticks cannot produce this pair.

`IndependentBuddyHost` is **not a candidate for the historical root cause**. TASK-13211 recorded the incident on 2026-09-06; Git history adds the independent host in `77e2f3765b` on 2026-09-08. Its arbitration is only a current topology risk worth distinguishing from the old loop:

- WebUI mounts `IndependentBuddyHost` in `_app.tsx` outside `ServerReadinessGate`; `WebLayout` separately mounts the legacy `BuddyShellHost` under `BuddyShellRenderContextProvider`.
- Legacy `BuddyShellHost` returns `null` whenever global `useBuddyManagementStore(...).attached` is true.
- `IndependentBuddyHost` returns `null` while canonical connection config is loading and keys `IndependentBuddySession` by server/auth identity. `useCanonicalConnectionConfig` sets `loading=true` whenever its storage-derived fallback or refresh-rotation key changes and asynchronously re-reads `tldwClient.getConfig()`.
- Every `IndependentBuddySession` unmount unconditionally calls `setAttached(false)`. Its next successful attachment fetch calls `setAttached(true)` (or true for an authoritative unavailable attachment). A transient config-driven remount can consequently expose the legacy host long enough for one paired legacy load, then suppress it again.
- The independent host polls its own Buddy endpoints every 5000 ms. The fresh live trace matches that intended cadence and currently has no attachment, so `attached` remains false and there is no five-second suppression/re-admission cycle. There is no source-level 250 ms loop here. A future loop caused by current arbitration would require separate live evidence of repeated canonical-config changes/remounts and attachment-state transitions.

For the historical implementation, the remaining source-compatible candidates are legacy-host remount/layout churn or active Persona/surface toggling. The route render-context publisher can participate in the latter, but still needs live evidence:

- `sidepanel-persona.tsx` publishes the `persona-garden` context from an effect with several route/session/voice dependencies. Every dependency change runs cleanup `setBuddyShellRenderContext(null)` and then publishes the next context.
- `CharacterSelect.tsx` is another publisher and likewise clears the shared context during cleanup.
- The provider has a semantic equality guard for non-null contexts, so equal context objects retain identity. That guard does not arbitrate ownership of competing publishers and cannot ignore a cleanup `null`.
- React commonly batches cleanup/setup state writes for a dependency update, so the source alone does not prove an intermediate null is committed or that the host remounts. Competing publishers or a committed inactive/null transition would need to be observed live before treating this as the cause.

The exact `250` ms constant found in WebUI is `PREFETCH_STEP_DELAY_MS`, used to warm routes sequentially after idle. Route prefetch should not mount those pages, and there is no code path connecting it to the Buddy load effects. It is only a timestamp correlation candidate. `ServerReadinessGate` retries at 2000 ms and the independent host polls at 5000 ms.

## Existing-test evidence and gap

Focused run:

```text
bunx vitest run \
  src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx \
  src/hooks/__tests__/usePersonaLiveControl.test.tsx \
  src/components/Common/PersonaBuddy/__tests__/IndependentBuddyHost.test.tsx \
  --maxWorkers=1 --no-file-parallelism

Test Files  3 passed (3)
Tests       81 passed (81)
Exit        0
```

The run emitted only existing i18next initialization warnings.

Those suites do not cover the observed topology:

- `BuddyShellHost.test.tsx` mocks `usePersonaLiveControl`, so it cannot count the real session-list request alongside the real visual requests.
- `usePersonaLiveControl.test.tsx` tests the hook in isolation, including stale work, unmount, Strict Mode, and concurrency.
- `IndependentBuddyHost.test.tsx` mocks canonical config and does not co-mount the legacy shell/provider. It covers identity changes, polling, and unavailable attachments, but cannot detect an `attached=false` window admitting the legacy host.
- WebUI layout tests mock the Buddy hosts. Route tests probe context but do not mount the real host and services together.

## Minimal deterministic regression, conditional on live evidence

For TASK-13211, first add instrumentation or a test probe that counts the legacy inner host lifetime and records its normalized `{personaId, surface, autoLoad}` tuple. Reproduce the live trigger before encoding a repair. The minimal regression should then mount the real legacy host and real `usePersonaLiveControl` under the actual publisher/lifetime boundary implicated by the evidence, while mocking only network/render edges. Assert one initial pack-list/detail plus session-list request pair and no additional pair across the precise live-state transition.

If the probe shows the inner host remounting while route identity stays constant, the regression should drive the identified parent readiness/layout transition. If it shows Persona or surface toggling without a host remount, drive that exact publisher transition. If it shows competing publishers, mount both and assert one publisher's cleanup cannot clear the other's active context. A generic rerender test would not reproduce either causal contract.

The following separate regression applies only if a *new* current-build failure is correlated with independent attachment traffic; it is not evidence or a proposed fix for TASK-13211:

If the live request trace shows an independent attachment request immediately before/after each legacy request burst, add one integration test that co-mounts:

- `BuddyShellRenderContextProvider` with an active Persona context;
- the real `BuddyShellHost` (mocking only network/render boundaries);
- `IndependentBuddyHost` with controllable canonical config and deferred attachment requests.

Drive the exact observed config transition. Establish an authoritative independent attachment, trigger the transient loading/rekey boundary, and assert all of the following:

1. the independent attachment remains authoritative through the transient refresh;
2. the legacy visual pack list/detail and live-session list do not restart;
3. private independent state is still cleared when the actual server/auth identity changes.

This preserves the existing credential-boundary contract while testing the current arbitration mechanism. A fix should not be selected until the live trace establishes whether the transition is benign config refresh or a real identity change.

## Live evidence requested

For one reproduced historical-style burst, capture the ordered URL sequence plus a temporary count/log of legacy inner-host mount/cleanup and normalized Persona/surface inputs. Correlate each pair with:

- a route/context transition or visible legacy dock flicker;
- a full layout/remount/readiness transition.

For the current build, record independent `/api/v1/buddies` and attachment endpoints separately so their expected five-second poll is not mistaken for the historical 250 ms legacy loop. Also compare burst count/timestamps with route-prefetch entries. If there is no host lifetime or Persona/surface transition, the current source does not explain a sustained paired loop and browser initiator evidence is required before proposing any production fix.
