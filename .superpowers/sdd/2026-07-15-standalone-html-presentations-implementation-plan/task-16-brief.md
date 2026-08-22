## Task 16: Keep The Extension Source-Free And Add WebUI Handoff

**Primary files:**
- Modify: `apps/packages/ui/src/routes/route-registry.tsx`
- Modify: `apps/packages/ui/src/routes/route-metadata.ts`
- Modify: `apps/packages/ui/src/components/Option/PresentationStudio/ExtensionStartPanel.tsx`
- Modify (RED-proven approved scope): `apps/packages/ui/src/components/Option/PresentationStudio/PresentationStudioIndex.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/option-presentation-studio-route-guards.test.tsx`
- Test (approved compatibility scope): `apps/packages/ui/src/routes/__tests__/option-presentation-studio-start.test.tsx`
- Test: `apps/extension/tests/e2e/presentation-studio-start.spec.ts`
- Create: `.superpowers/sdd/2026-07-15-standalone-html-presentations-implementation-plan/task-16-report.md`

Task BASE is `b8197101345fe3d2762e8a50ed2c322d27bc8ca3`.

IMPECCABLE_PREFLIGHT: context=pass product=pass command_reference=pass shape=pass image_gate=skipped:no imagery belongs in a metadata-only handoff mutation=open

- [x] **Step 1: Finish the read-only preflight and lock the transport-aware route shape**

Read `PRODUCT.md`, `DESIGN.md`, the approved Task 16 plan section, and the extension/frontend sections of the standalone HTML design. Inspect at least five real analogues: deferred extension routing and target filtering; route availability metadata; Task15 metadata-first Presentation Studio detail dispatch; Task13 source-free metadata normalization; the sidepanel canonical WebUI-base helper; and built-extension Playwright launch/network interception.

The extension keeps `/presentation-studio` as the metadata index and `/presentation-studio/start` as structured quick-start. It must not register the WebUI-only `/presentation-studio/new` editor. Because the parameter route would otherwise accept literal `new`, reserve that literal and redirect it to `/presentation-studio/start` without making a metadata request or importing an editor. Preserve `/start` precedence.

The extension direct project route is a dedicated resolver. It requests exact-ID source-free metadata first. Only `structured_slides` may then mount the existing structured detail wrapper. `standalone_html` and unknown future kinds render a bounded metadata-only handoff. Route changes, unmount, config changes, auth-principal changes, and Slides scope mismatch retire the prior request/result and fence both adoption and click-time opening.

- [x] **Step 2: Write the complete unit and E2E regression set before production edits**

Use real route definitions, route metadata, and panel behavior while mocking only transport/runtime boundaries. Cover the extension route inventory and ordering; literal `new`; WebUI/extension availability for base, new, start, and detail; exact-ID metadata gating; standalone/unknown handoff; structured compatibility; stale HTML-to-structured and structured-to-HTML transitions; malformed/blank/mismatched IDs; metadata/error/offline/auth/capability states; unmount and authority-event fencing; and callback-time configuration fencing.

Project source-free metadata is still untrusted. Before any response value enters component state or the DOM, project it into a closed object and validate Unicode/scalar bounds linearly: exact nonblank route ID at most 256 scalars; title nonblank at most 512; kind/unknown-kind and each nonblank provenance field at most 256; optional description at most 2,048; counts finite nonnegative integers; and no C0/C1, bidi-format control, NUL, or unpaired surrogate in any accepted string. Reject rather than truncate or retain the raw response. Add oversized, control, bidi, surrogate, invalid-count, and malformed-envelope cases. Authority events synchronously retire the current trusted-ready record so a still-mounted button cannot capture the new epoch and open stale metadata before React rerenders.

Spy on every source-bearing presentation client method, extension storage write, and runtime message. HTML/unknown flows may call only `getPresentationMetadata`; they never call generic detail, version content, save, download, export, render, structured store initialization, or source-bearing extension IPC/storage. A stale index request after config/principal scope change must not publish old metadata. The canonical RED proved the current index lacked this fence, and the controller approved the narrow `PresentationStudioIndex.tsx` production scope for identity-fenced invalidation/reload.

The WebUI target is computed only on user activation. Read `tldwClient.getConfig()` at click time, extract only runtime `serverUrl`, `webUiUrl`, and `webuiUrl`, and require at least one configured candidate that independently parses as HTTP(S). Then resolve those fields through `resolveSidepanelChatWebUiBaseUrl` and append relative `presentation-studio/${encodeURIComponent(trustedRouteId)}`. Preserve an explicit configured WebUI subpath and the helper's documented API `:8000` to WebUI `:8080` inference; strip credentials, query, and fragment through the canonical helper. All-missing/all-invalid configuration fails closed instead of accepting the helper's default. Metadata/title/provenance/project payload never contributes to the target. Open only `_blank` with `noopener,noreferrer`. Retire the structured-create response and handoff if an authority boundary or unmount occurs before opening.

Extend the real built-extension E2E without removing seeded structured quick-start. Add a standalone metadata direct-link handoff, fixed configured WebUI subpath, exact target/open attributes, and network tripwires for generic detail, versions/version content, save, draft attachment, export, render, and other source-bearing endpoints. Instrument extension storage/runtime messages for source-bearing payloads without rejecting unrelated infrastructure traffic.

- [x] **Step 3: Run the exact canonical commands and capture genuine RED**

```bash
cd apps/tldw-frontend
bun run test:run -- \
  ../packages/ui/src/routes/__tests__/option-presentation-studio-route-guards.test.tsx \
  --maxWorkers=1 --no-file-parallelism
cd ../extension
bun run compile
```

The unit suite must collect and fail on missing Task16 behavior before production edits. Compile may additionally fail on test-first missing exports/types; record its exact independent disposition. Do not weaken the behavioral tests to manufacture compilation success. Stop after three attempts at the same root cause.

- [x] **Step 4: Implement the smallest source-free resolver and route policy**

Add one extension-only project resolver beside `ExtensionStartPanel`. It owns only bounded source-free metadata and request identity. Reuse `tldwClient.getPresentationMetadata`, the existing structured route wrapper, canonical WebUI-base helper, route/error/layout primitives, and existing online/capability state. Do not introduce a client, cache, storage record, request-core/background-proxy change, generic HTML workspace import, source model, or source-bearing type.

Register the WebUI `/new` route only outside extension runtime. In extension runtime, register the reserved `/new` redirect, retain `/start`, and map `/:projectId` to the metadata resolver. Add exact route metadata records for root, new, start, and detail with their real WebUI/extension availability; do not broaden global exact route lookup.

The handoff UI has one clear `h1`, bounded title/kind/provenance text, accessible loading/error/retry states, keyboard/focus-visible controls, shared product tokens/primitives, and no em dash, gradients, glass, card grids, or decorative motion. Unknown kinds receive the same metadata-only handoff with a read-only explanation.

- [x] **Step 5: Run focused, compile, browser, broader, and static gates**

Run the exact unit command and `apps/extension bun run compile`, followed by:

```bash
cd apps/extension
bunx playwright test tests/e2e/presentation-studio-start.spec.ts --reporter=line
```

Run directly associated route-metadata/governance, Presentation Studio index/page/client, auth/config lifecycle, and canonical WebUI-helper tests if present. Characterize explicit alias/subpath priority, server-port inference, credentials/query/hash stripping, and all-invalid failure. Run diff hygiene and static searches across the Task16 diff for source-bearing client calls, HTML/source payload fields, storage/message/cache/logging, popup/navigation construction, render/execution sinks, and source-derived URLs. Every navigation hit must be the fixed application-owned handoff; every storage/message hit must be a negative test or unrelated structured quick-start infrastructure.

- [x] **Step 6: Report, stage explicitly, audit, and commit**

Write `task-16-report.md` with analogue findings, canonical RED, implementation decisions, exact GREEN/compile/E2E/broader evidence, static source-sink audit, and product/accessibility review. Stage only the seven approved implementation/test paths, this brief, the global constraints, and the report. Audit staged paths, preserve the protected `antd` symlink and two Watchlist templates, and commit exactly:

```text
feat(extension): hand off HTML presentations safely (TASK-12115)
```

Do not install, call or edit Backlog, or push.
