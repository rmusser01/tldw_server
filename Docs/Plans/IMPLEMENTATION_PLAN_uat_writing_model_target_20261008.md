# Writing model target repair — TASK-13260.278.18.83.45

C06 live Custom failed twice before a proposal; the original persisted manuscript remains intact. Writing passes the UI-only `tldw:` namespace and omits provider resolution; native Chat already uses the shared catalog resolver and strips this namespace. Do not claim the generic live error alone establishes its backend cause.

ADR assessment: no new ADR is needed; reuse existing client model catalog and canonical Chat request contracts. No provider policy, credentials, backend, dependencies, or saved content change.

## Stage 1: Reproduce request mismatch
**Goal**: Test the mounted request boundary for continuation and revision.
**Success Criteria**: Wrong client namespace/provider fails; opaque/auto and cancellation controls retain behavior.
**Tests**: UI mounted streaming/nonstream/revision targets; held lookup unmount/Stop.
**Status**: Complete

## Stage 2: Reuse target resolution
**Goal**: Resolve the existing provider and remove only the client namespace at dispatch.
**Success Criteria**: Original text, service-prompt scope and request cancellation remain fenced; affected configured clients pass.
**Tests**: Full changed mounted suite in UI and WebUI, scoped ESLint, configured types with final combined candidate. Bandit N/A for TypeScript.
**Status**: Complete

## Stage 3: Review and real UAT
**Goal**: Independently review, load the combined frontend and complete a bounded actual revision/reopen/undo/export journey.
**Success Criteria**: Real suggestion and selected final artifact match the user decision; no full-family acceptance without every required cell.
**Tests**: Fresh frozen-source/build ownership; actual C06 live request and persistence checks.
**Status**: In Progress

## Verification notes

- Initial eight new mounted controls: causal red 5 failed / 3 passed / 76 excluded, natural exit 1, 5.883 seconds outer. Failure showed raw namespace/missing provider in three paths and revision dispatch before a held lookup could retire.
- First complete affected pair after implementation failed three original admission/lease controls in each client: target resolution preceded the existing immediate service-prompt lookup. Corrected only production ordering; original assertions untouched.
- Corrected complete pair: UI 84/84 passed natural 0 in 86.055 seconds outer; WebUI 84/84 passed natural 0 in 85.500 seconds outer. No skipped cases in either qualified run.
- Independent review found the padded client ID normalization seam and noted the new continuation unmount case must prove its generation lookup starts, rather than merely observing the annotation lookup. One minimal trim-order correction and one padded model control authored; the original unmount assertions retain their stronger actual-await admission check. Qualify only these genuinely new/strengthened controls, retaining the 84-pair scope above.
- Initial ESLint invocation from the WebUI directory ignored shared UI paths outside its base; it is not lint qualification. Correct scope uses the existing apps-level config.
- TypeScript-only scope: Bandit does not apply. No runtime, model, install, saved-content mutation, or full C06 acceptance follows from these source checks.

- Padded model control plus two held-target unmount controls passed in each client (3 selected / 82 excluded): UI 2.868 seconds and WebUI 2.916 seconds outer. Review then identified Vitest equality ignoring undefined property presence; strengthened only the two waits to require Object.hasOwn(target, "explicitProvider"). Final two strengthened controls passed in each client: UI 2/2, natural 0, 3.626 seconds outer; WebUI 2/2, natural 0, 3.006 seconds outer, 83 excluded in each. No full 85-case replay.
- Independent final connected production and all nine new test bodies reviewed CLEAR. Frozen production SHA256 1a6512b2475f83e03751f49cd028a6a07a7dc4066c850160548ff31ca72afe92; tests 2f77203dd71a0c27b35e3964e44975b2af216e90ff320b0409d73367b76b55c0. Correct apps-level configured lint: production zero errors / 27 exact inherited warnings, tests zero errors / warnings. Combined final configured types and live load remain open.

- Root corrected final combined configured semantic types: UI natural0, no diagnostics,66.245 seconds outer (finished2026-10-08T22:47:13.773923Z); sequential WebUI natural0, no diagnostics,42.918 seconds outer (finished22:47:56.986458Z). Both use8GB/600second cap/noEmit/nonincremental. All7293 inventoried apps files match before/after; freeze digest13bf4be7bcfd0bfd39be60868cbadcbe78756b28661c26334cdafb97a99fc18c. This qualifies current Writing/C02/C12/A10/Retry source together after five ignored Testing Library option corrections. Root final source reviews CLEAR; combined production build/live UAT remains open.
