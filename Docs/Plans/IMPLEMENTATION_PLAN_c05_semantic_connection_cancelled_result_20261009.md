# C05 stable connection and cancelled-turn feedback

Task: TASK-13260.278.18.83.45. Actual native cancellation cause remains unknown.

ADR assessment: no new ADR; retain existing ACP ownership/authentication/permission and actual downstream settlement contracts. Equal config rehydration must preserve the existing connection; changed authority must retire it. Native cancellation must remain visibly distinct from successful completion.

Scope: ACP connection hook, shared ACP event projection, canonical history hydration, the actual parent submit handler in both layouts and their existing normal mounted tests. No backend, runner, provider, permission, timeout or shared authentication change.

## Stage 1: Reproduce source defects
**Goal**: Mounted canonical-config clone closes an active socket on unchanged code; native cancelled result is absent in live/restored UI.
**Success Criteria**: Causal failures reach behavior; raw result and original successful controls retained.
**Tests**: Equal-config running prompt; changed authority; live/restored cancellation; duplicate Stop.
**Status**: Complete

## Stage 2: Minimal correction and qualification
**Goal**: Keep sockets for equal semantic inputs, retire changed authority and display cancelled terminal once.
**Success Criteria**: Configured Node20 UI/Web affected suites/types/lint pass; complete independent source review.
**Tests**: New causal controls plus existing ACP hook/projection/store/history consumers. TS-only Bandit not applicable.
**Status**: Complete

## Stage 3: Actual browser recovery
**Goal**: New reviewed build validates a declared native successor/permission workflow with actual effects.
**Success Criteria**: No automatic replay; original cancelled attempt remains unqualified; permissions and file effects independently checked.
**Tests**: One separately declared recovery under existing caps after new build.
**Status**: Not Started

Qualification: first causal source red3failed9passed24filtered; first correction UI57/Web57 natural0. Complete peer review identified cancelled retained-tool delimiter loss (new causal1failed) and completion-before-Stop ACK duplication (new causal1failed plus new prompt-start visibility control failed). Corrected view-only boundaries preserve every retained raw payload; actual accepted parent send creates a non-rendering turn start in both layouts. No native retry performed; actual prior cancellation cause remains unknown. Initial combined UI/Web types natural0 before bounded peer corrections; final corrected qualification pending. Initial ACP lint0errors/2same inherited store warnings; TS-only Bandit not applicable.

Final corrected affected consumer qualification: UI46/Web46 in sixfiles natural0/no skips/outer5.406s and5.184s. Unchanged hook15 retained from priorUI57/Web57; final61distinct cases are split15+46, never onecombined61 claim. Complete independent corrected source/newtest/caller review CLEAR at frozen projection76ee45f6/store8a5dc82c/parentfaeedac8/testab1c5cf6. Final proper lint0errors; store retains2exact inherited warnings. Final configured project types pending; root actualpermission recovery remains separate.

Final combined configured UI typecheck natural0/64.952s and final configured Web typecheck natural0 after Claims401+ACP freeze; source guards retained. Source/tests remain frozen and unpublished; current isolated multi build is a new functional artifact. Stage3 actual single-user native permission recovery still pending; no successful permission-denial or original cancellation-cause attribution claimed.
