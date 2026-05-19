# Wave 5 Shared API And Client Boundary Cleanup Design

## Summary

Wave 5 hardens the shared frontend API client boundary under `apps/packages/ui/src/services/tldw/` so that the runtime source of truth is explicit, reviewable, and easier to evolve.

The key design decision is to treat this as a staged de-shadowing and contract-hardening pass, not as an all-at-once service-layer rewrite.

Today, `TldwApiClient.ts` contains a large legacy class body and then applies domain mixins onto the prototype with `Object.assign(...)`. That means many feature-facing methods appear to be owned by the class while the runtime behavior is actually owned by the domain modules that are assigned later. This hidden ownership is the main Wave 5 risk and velocity drag.

Wave 5 should:

- make `TldwApiClient` the explicit core transport and bootstrap layer
- make domain modules the explicit owners of feature-facing client behavior
- preserve the public client surface for existing consumers unless a change is required to remove ambiguity
- add tests that fail when ownership drifts or a shadowed method quietly reappears

Wave 5 should not attempt to replace the mixin pattern entirely. It should make the existing pattern legible and safe enough to support ongoing work without hidden runtime surprises.

## Goals

- Eliminate hidden runtime ownership where `TldwApiClient` methods are later overwritten by mixed-in domain methods.
- Make the ownership boundary between core client responsibilities and domain responsibilities explicit.
- Preserve current consumer-facing imports and the `tldwClient` singleton where possible so the cleanup reduces risk instead of creating avoidable churn.
- Keep shared request, path-resolution, hosted/proxy, and OpenAPI-path behavior in one predictable contract layer.
- Prevent contract drift from being discovered during UI work by adding targeted collision, export, and transport tests.
- Improve maintainer clarity around where new client behavior belongs and how OpenAPI verification should actually be run.

## Non-Goals

- Replacing the mixin-based client architecture with a fully composed service-object design.
- Rewriting backend endpoints or changing backend ownership boundaries.
- Refactoring every large domain module for style or file-size reasons alone.
- Normalizing feature-local compatibility fallbacks unless the same rule is clearly shared across multiple domains.
- Moving every type out of `TldwApiClient.ts` in this wave.
- Folding single-feature endpoint quirks into Wave 5 when they still belong to their owning subsystem wave.

## Current State

### Runtime Ownership Is Hidden

`apps/packages/ui/src/services/tldw/TldwApiClient.ts` currently exports the large client class and then applies domain mixins onto `TldwApiClient.prototype` with:

- `adminMethods`
- `mediaMethods`
- `characterMethods`
- `chatRagMethods`
- `collectionsMethods`
- `modelsAudioMethods`
- `presentationsMethods`
- `workspaceApiMethods`
- `webClipperMethods`

This means the file presents two overlapping ownership stories:

- the class body appears to define feature-facing methods
- the runtime behavior is often determined later by the mixed-in domain method of the same name

That overlap is not theoretical. It already spans the major domain surfaces, including admin, characters, chat/rag, collections, media, models/audio, presentations, and workspace API.

### Domain Modules Still Depend On `TldwApiClient.ts`

The domain modules are not yet cleanly isolated from `TldwApiClient.ts`.

They currently import:

- shared types from `../TldwApiClient`
- `TldwApiClientCore` from `../TldwApiClient`
- runtime helpers such as `normalizePersonaProfile`, `normalizePersonaExemplar`, and `clonePresentationVisualStyleSnapshot`

That means a naive "delete the overlapping class methods" pass would be unsafe. If runtime helpers are moved or removed in the wrong order, the wave risks introducing circular-dependency and export-break regressions while trying to remove ownership ambiguity.

### Shared Contract Logic Already Exists At Multiple Layers

The current code already has a meaningful split between:

- shared transport and path behavior in `request-core.ts`, `openapi-guard.ts`, and the core client helper methods
- feature-local compatibility and fallback behavior inside domain modules

Examples:

- `request-core.ts` already owns hosted-mode proxy routing, timeout derivation, and path normalization quirks
- `TldwApiClient.resolveApiPath()` already centralizes OpenAPI-path candidate resolution
- `domains/characters.ts` already contains feature-specific compatibility fallback logic for character list and query routes

This is important because not all fallback logic should be centralized. Some behavior is genuinely shared transport policy; some is feature-local compatibility code that still belongs with the owning domain.

### Verification And Operational Clarity Are Incomplete

Current tests protect some important surfaces, but not the Wave 5 failure mode directly.

Existing coverage includes:

- request-core hosted/proxy behavior
- request-core path normalization and absolute URL policy
- helper export coverage from `TldwApiClient.ts`

But current tests do not fail if:

- a class method is reintroduced that is later silently overwritten by a domain mixin
- the runtime ownership map drifts back into ambiguity

Operational clarity also has a gap:

- `openapi-guard.ts` documents `verify:openapi`
- the runnable script currently lives under `apps/extension/package.json`, not the UI package that owns the shared client code

Wave 5 should close that clarity gap instead of leaving a misleading maintenance instruction in place.

## Requirements Confirmed With User

- Wave 5 should include de-shadowing, not only shared contract tests.
- The design should be pressure-tested for scope risk before continuing.
- The wave should stay bounded and should not silently turn into a broad service-architecture rewrite.

## Approaches Considered

### Approach 1: Contract-Only Cleanup

Keep the current `TldwApiClient` structure, add better tests around request and path behavior, and leave the shadowed ownership alone.

Pros:

- Lowest immediate churn
- Easy to scope

Cons:

- Leaves the main developer-velocity problem intact
- Reviewers and maintainers still have to reason about dead-or-shadowed class implementations
- Does not make the runtime source of truth explicit

### Approach 2: Staged De-Shadowed Facade

Reduce `TldwApiClient` to core transport and bootstrap responsibilities, extract shared runtime helpers that domains currently import from it, and treat domain modules as the sole owners of feature-facing methods.

Pros:

- Best balance of risk reduction and maintainability payoff
- Fixes the real ownership ambiguity instead of only documenting around it
- Allows compatibility-safe migration through explicit stages

Cons:

- Requires a more disciplined sequence
- Needs stronger verification than a simple transport-only cleanup

### Approach 3: Full Service-Object Rewrite

Replace the mixin pattern with explicit composed services and remove the prototype-based client assembly entirely.

Pros:

- Cleanest architecture in the abstract

Cons:

- Too broad for this wave
- High churn with weaker short-term risk reduction
- Likely to reopen stable surfaces that are not currently failing

## Recommendation

Use Approach 2.

Wave 5 should deliver a staged de-shadowed facade:

- `TldwApiClient` becomes the explicit core client layer
- domain modules remain the explicit owners of feature-facing methods
- shared runtime helpers move out of `TldwApiClient` only when needed to break ownership ambiguity safely
- export compatibility is preserved during the transition through tests and temporary re-exports where practical

This is the highest-payoff cleanup that still matches the wave model.

## Proposed Architecture

### Ownership Model

After Wave 5, responsibilities should be split as follows.

`TldwApiClient.ts` should own only:

- config and storage bootstrap
- auth-aware request and upload primitives
- OpenAPI lookup, path normalization, and path-parameter filling
- client-side caches and singleton assembly
- the type surface that still reasonably belongs with the core client

Domain modules under `apps/packages/ui/src/services/tldw/domains/` should own:

- feature-facing methods for their subsystem
- feature-local response normalization
- feature-local compatibility handling that is not truly shared across domains

Shared helper modules under `apps/packages/ui/src/services/tldw/` should own:

- runtime helpers that are imported by more than one domain or that must exist outside the class file to avoid circular coupling

Important guardrail:

- do not create a generic helper dumping ground
- extract only the runtime helpers needed to make ownership explicit and dependencies sane

### Shared Contract Boundary

Wave 5 should define the shared boundary narrowly.

Shared contract logic in scope:

- hosted/proxy routing behavior
- auth injection and absolute-URL policy
- request timeout and path normalization behavior
- OpenAPI-path lookup and candidate resolution
- ownership and export compatibility of the shared client surface

Feature-local compatibility logic out of scope unless clearly shared:

- domain-specific fallback behavior that only serves one feature family
- single-endpoint response-shape quirks that do not affect other frontend consumers

Rule:

- if a compatibility rule appears only inside one domain and is tied to one feature family, keep it in that domain
- only centralize compatibility logic when at least two independent domains share the same behavior and the shared abstraction is simpler than duplication

### Staged Cutover

Wave 5 should not delete overlapping methods in one sweep.

It should proceed in stages.

#### Stage 0: Ownership Inventory And Collision Guard

Before removing or moving behavior, the implementation must build an ownership inventory that answers:

- which methods are core-owned
- which methods are domain-owned
- which runtime helpers are imported by domains from `TldwApiClient.ts`
- which overlaps currently exist between the class body and mixed-in domains

This stage also needs one authoritative comparison point for guard coverage.

Recommended options:

- expose a pre-mixin `TldwApiClientBase` export that contains the class implementation before `Object.assign(...)`
- or define an explicit ownership manifest checked by tests against both the base class and the mixed-in domain objects

Guardrail:

- do not rely on the final runtime prototype alone for collision detection, because the mixed-in methods have already overwritten the overlapping class methods by that point

Wave 5 should add a collision-guard test that fails when:

- a domain-owned method exists both in the class body and in a domain mixin
- a new overlap is introduced without an explicit migration decision

This turns hidden ownership drift into an immediate test failure.

#### Stage 1: Extract Shared Runtime Helpers

Before de-shadowing feature methods, move only the needed runtime helpers out of `TldwApiClient.ts`.

This includes helpers currently imported by domains for runtime behavior, such as:

- persona normalization helpers
- presentation visual-style cloning helpers
- similar non-method utilities that domains depend on directly

Guardrails:

- preserve consumer compatibility through re-exports from `TldwApiClient.ts` where that avoids broad churn
- treat runtime-helper re-exports as transitional by default unless the implementation plan explicitly justifies keeping a given helper as permanent public API
- do not use this stage as an excuse to move all types or all helpers out of the file

#### Stage 2: De-Shadow By Reviewable Slices

Once helper dependencies are safe, remove overlapping legacy class methods in reviewable slices.

Slice selection should be driven by Stage 0 evidence, not by a hardcoded order.

Selection criteria:

- low or moderate dependency coupling
- high overlap reduction per touched file
- small, coherent verification surface
- limited consumer churn outside the touched domain
- clear ownership payoff relative to the size of the slice

Candidate early slices include:

- admin
- workspace API
- presentations
- lower-coupling models/audio surfaces

Higher-coupling surfaces such as characters, chat/rag, media, and collections should be taken only if the ownership inventory and verification burden still remain coherent inside the same wave.

If the overlap proves too broad for one reviewable pass, Wave 5 must split the overflow into an explicit follow-on backlog rather than forcing every domain through one large cutover.

#### Stage 3: Shared Contract Hardening And Operational Clarity

After ownership is explicit, finish the wave by tightening the shared contract layer and its maintenance story.

This includes:

- request-core and path-resolution regression coverage
- export compatibility coverage
- documentation on where new client behavior belongs
- correcting or aliasing the actual `verify:openapi` command path so maintainers are not sent to the wrong workspace

## Testing Strategy

Wave 5 should add or update the following test classes.

### Ownership Collision Tests

Add a focused test that verifies:

- the agreed set of domain-owned method names is not redefined in the class body
- `Object.assign(...)` is not masking newly introduced collisions

Implementation note:

- the test must compare against either the pre-mixin `TldwApiClientBase` export or the explicit ownership manifest from Stage 0 so it can detect collisions before runtime prototype overwrites hide them

This is the most important new regression coverage in the wave.

### Export Compatibility Tests

Extend the existing module export coverage so it protects:

- helper re-exports relied on by domains
- any compatibility aliases introduced during helper extraction
- the stable availability of `tldwClient` and core client exports

### Shared Request Contract Tests

Keep and extend targeted coverage for:

- hosted-mode proxy routing
- path normalization
- absolute URL allowlist behavior
- OpenAPI-path resolution where a shared helper or alias changes

### Domain Regression Tests

Only the domains touched in a given de-shadow slice should get targeted regression updates. The wave should not require broad feature-suite expansion for untouched domains.

## Risks And Mitigations

### Risk: Wave Turns Into A Client Rewrite

Mitigation:

- require the Stage 0 ownership inventory first
- de-shadow in slices
- split overflow into an explicit follow-on backlog if verification becomes too broad

### Risk: Circular Imports Or Broken Runtime Helpers

Mitigation:

- extract runtime helpers before deleting overlapping methods
- preserve re-exports during the transition
- extend export tests immediately after extraction

### Risk: Shared Abstractions Swallow Feature-Local Logic

Mitigation:

- centralize only transport and path behavior that is truly shared
- keep feature-local compatibility logic in the owning domain unless at least two domains share it

### Risk: False Confidence From OpenAPI Verification Alone

Mitigation:

- treat `verify:openapi` as one guard, not the main guard
- add an explicit ownership collision test because OpenAPI validation does not protect against shadowed runtime methods

### Risk: Maintainers Still Do Not Know Where To Put New Methods

Mitigation:

- end the wave with a short maintainer note that defines:
  - what belongs in `TldwApiClient.ts`
  - what belongs in domain modules
  - when a shared helper module is justified

## Definition Of Done

Wave 5 is complete only when:

- the core vs domain ownership boundary is explicit and tested
- the Stage 0 inventory is recorded and the Wave 5 implementation plan names the de-shadowing slices actually in scope for this wave
- every Wave 5 in-scope slice has landed without export regressions, or has an explicit handoff into a follow-on backlog with the remaining overlaps and next action named
- `TldwApiClient.ts` no longer presents overlapping ownership for the completed slice
- shared runtime helpers used by domains no longer require risky coupling to the class file
- request-core and path-resolution behavior remain green
- the actual OpenAPI verification workflow is documented or aliased from the correct workspace
- the touched client surface is easier to review and modify than before

## Success Criteria

Wave 5 is successful if maintainers can answer these questions without reading both a large class body and the final `Object.assign(...)` block:

- where should a new shared client capability live
- which file owns the runtime behavior for an existing method
- which compatibility rules are shared transport policy versus feature-local exceptions
- which tests fail if the ownership boundary drifts again
