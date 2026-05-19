# Branch-Aware WebUI and Extension Artifact Profiles Design

Date: 2026-04-10
Status: Approved for planning
Owner: Codex brainstorming session

## Summary

Make local WebUI and browser-extension builds branch-aware so `main` produces user-ready production artifacts while every other branch produces development artifacts. The default `build` command should follow the checked-out branch automatically, while explicit `build:prod` and `build:dev` commands remain available for deterministic local and CI usage.

For the WebUI, branch-aware behavior maps onto the existing networking split already present in the codebase:

- `production` profile uses the current quickstart-style artifact
- `development` profile uses the current advanced/custom-host artifact

For the extension, branch-aware behavior changes artifact profile and output naming, not the meaning of the live `dev` command. Non-`main` builds remain packaged extension artifacts, but they are branded in exported artifact names as development builds. The implementation should keep canonical internal build roots stable for tooling, then emit developer-facing unpacked install directories and zipped artifacts with a `-dev` suffix.

The design keeps local behavior aligned with branch intent while preserving explicit production validation in CI so `main`-grade artifacts do not silently degrade.

## Problem

The repository already distinguishes between development workflows and production-oriented release paths, but that distinction is inconsistent across local build entrypoints:

- the WebUI release container path already bakes in production-ready quickstart settings
- the WebUI local build command is not branch-aware
- the extension distinguishes `wxt` live development from `wxt build`, but packaged artifact builds are not branch-aware
- contributors on feature branches can accidentally produce artifacts that look release-like even when they are intended for developer testing
- there is no single shared contract that says which branches produce user-ready artifacts versus developer artifacts

The requested product policy is clear:

- `main` is the user-facing branch and should build artifacts that are ready for normal use
- `dev` and every other non-`main` branch are for development and technical users

That policy should apply locally by default, not only inside GitHub Actions.

## Goals

- Make local `build` commands branch-aware for both the WebUI and extension
- Treat `main` as the only branch that produces production artifacts by default
- Treat every branch other than `main` as a development artifact source by default
- Keep explicit overrides available through `build:prod` and `build:dev`
- Reuse the current WebUI quickstart versus advanced networking model instead of inventing a second release-mode abstraction
- Keep extension development branding in filenames rather than in the user-facing UI
- Apply `-dev` naming to exported unpacked extension install directories and zipped extension artifacts while keeping internal canonical build roots stable
- Keep CI and release automation deterministic by allowing forced production builds
- Minimize disruption to existing developer flows and existing release/container conventions

## Non-Goals

- Replace `next dev` or `wxt` live development behavior
- Change which branch names are protected or recommended for contributors
- Add visible in-app WebUI or extension UI banners for development artifacts
- Redesign store-publishing or browser-extension signing workflows
- Replace the existing WebUI production container strategy
- Introduce a new global release-management system for all repository artifacts

## Current State

### WebUI

The WebUI package in [`apps/tldw-frontend/package.json`](../../../apps/tldw-frontend/package.json) already separates `dev` from `build`, but `build` itself is not branch-aware.

The existing networking validator in [`apps/tldw-frontend/scripts/validate-networking-config.mjs`](../../../apps/tldw-frontend/scripts/validate-networking-config.mjs) already enforces two meaningful artifact modes:

- `quickstart`
  - requires `TLDW_INTERNAL_API_ORIGIN`
  - rejects an absolute browser-visible `NEXT_PUBLIC_API_URL`
- `advanced`
  - requires an absolute `NEXT_PUBLIC_API_URL`

The WebUI Docker release path already treats the user-ready artifact as a quickstart-style build:

- [`Dockerfiles/Dockerfile.webui`](../../../Dockerfiles/Dockerfile.webui)
- [`.github/workflows/publish-ghcr-main.yml`](../../../.github/workflows/publish-ghcr-main.yml)

This means the repository already has a stable definition of a production WebUI artifact. The gap is that local builds do not default to that definition only on `main`.

### Extension

The extension package in [`apps/extension/package.json`](../../../apps/extension/package.json) already separates:

- `dev` for live WXT development
- `build` and `build:chrome` / `build:firefox` / `build:edge` for packaged artifacts

However, packaged artifact builds are not branch-aware today. A developer on a feature branch still produces standard-looking build outputs unless they manually treat them as non-release artifacts.

The extension also has many tests, helper scripts, and docs that directly reference current unsuffixed paths such as `build/chrome-mv3` and `.output/chrome-mv3`. That makes a full path rename much higher risk than the original proposal first implied.

The current WXT config in [`apps/extension/wxt.config.ts`](../../../apps/extension/wxt.config.ts) does not yet carry an explicit artifact-profile concept.

### CI

CI workflows already distinguish between PR validation and `main` publishing, but they do not consistently use explicit artifact-profile wrappers because those wrappers do not exist yet.

This creates a risk that branch-aware local defaults could reduce production-path coverage unless CI deliberately forces production builds in at least one required path.

## Proposed Design

### 1. Introduce one shared build-profile resolver

Add a small shared script under `apps/scripts/`, for example:

- `apps/scripts/resolve-build-profile.mjs`

Responsibilities:

- determine the effective build profile
- emit `production` or `development`
- support both package-level build wrappers and CI usage

Resolution order:

1. `TLDW_BUILD_PROFILE` environment override
2. current git branch
3. fallback to `development` if branch detection is unavailable

Branch rule:

- `main` => `production`
- any other branch => `development`

The fallback to `development` is deliberate. If someone builds from a source export or a detached environment without usable git metadata, the safer default is the non-release artifact. Production automation should use explicit `build:prod` commands rather than depend on branch detection.

### 2. Add explicit and implicit build entrypoints

The WebUI package should expose:

- `build`
- `build:prod`
- `build:dev`
- `compile`
- `compile:prod`
- `compile:dev`

The extension package should expose the same top-level commands, plus profile-aware browser-specific and archive commands because those are the entrypoints developers and CI actually use today:

- `build`
- `build:prod`
- `build:dev`
- `build:chrome`
- `build:chrome:prod`
- `build:chrome:dev`
- `build:firefox`
- `build:firefox:prod`
- `build:firefox:dev`
- `build:edge`
- `build:edge:prod`
- `build:edge:dev`
- `zip`
- `zip:prod`
- `zip:dev`
- `zip:firefox`
- `zip:firefox:prod`
- `zip:firefox:dev`

Behavior:

- `build` uses branch-aware resolution
- `build:prod` forces `TLDW_BUILD_PROFILE=production`
- `build:dev` forces `TLDW_BUILD_PROFILE=development`

This hybrid model avoids two common failure modes:

- implicit-only behavior that becomes hard to reason about in CI
- explicit-only behavior that contributors forget to use locally

### 3. WebUI profile mapping

The WebUI should keep using `next build`; branch-aware behavior should only control the environment injected before the build runs. Any alternate artifact-producing WebUI entrypoint, including the existing webpack-based `compile` path, should use the same profile resolution and env shaping so it does not become a bypass around the branch policy.

#### Production profile

`production` should map to the current quickstart-style artifact:

- set `NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=quickstart`
- clear `NEXT_PUBLIC_API_URL`
- preserve or default `TLDW_INTERNAL_API_ORIGIN`
- allow `NEXT_PUBLIC_API_BASE_URL`, `NEXT_PUBLIC_API_VERSION`, and similar release-safe values if explicitly provided

Default:

- if `TLDW_INTERNAL_API_ORIGIN` is not set, default it to `http://127.0.0.1:8000` for local production-style builds

This keeps `main` local builds aligned with the existing user-ready container and quickstart story.

#### Development profile

`development` should map to the current advanced/custom-host artifact:

- set `NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced`
- require `NEXT_PUBLIC_API_URL` to remain present and absolute
- preserve the current `.env.local`-driven developer workflow

This keeps non-`main` branches aligned with the technical-user and custom-backend development story already documented in [`apps/tldw-frontend/README.md`](../../../apps/tldw-frontend/README.md).

#### WebUI wrapper implementation

Add a wrapper script, for example:

- `apps/tldw-frontend/scripts/build-with-profile.mjs`

Responsibilities:

- call the shared resolver
- prepare the right env for `production` or `development`
- invoke the existing `next build` command
- reuse the existing validator as the final guardrail

The validator remains authoritative; the wrapper only chooses the profile and populates defaults.

### 4. Extension profile mapping

The extension should continue to distinguish live development from packaged artifacts:

- `dev` still means live WXT development server
- `build` still means packaged extension artifacts

Branch-aware behavior should not replace packaged builds with `wxt` live mode.

#### Production profile

`production` should preserve the current artifact behavior:

- standard output names
- standard packaging names
- current production-only restrictions remain enforced

#### Development profile

`development` should still produce packaged artifacts, but those exported artifacts must be visibly distinguishable in filenames.

Required naming behavior:

- exported unpacked install directories include `-dev`
- zipped artifact names include `-dev`

Examples:

- `build/chrome-mv3-dev`
- `build/firefox-mv2-dev`
- `tldw-assistant-chrome-dev.zip`

The branding should stay at the exported artifact layer rather than by broadly renaming extension identity everywhere. That reduces compatibility risk for browser-specific behavior, extension IDs, and future store-related workflows.

Canonical internal build roots used by existing tooling should remain unchanged. In practice, the wrapper should:

- build into the current canonical roots expected by WXT and existing helpers
- run post-build verification against those canonical roots
- then materialize suffixed exported install directories such as `build/chrome-mv3-dev`

This preserves the user's requirement that developer-facing unpacked artifacts carry `-dev` while avoiding a broad internal path migration.

#### Extension wrapper implementation

Add a wrapper script, for example:

- `apps/extension/scripts/build-with-profile.mjs`

Responsibilities:

- call the shared resolver
- export a clear profile env such as `TLDW_BUILD_PROFILE` or `VITE_TLDW_ARTIFACT_PROFILE`
- invoke the current browser-specific `wxt build` commands
- preserve canonical internal output roots for tooling
- create suffixed exported install directories and archive names according to the resolved profile

The development suffix should apply consistently across:

- exported Chrome unpacked artifacts
- exported Firefox unpacked artifacts
- exported Edge unpacked artifacts
- zip packaging outputs

### 5. Keep CI deterministic

Local `build` commands should remain branch-aware, but CI must not rely on implicit branch detection in every important path.

Rules:

- `main` publishing jobs force `build:prod`
- any release-critical container or packaging workflow forces `build:prod`
- at least one required validation job for the WebUI and extension should also force production builds on pull requests

This preserves ongoing verification of the real production artifact even when contributors work almost entirely on non-`main` branches.

### 6. Documentation contract

Update contributor-facing docs for the WebUI and extension to state:

- `main` builds user-ready artifacts
- every other branch builds development artifacts by default
- `build:prod` is the escape hatch when a contributor needs a production artifact from a non-`main` branch
- extension development artifacts are intentionally filename-branded with `-dev`

This should be documented in:

- [`apps/tldw-frontend/README.md`](../../../apps/tldw-frontend/README.md)
- [`apps/extension/README.md`](../../../apps/extension/README.md)
- any release checklist or contributor guide that currently assumes all packaged outputs are release-like by default

## Design Details

### Shared contract

The repository should treat artifact profile as a first-class concept with exactly two values:

- `production`
- `development`

It should not overload:

- `NODE_ENV`
- Next.js internal production versus development runtime behavior
- WXT live development mode

Those concepts remain separate. The new build profile answers a narrower question:

> Is this artifact intended to be the user-ready `main` artifact, or a developer-oriented branch artifact?

### Why branch-aware by default

The user-facing product policy is branch-based, not person-based. A contributor on a feature branch should not have to remember special commands just to avoid producing release-like artifacts. The default build behavior should reflect repository intent automatically.

### Why explicit overrides still matter

Some workflows need determinism more than convenience:

- release jobs
- production smoke validation
- debugging a production-only issue from a feature branch

Explicit override commands solve that without weakening the local default policy.

### Why extension branding stays in filenames

The user requested visible development branding only in filenames. This is the lowest-risk place to create separation because it:

- reduces accidental sharing of non-release artifacts as if they were production builds
- avoids intrusive in-product messaging
- does not require new runtime UI logic
- avoids unnecessary manifest-level churn outside the packaging layer

## Testing Strategy

### Unit tests

Add focused tests for the shared resolver:

- `main` resolves to `production`
- `dev` resolves to `development`
- arbitrary feature branches resolve to `development`
- `TLDW_BUILD_PROFILE` override wins over branch detection
- missing branch metadata falls back to `production`

Add WebUI wrapper tests:

- `production` injects quickstart-compatible env
- `development` injects advanced-compatible env
- invalid env combinations still fail through the validator

Add extension wrapper tests:

- `production` preserves standard output naming
- `development` preserves canonical internal output roots
- `development` adds `-dev` to exported unpacked install directories
- `development` adds `-dev` to zip artifact names

### Integration checks

WebUI:

- run a forced production build from a non-`main` branch
- run a forced development build from a non-`main` branch
- confirm local branch-aware `build` matches the expected profile

Extension:

- run branch-aware `build:chrome` on a non-`main` branch and verify canonical internal roots still work
- verify exported install directories carry the `-dev` suffix
- run forced `build:prod` from a non-`main` branch and verify standard naming
- run zip packaging in both profiles and verify suffix behavior

CI:

- ensure at least one required job invokes forced production builds

## Risks and Mitigations

### Risk: WebUI local production builds become harder to use

If `main` forces quickstart mode locally, a contributor who expects advanced/custom-host behavior may be surprised.

Mitigation:

- keep `build:dev` available
- document the profile rule clearly
- use the existing quickstart defaults rather than inventing a new local production story

### Risk: CI stops covering production artifacts

If CI only uses implicit branch-aware builds, most PRs will validate only development artifacts.

Mitigation:

- require explicit `build:prod` in at least one important PR validation path

### Risk: Extension tooling assumes fixed output paths

Some Playwright or packaging utilities assume current non-suffixed canonical paths such as `build/chrome-mv3` or `.output/chrome-mv3`.

Mitigation:

- keep canonical internal build roots unchanged
- add `-dev` only to exported install directories and archive names
- update path-discovery utilities only where they need to discover exported developer-facing artifacts
- add tests that prove existing tooling still works with canonical paths

## Rollout Plan

1. Add the shared resolver script under `apps/scripts/`
2. Add WebUI and extension wrapper scripts
3. Add explicit `build:prod` and `build:dev` scripts
4. Update extension artifact path and naming logic for `-dev`
5. Update CI workflows to force production builds where required
6. Update contributor and release documentation
7. Validate both local branch-aware defaults and explicit override paths

## Approval

This design is approved for implementation planning once the written spec is reviewed by the user.
