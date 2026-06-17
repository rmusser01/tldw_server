# ACP Aider And Continue Entrypoint Decisions

## Goal

Resolve the remaining Aider and Continue child issues for ACP live certification without making unsupported compatibility claims.

## Current Evidence

Aider 0.86.2 is installed locally and direct one-shot prompting works with the local llama.cpp setup, but the installed Aider CLI exposes no native ACP stdio server command. A third-party `aider-acp` bridge exists, so Aider has an external adapter candidate, but that adapter is not installed or live-certified here.

Continue CLI 1.5.46 is available through the npm package `@continuedev/cli` as the `cn` binary. The local shell still resolves `continue` as a shell builtin, and `cn --help` exposes interactive/headless/review modes but no ACP stdio server command.

## Design

Keep Aider and Continue at `documented_unverified` / `documented_only`.

Represent Aider as an `external_acp_adapter` candidate with `acp_command: aider-acp`, adapter metadata, and explicit caveats that no live support claim exists until the adapter is installed and initialize/session/prompt evidence passes.

Represent Continue as a `documented_candidate` using display command `cn`, with notes explaining the historical `continue` shell-builtin collision and the lack of an ACP stdio entrypoint in the current CLI package.

Update the compatibility matrix, certification checklist, and old OSS/custom certification note so the remaining parent tracker can be reconciled after these child decisions merge.

## Non-Goals

- Do not install or vendor `aider-acp`.
- Do not build an Aider or Continue adapter in this PR.
- Do not promote either profile to `supported_with_caveats`.
- Do not change OpenCode, Goose, Hermes, Codex, or custom-profile support state except for parent-tracker wording if needed.

## Verification

Use TDD around the seeded registry rows and registry-backed certification manifests:

- Aider row exposes the adapter candidate metadata and classifies as blocked when `aider-acp` is unavailable.
- Continue row uses `cn`, remains documented-only, and keeps `entrypoint_strategy_missing` as the certification blocker.
- Manifest rendering refuses blocked Aider adapter runs and documented-only Continue runs.
- Docs include the decision evidence and do not imply live ACP support.
