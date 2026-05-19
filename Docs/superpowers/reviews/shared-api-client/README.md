# Shared API Client Review Artifacts

This directory stores review artifacts for the Wave 5 shared API client boundary cleanup in `apps/packages/ui`.

Current artifacts:

- `2026-04-17-wave5-ownership-inventory.md`: the live overlap inventory between the pre-mixin `TldwApiClientBase` class body and the domain mixins, plus the bounded Wave 5 slice decision.

The code manifest at `apps/packages/ui/src/services/tldw/client-ownership.ts` is the authoritative branch-local overlap baseline. These review docs mirror that state so later waves can compare new cleanup work against the live branch instead of older planning notes.
