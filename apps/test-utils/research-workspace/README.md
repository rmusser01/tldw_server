# Research Workspace Parity Contract

Shared deterministic parity contract for `/research-workspace` across:

- WebUI (`apps/tldw-frontend/e2e/workflows/research-workspace.parity.spec.ts`)
- Extension (`apps/extension/tests/e2e/research-workspace.parity.spec.ts`)

## Scope (PR Gate)

The contract validates:

- route boot + baseline pane visibility (`sources`, `chat`, `studio`)
- studio generated output section behavior
- deterministic artifact rendering and action controls
- accordion state transition (collapse and restore)

## Run Commands

From `apps/tldw-frontend`:

```bash
bun run e2e:research-workspace:parity
```

From `apps/extension`:

```bash
bun run test:e2e:workspace-parity
bun run test:e2e:workspace-parity:strict
```

## Deep Coverage

Real-backend coverage remains separate and is run by:

- WebUI: `e2e/workflows/research-workspace.real-backend.spec.ts`
- Extension: `tests/e2e/research-workspace.real-backend.spec.ts`
