# Serial four-cell source/dependency isolation

Preparation only, TASK13260, 2026-09-16. No archive, worktree, dependency copy, install, runtime, browser or configuration change was performed. The [full-run gate remains closed](../../IMPLEMENTATION_PLAN_uat_cycle_5.md). This supplements [PROTOCOL.md](PROTOCOL.md); commands below are for a later approved preparation/preflight, not records of execution.

## Recommendation

Use **four separate archives of one released commit, executed serially**, with fresh data/browser state per cell. Reuse the installed dependencies through ordinary macOS copy-on-write copies, not application-source symlinks. Share one private, sanitized Python dependency copy across the four cells; keep each cell's frontend dependency tree inside its source root. This uses normal `python -m` and the existing Next command/config. It needs no dependency installation, product patch, custom import hook, or enlarged Turbopack root.

macOS `cp -c` uses `clonefile`; its installed manual says it falls back to a normal copy across unsupported filesystems. `cp -a` preserves symlinks instead of following them. Thus `cp -ac` reuses current bytes and can share physical storage while giving each cell independent paths. No speed, disk-space saving or successful clone is claimed until the actual preparation runs. A fallback full copy is still reused dependencies, not a fresh dependency installation.

## What the current installation actually contains

| Inspected boundary | Observation / consequence |
|---|---|
| Python editable install | `.venv/lib/python3.11/site-packages/__editable__.tldw_server-0.1.32.pth` loads a finder mapping **both `tldw_Server_API` and `mcp_unified` to `.worktrees/standalone-html-presentations`**. Its `direct_url.json` confirms that older editable root. It can act as fallback even when an earlier path provides the top-level package. Merely prepending a snapshot path does not remove that fallback. |
| Other editable path | `__editable__.backlog_py-0.1.0.pth` adds `.worktrees/backlog-py-inventory-scaffold/tools/backlog-py/src`. It is not required for normal application runtime; the controller can continue using its existing Backlog CLI outside the application process. |
| Required application roots | [setuptools discovery](../../pyproject.toml#L499) has exactly three: repository root, `apps/mcp-unified/src`, `packages/tldw_profile_core/src`. The editable mapping does **not** supply the third. |
| Other `.pth` entries | `_virtualenv`, `distutils-precedence`, `coloredlogs`, `google_auth` namespace setup, and `pytest-cov` are present. Keep third-party startup behavior; do not broadly disable `site` or all `.pth` processing. No direct symlinks were found at site-packages' top level. |
| Interpreter / entrypoints | `.venv/bin/python` is a symlink to the installed uv CPython3.11.13 executable. `pyvenv.cfg` identifies that installation. Copied console scripts/activation scripts may retain original absolute paths: use the private copy's absolute `bin/python -m ...`, not copied shebang entrypoints or copied `activate`. Verify its `sys.prefix` before use. |
| Frontend package manager | Current installed layout is **Bun**, `apps/node_modules/.bun`, regardless of older pnpm terminology. `frontend/node_modules/@tldw/ui -> ../../../packages/ui`. Linking the whole original frontend `node_modules` exposes original workspace source. |
| External packages | Next16.1.4, React18.3.1, React DOM and TypeScript links are relative into `apps/node_modules/.bun`. A read-only scan checked **8,754 package-edge symlinks**, all resolved inside that store, zero broken. This was a package-edge scan, not a universal audit of every nested file. |
| One absolute alias | `apps/node_modules/node_modules` points absolutely back to original `apps/node_modules`. In a copied dependency tree, replace only that copied link with a local `.` target, or omit it if proven unnecessary. Do not alter the original link. |
| Bundler boundary | [Next config](../../apps/tldw-frontend/next.config.mjs#L183) sets `turbopack.root` and `outputFileTracingRoot` to the repository root. Installed `next/dist/build/swc/generated-native.d.ts:113` says files must reside beneath `rootPath`; `hot-reloader-turbopack.js:177` selects this configured root. A `.bun` store symlink outside the cell therefore does not meet the declared boundary. Do not broaden root to the mutable checkout to accommodate it. |

## Smallest later preparation

1. **Choose the release once.** Parent records the approved full commit and gate decision. Archive that exact commit for each of `sqlite-single`, `sqlite-multi`, `pg-single`, `pg-multi`; do not copy the dirty checkout. An archive has no `.git`, but [get_project_root](../../tldw_Server_API/app/core/Utils/Utils.py#L149) supports `pyproject.toml` plus `tldw_Server_API` as its root marker. Retain the source manifest before runtime-generated files appear.
2. **One Python dependency copy.** Copy the existing `.venv` with `cp -ac` to a new private donor path once. In that copy only, rename the inspected `__editable__.tldw_server-0.1.32.pth` and `__editable__.backlog_py-0.1.0.pth` files to a non-`.pth` suffix. The tldw_server finder maps both app and MCP; backlog_py is the second entry. Keep a before/after manifest and the old files; leave all other `.pth` entries alone. No reinstall or change to the original environment. Recheck the inventory before doing this: fail if new editable hooks or project-source path entries appear. All four cells can call this donor's absolute `bin/python`; set `PYTHONDONTWRITEBYTECODE=1` so shared bytecode is not a new mutable cross-cell cache. Do not expose this as clean-machine installation.
3. **Cell-local frontend dependencies.** Reuse three existing trees at the same relative locations: `apps/node_modules`, `apps/tldw-frontend/node_modules`, `apps/packages/ui/node_modules`. `cp -ac` each into the newly created archive; preserve relative symlinks and fix the copied absolute self-link above. Exclude/remove only the new copies of `.vite`/`.cache` caches, not their originals. Do not copy `.next*` from the working checkout. This preserves the standard workspace/store geometry and keeps external package files under the unchanged Turbopack root. No `NODE_PATH`, `--preserve-symlinks`, package-manager install or Next-config workaround is needed.
4. **Adapt the private launcher, not product code.** Existing [recovery launcher](../fresh-uat-recovery-20260916/recovery-launcher.mjs) already separates app environment, profile state and Next build directories. Give it explicit frozen-source-root and private-Python paths instead of deriving both from its own location. Its `git show HEAD:...` template reads must read the already-manifested archive files instead: an archive has no repository and later mutable HEAD must not influence the run. Keep the existing config scrubbing/private dotenv/credential policy, all three Python source roots, and profile-specific storage overrides. Set `WORKFLOWS_ARTIFACTS_DIR` as already required. Use a real `.next-live-tier-<cell-run-id>` child of that cell's frontend, never a build-directory symlink.
5. **Run serially without reusing application state.** Start only the selected cell; preserve and stop it before the next. Each cell retains its own source-root `Databases/system_ops.json`, scraper state, document drafts, private runtime DB/cache/log paths, Next build and browser context. Reuse neither target-acceptance data nor the previous cell's source-root runtime files. The external inference service may be shared under the existing single-generation lease discipline; its identity/capabilities are recorded explicitly.
6. **Keep PostgreSQL provisioning unchanged.** Use [official fixture holder pattern](../fresh-uat-recovery-20260916/run-pg-holder.mjs) and the snapshot's `tests/_plugins/postgres.py`, with fresh separate auth/content fixture databases for each PG cell. The holder can run with the same private Python dependency copy and three snapshot source paths. Keep pytest/plugin variables in the holder only; never pass them to the app. No handwritten database creation or AuthNZ test pool. Initialize/login through the normal approved paths with the snapshot interpreter.

The copied dependency files are intentionally reused installation artifacts. Snapshot application files and first-use mutable state are fresh. Sharing physical clone extents is safe from cross-cell writes; hard-linking mutable files would not provide that property. The original model binaries, uv interpreter and installed Playwright browser may also be reused and must be listed as external dependencies.

## Concrete preflight checks after preparation

These are path/metadata checks, not substitutes for actual startup and the native matrix. `CELL_ROOT` must name the newly prepared archive; `DEPENDENCY_VENV` the private dependency copy; neither variable replaces HOME/CODEX_HOME. For repository policy, source the existing project venv before Python shell checks, then invoke the explicit private interpreter; do not rely on the copied activation script.

### Python: normal site startup, frozen application sources

```sh
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$CELL_ROOT:$CELL_ROOT/apps/mcp-unified/src:$CELL_ROOT/packages/tldw_profile_core/src" \
"$DEPENDENCY_VENV/bin/python" - <<'PY'
import importlib.util
import os
from pathlib import Path
import sys

cell = Path(os.environ['CELL_ROOT']).resolve()
deps = Path(os.environ['DEPENDENCY_VENV']).resolve()
assert Path(sys.prefix).resolve() == deps, (sys.prefix, deps)
assert not any('__editable__' in getattr(f, '__module__', '') for f in sys.meta_path)
assert not any('/.worktrees/' in str(p) for p in sys.path)
for name in ('tldw_Server_API', 'mcp_unified', 'tldw_profile_core'):
    spec = importlib.util.find_spec(name)  # top-level lookup, not app import
    assert spec and spec.origin, name
    origin = Path(spec.origin).resolve()
    assert origin.is_relative_to(cell), (name, origin)
    print(name, origin)
for name in ('uvicorn', 'fastapi', 'pydantic', 'psycopg'):
    spec = importlib.util.find_spec(name)
    assert spec and spec.origin, name
    assert Path(spec.origin).resolve().is_relative_to(deps), (name, spec.origin)
print('python-prefix-and-source-paths-ok')
PY
```

Export the two task variables before invoking this snippet. Run from the profile's private CWD, with the launcher's ordinary environment allowlist. An unexpected origin is a preparation failure; do not compensate by adding the original checkout to PYTHONPATH. Since editable fallback is removed, missing snapshot source fails explicitly. `find_spec` verifies initial resolution, not every future dynamic import; verify actual app startup logs/module origins before accepting the cell.

### Frontend: real paths and workspace aliases remain within the cell

```sh
node --input-type=module <<'JS'
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const cell = fs.realpathSync(process.env.CELL_ROOT);
const frontend = path.join(cell, 'apps/tldw-frontend');
const require = createRequire(path.join(frontend, 'package.json'));
const inside = p => p === cell || p.startsWith(cell + path.sep);
for (const name of ['next/package.json', 'react/package.json', 'react-dom/package.json', 'typescript/package.json']) {
  const resolved = fs.realpathSync(require.resolve(name));
  if (!inside(resolved)) throw Error(`${name} escaped: ${resolved}`);
  console.log(name, resolved);
}
const ui = fs.realpathSync(path.join(frontend, 'node_modules/@tldw/ui'));
if (ui !== path.join(cell, 'apps/packages/ui')) throw Error(`workspace escaped: ${ui}`);
const ts = JSON.parse(fs.readFileSync(path.join(frontend, 'tsconfig.json'), 'utf8'));
for (const key of ['@tldw/ui', '@tldw/ui/*', '@/*', '~/*']) {
  for (const value of ts.compilerOptions.paths[key]) {
    if (!inside(path.resolve(frontend, value))) throw Error(`alias escaped: ${key}`);
  }
}
console.log('frontend-dependency-and-alias-paths-ok');
JS
```

Also walk symlinks without following them in all three copied node_modules trees; resolve each link and reject broken links or a destination outside `CELL_ROOT`. This includes the `.bun` package edges and catches newly introduced absolute workspace links. Retain the resolved Next CLI path, actual Next config root, config/lockfile/source hashes, and dependency package versions. Launcher must use `createRequire(<cell frontend>/package.json).resolve('next/dist/bin/next')`, with CWD at that frontend. Check real Next startup/page loading after release; static path checks do not prove successful compilation or generated CJS/ESM alias resolution.

## Unverified boundaries / why this choice

No copy, path probe against a prepared cell, server startup or compilation has run here. APFS clone support, relocation of the private Python interpreter/site-packages, and actual Next build behavior require later preflight. This approach retains ordinary site processing and existing relative frontend dependency topology, removing only proven mutable-source links. It is smaller than maintaining a custom Python import interceptor or widening the bundler root. If the private dependency copy fails preflight, document the exact failure before changing the scheme; do not silently fall back to mutable source or reinterpret a targeted profile as fresh.
