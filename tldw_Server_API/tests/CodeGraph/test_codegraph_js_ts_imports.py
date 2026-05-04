from __future__ import annotations

import json
from pathlib import Path

from tldw_Server_API.app.core.CodeGraph.extractors.js_ts_imports import (
    load_js_ts_project_config,
    resolve_js_ts_import,
)


def test_relative_import_resolves_extensionless_typescript_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "app.ts").write_text("import { helper } from './utils';\n", encoding="utf-8")
    (workspace / "src" / "utils.ts").write_text("export const helper = 1;\n", encoding="utf-8")

    result = resolve_js_ts_import(workspace, "src/app.ts", "./utils")

    assert result.resolution_kind == "relative"
    assert result.resolved_path == "src/utils.ts"
    assert result.reason is None


def test_parent_relative_import_resolves_tsx_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src" / "pages").mkdir(parents=True)
    (workspace / "src" / "shared").mkdir(parents=True)
    (workspace / "src" / "pages" / "app.ts").write_text(
        "import { Button } from '../shared/button';\n",
        encoding="utf-8",
    )
    (workspace / "src" / "shared" / "button.tsx").write_text(
        "export function Button() { return <button />; }\n",
        encoding="utf-8",
    )

    result = resolve_js_ts_import(workspace, "src/pages/app.ts", "../shared/button")

    assert result.resolution_kind == "relative"
    assert result.resolved_path == "src/shared/button.tsx"


def test_tsconfig_paths_resolve_common_frontend_aliases(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_json(
        workspace / "apps" / "tldw-frontend" / "tsconfig.json",
        {
            "compilerOptions": {
                "baseUrl": ".",
                "paths": {
                    "@/*": ["../packages/ui/src/*"],
                    "~/*": ["../packages/ui/src/*"],
                    "@web/*": ["./*"],
                    "@tldw/ui/*": ["../packages/ui/src/*"],
                },
            }
        },
    )
    (workspace / "apps" / "tldw-frontend" / "pages").mkdir(parents=True)
    (workspace / "apps" / "packages" / "ui" / "src" / "components").mkdir(parents=True)
    (workspace / "apps" / "tldw-frontend" / "lib").mkdir(parents=True)
    (workspace / "apps" / "packages" / "ui" / "src" / "components" / "Button.tsx").write_text(
        "export function Button() { return <button />; }\n",
        encoding="utf-8",
    )
    (workspace / "apps" / "tldw-frontend" / "lib" / "routes.ts").write_text(
        "export const routes = [];\n",
        encoding="utf-8",
    )

    assert (
        resolve_js_ts_import(
            workspace,
            "apps/tldw-frontend/pages/index.tsx",
            "@/components/Button",
        ).resolved_path
        == "apps/packages/ui/src/components/Button.tsx"
    )
    assert (
        resolve_js_ts_import(
            workspace,
            "apps/tldw-frontend/pages/index.tsx",
            "~/components/Button",
        ).resolved_path
        == "apps/packages/ui/src/components/Button.tsx"
    )
    assert (
        resolve_js_ts_import(
            workspace,
            "apps/tldw-frontend/pages/index.tsx",
            "@tldw/ui/components/Button",
        ).resolved_path
        == "apps/packages/ui/src/components/Button.tsx"
    )
    assert (
        resolve_js_ts_import(
            workspace,
            "apps/tldw-frontend/pages/index.tsx",
            "@web/lib/routes",
        ).resolved_path
        == "apps/tldw-frontend/lib/routes.ts"
    )


def test_project_config_loads_nearest_tsconfig(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_json(workspace / "tsconfig.json", {"compilerOptions": {"baseUrl": "src"}})
    _write_json(
        workspace / "packages" / "web" / "tsconfig.json",
        {"compilerOptions": {"baseUrl": ".", "paths": {"@web/*": ["src/*"]}}},
    )

    config = load_js_ts_project_config(workspace, "packages/web/src/app.ts")

    assert config is not None
    assert config.config_path == "packages/web/tsconfig.json"
    assert config.base_url == "packages/web"
    assert config.paths == {"@web/*": ("src/*",)}


def test_invalid_jsonc_project_config_does_not_abort_resolution(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "app.ts").write_text("import Button from '@/Button';\n", encoding="utf-8")
    (workspace / "tsconfig.json").write_text(
        """
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
    },
  },
}
""",
        encoding="utf-8",
    )

    result = resolve_js_ts_import(workspace, "src/app.ts", "@/Button")

    assert result.resolution_kind == "external"
    assert result.reason == "external_package"


def test_alias_targets_escaping_workspace_are_not_resolved(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_json(
        workspace / "tsconfig.json",
        {"compilerOptions": {"baseUrl": ".", "paths": {"@bad/*": ["../outside/*"]}}},
    )
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.ts").write_text("export const secret = 1;\n", encoding="utf-8")

    result = resolve_js_ts_import(workspace, "src/app.ts", "@bad/secret")

    assert result.resolved_path is None
    assert result.resolution_kind == "unresolved"
    assert result.reason == "alias_target_escapes_workspace"
    assert result.candidates == ()


def test_external_package_import_is_not_resolved_into_node_modules(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "node_modules" / "react").mkdir(parents=True)
    (workspace / "node_modules" / "react" / "index.js").write_text(
        "module.exports = {};\n",
        encoding="utf-8",
    )

    result = resolve_js_ts_import(workspace, "src/app.ts", "react")

    assert result.resolved_path is None
    assert result.resolution_kind == "external"
    assert result.reason == "external_package"
    assert result.candidates == ()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
