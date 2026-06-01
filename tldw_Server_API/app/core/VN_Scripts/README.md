# VN_Scripts

VN_Scripts manages visual novel script authoring, validation, templates, published snapshots, playtest support, authoring catalogs, graph views, and snippet patching. It is the authoring counterpart to `VN_Play`, with persistence in `VNScripts_DB` and API exposure through `/vn/scripts`.

## Start Here

- `service.py` is the primary VN script authoring and persistence service.
- `validator.py` validates script structure and references.
- `models.py` defines script, scene, choice, validation, and authoring models.
- `playtest.py` prepares script playtest flows through VN play runtime helpers.
- Related API surface: `app/api/v1/endpoints/vn_scripts.py`.
- Related tests: `tests/VN_Scripts/`.

## Responsibilities

- Create, update, validate, publish, and read VN scripts.
- Produce and store published script snapshots for runtime use.
- Validate scenes, choices, branches, assets, policy references, and script structure.
- Provide authoring catalogs, graph views, templates, and snippet patch helpers.
- Support playtest preparation before a script is used by VN play.
- Surface authoring-specific errors and validation details to API callers.

## Module Map

- `service.py` - script authoring service and snapshot orchestration.
- `models.py` - script and authoring domain models.
- `validator.py` - structural and reference validation.
- `templates.py` - starter script templates.
- `playtest.py` - playtest setup helpers.
- `authoring_catalog.py` - catalog data for authoring UIs.
- `authoring_graph.py` - graph view of script structure.
- `snippet_patcher.py` - targeted script snippet updates.
- `authoring_errors.py` - authoring-specific error types.

## How It Connects

- `app/api/v1/endpoints/vn_scripts.py` exposes script authoring, validation, publish, playtest, graph, catalog, and snippet routes.
- `app/api/v1/schemas/vn_script_schemas.py` defines API contracts.
- `app/core/DB_Management/VNScripts_DB.py` stores scripts and published snapshots.
- `app/core/VN_Play/` consumes published script snapshots and playtest setup.
- `app/core/VN_Assets/` and `app/core/VN_Policy/` provide asset and policy references used during validation.

## Extension Points

- For a new script field or node type, update `models.py`, `validator.py`, schemas, DB persistence, and validator tests.
- For authoring UI support data, start in `authoring_catalog.py` and `authoring_graph.py`.
- For snippet behavior, update `snippet_patcher.py` and API tests that cover targeted updates.
- For playtest changes, inspect `playtest.py` and VN play runtime tests together.

## Testing

- `tests/VN_Scripts/test_vn_scripts_api.py`
- `tests/VN_Scripts/test_vn_scripts_db.py`
- `tests/VN_Scripts/test_vn_script_validator.py`
- `tests/VN_Scripts/test_vn_script_publish_snapshots.py`
- `tests/VN_Scripts/test_vn_script_authoring_catalog.py`
- `tests/VN_Scripts/test_vn_script_authoring_graph.py`
- `tests/VN_Scripts/test_vn_script_playtest.py`

## Gotchas

- Published snapshots are runtime inputs for VN play; do not mutate them through draft-authoring paths.
- Validation should run before publish and playtest flows so runtime code does not receive incomplete scene graphs.
- Script, asset, and policy references cross modules, so tests should cover missing or stale references explicitly.
