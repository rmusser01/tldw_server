# VN_Play

VN_Play implements visual novel runtime behavior: save slots, scripted and generated turns, scene state, branch navigation, gates, generated outputs, setup options, and asset resolution. It connects VN scripts, VN assets, VN policy, provider adapters, and `VNPlay_DB` persistence behind the `/vn/play` API.

## Start Here

- `service.py` is the main play runtime service.
- `script_runtime.py`, `parser.py`, and `state.py` handle scripted scene parsing and runtime state.
- `adapters.py` defines provider adapter boundaries for turn and script generation.
- `gates.py`, `branch_navigation.py`, and `generated_outputs.py` implement runtime controls around choices and generated content.
- Related API surface: `app/api/v1/endpoints/vn_play.py`.
- Related tests: `tests/VN_Play/`.

## Responsibilities

- Manage VN play sessions, save slots, turns, and action requests.
- Parse and run scripted VN scenes alongside generated turn flows.
- Resolve setup options, script snapshots, asset packs, and available runtime assets.
- Evaluate gates and branch navigation for choices and scene transitions.
- Persist generated outputs and runtime state to `VNPlay_DB`.
- Use policy and provider adapters for generated character/scene behavior.

## Module Map

- `service.py` - primary runtime orchestration service.
- `models.py` - runtime domain models.
- `adapters.py` - provider adapter protocols and implementations.
- `parser.py` - VN script parsing helpers.
- `script_runtime.py` - scripted scene runtime helpers.
- `state.py` - state derivation and persistence helpers.
- `gates.py` - runtime gate evaluation.
- `branch_navigation.py` - choice and branch navigation helpers.
- `generated_outputs.py` - generated-output persistence helpers.
- `setup_options.py` - setup option resolution.
- `assets.py` - runtime asset resolution.
- `constants.py` and `errors.py` - shared constants and error types.

## How It Connects

- `app/api/v1/endpoints/vn_play.py` exposes play, save-slot, turn, action, and setup routes.
- `app/api/v1/schemas/vn_play_schemas.py` defines play API contracts.
- `app/core/DB_Management/VNPlay_DB.py` stores runtime state.
- `app/core/VN_Scripts/` supplies published script snapshots and validation context.
- `app/core/VN_Assets/` supplies asset packs and manifests.
- `app/core/VN_Policy/` supplies policy profiles and safety definitions.
- LLM provider adapters are used for generated turns and generated script output.

## Extension Points

- For a new runtime action, update `models.py`, `service.py`, schemas, and action-request tests.
- For script syntax or parsing changes, update `parser.py`, `script_runtime.py`, and VN script/play tests.
- For branching or gate behavior, start in `branch_navigation.py` and `gates.py`.
- For provider changes, update `adapters.py` and keep tests isolated from external providers.

## Testing

- `tests/VN_Play/test_vn_play_api.py`
- `tests/VN_Play/test_vn_play_turns.py`
- `tests/VN_Play/test_vn_play_action_requests.py`
- `tests/VN_Play/test_vn_play_save_slots.py`
- `tests/VN_Play/test_vn_play_state.py`
- `tests/VN_Play/test_vn_play_gates.py`
- `tests/VN_Play/test_vn_play_branch_navigation.py`
- `tests/VN_Play/test_vn_play_scripted_generation_runtime.py`
- `tests/VN_Play/test_vn_play_generated_outputs.py`
- `tests/VN_Play/test_vn_play_assets.py`
- `tests/VN_Play/test_vn_play_db.py`

## Gotchas

- Runtime behavior spans DB state, script snapshots, asset readiness, and policy profiles; tests should set up all required adjacent state explicitly.
- Scripted and generated flows share service boundaries but have different invariants around parser output and provider output.
- Branch and gate changes can alter save-slot replay behavior.
