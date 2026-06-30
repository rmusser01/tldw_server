## Stage 1: Confirm Schema Mismatch
**Goal**: Reproduce the failing setup audio lifecycle tests and identify the exact request-model mismatch causing the failures.
**Success Criteria**: One or more targeted tests fail with evidence pointing to the incorrect request fields or model source.
**Tests**: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Setup/test_setup_audio_installer_lifecycle_api.py -k "audio_pack_import or audio_pack_export or admin_audio_provision or admin_audio_verify"`
**Status**: Complete

## Stage 2: Align Endpoint Request Models
**Goal**: Remove or replace stale endpoint-local setup request models so setup audio endpoints use the shared request schema definitions consistently.
**Success Criteria**: The affected endpoints accept the expected request shapes and the targeted setup tests pass.
**Tests**: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Setup/test_setup_audio_installer_lifecycle_api.py -k "audio_pack_import or audio_pack_export or admin_audio_provision or admin_audio_verify"`
**Status**: Complete

## Stage 3: Verify No Hang Regression
**Goal**: Re-run the previously hanging mixed test selection plus the full setup lifecycle module and confirm behavior.
**Success Criteria**: The mixed selection completes without hanging, and the setup lifecycle module no longer fails on the audio pack / TTS choice cases addressed here.
**Tests**: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py tldw_Server_API/tests/Setup/test_setup_audio_installer_lifecycle_api.py -k "test_generate_omnivoice_without_voice_normalizes_to_auto or test_streaming_pocket_tts_cpp_custom_voice_request"`; `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Setup/test_setup_audio_installer_lifecycle_api.py`
**Status**: Complete
