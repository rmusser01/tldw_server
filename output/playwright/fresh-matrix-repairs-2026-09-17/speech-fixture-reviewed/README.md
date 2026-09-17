# Reviewed speech timeout fixture correction

UAT249 is test-only. Public configuration persistence and the actual WebUI transport mock restore both original timeout assertions; all9original assertion lines remain unchanged. Independent9tests pass with no skips or new lint warnings. No production timeout or native TTS behavior changed. Final adjacent regression run waits for concurrent Character248 integration; the earlier adjacent RED is retained.
