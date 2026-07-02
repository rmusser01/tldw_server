## Stage 1: Provider Catalog Regression
**Goal**: Prove `/llm/providers` and `/llm/models/metadata` use the same env-resolved custom OpenAI endpoint/model as chat completions.
**Success Criteria**: A regression test fails on current `dev` when `CUSTOM_OPENAI_API_URL` and `CUSTOM_OPENAI_API_MODEL` are set.
**Tests**: Targeted pytest for `custom_openai_api` provider metadata.
**Status**: Complete

## Stage 2: Shared WebUI Auth Regression
**Goal**: Prove browser requests from the shared Tldw client can use the WebUI-provided single-user API key without manual localStorage seeding.
**Success Criteria**: A regression test fails on current `dev` and passes after the shared runtime auth source is wired.
**Tests**: Targeted Vitest for request auth or client config bootstrap.
**Status**: Complete

## Stage 3: Minimal Fixes
**Goal**: Make provider discovery and WebUI auth follow existing config/auth ownership boundaries.
**Success Criteria**: Regression tests pass without weakening model-readiness filtering.
**Tests**: Re-run targeted pytest and Vitest suites.
**Status**: Complete

## Stage 4: End-to-End Screenshot Flow
**Goal**: Restart the clean dev instance, import the character card, chat through llama.cpp, and capture screenshots.
**Success Criteria**: Screenshots show regular chat and character-card roleplay chat with real model responses.
**Tests**: Browser/Playwright visual validation against `http://127.0.0.1:8080/chat`.
**Status**: Complete

## Stage 5: Character Card Avatar Hydration
**Goal**: Ensure loaded character-card chats show the embedded card image instead of the generic assistant placeholder.
**Success Criteria**: Hydrating a selected character with `image_base64` produces an avatar data URL when `avatar_url` is absent.
**Tests**: Targeted Vitest for `characterToAssistantSelection()` embedded image handling plus browser DOM verification on the imported Miku card.
**Status**: Complete
