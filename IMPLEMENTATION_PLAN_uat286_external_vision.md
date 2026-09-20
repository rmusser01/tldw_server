# UAT286: external llama.cpp image capability

Backlog: TASK-13260.223. Existing authorization: repair identified fresh-install UAT defects.

## Stage 1: reproduce and specify
**Goal**: Trace the real vision=true server response to the conservative catalog fallback.
**Success Criteria**: API regression fails because the exact configured model remains text-only.
**Tests**: Matching alias/path, unrelated model, malformed/unauthorized/unavailable/text-only controls.
**Status**: Complete

## Stage 2: bounded capability discovery
**Goal**: Read only configured llama.cpp /props using existing HTTP egress/auth patterns; match exact reported model identity.
**Success Criteria**: Coherent vision fields on both catalog endpoints; conservative fallback; short cache invalidated by endpoint/credentials and expiry; no redirected credential delivery.
**Tests**: Focused provider metadata tests, request-policy assertions, cache controls, Bandit, review.
**Status**: Complete

## Stage 3: real PostgreSQL acceptance
**Goal**: Send and retry the actual attached image through normal Chat on fresh committed source.
**Success Criteria**: Real provider response, identical image on retry, canonical reload, tracker updated, owned processes released.
**Tests**: Native browser and real llama.cpp9099 with official PostgreSQL fixture; source-integrity audit.
**Status**: In Progress
