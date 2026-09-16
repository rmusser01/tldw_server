# Bounded native UAT108 recheck

TASK13260 / correction under task49. No full UAT or profile reset.

## Gate

Wait for root's explicit API18301 restart-ready confirmation and inference lease before browser requests or generation. Do not restart the runtime. Existing browser wrapper: node /private/tmp/cycle4-safe-browser.mjs multi. Private credential source remains unchanged and must never be printed.

## Exact checks

1. Inspect current Alice UI using preserved session and start an explicit NEW saved ordinary conversation. Choose General chat if active mode is Character; confirm no old transcript and Saved persistence.
2. Capture credential-free request/response metadata for this new page only. Select existing Ollama/gemma3:1b, send a unique synthetic user request, and establish actual provider502 with a freshly created canonical conversation ID.
3. Restore the exact configured LLaMa.cpp Gemma model: ../../../Working/Language_Models/gemma-4-26B-A4B/gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf. Click actual Retry chat once.
4. Verify Retry200, explicit retry metadata, original user request exactly once and no display-error JSON in model messages. Retain canonical ACK IDs from real JSON/SSE when present; no transport mocks or API seeding.
5. Capture actual GET canonical messages before and after ordinary reload via normal UI/network observations. If a separate UI history-open gesture is required for the pre-reload GET, record it honestly. Required shape: one original user row plus final assistant row, with existing system row allowed; same canonical user and assistant IDs retained after reload.
6. If practical, use actual Save-to-Note action for the final assistant, verify its note source link and return navigation to this exact canonical conversation/message. No extra inference.
7. Release inference lease immediately once the final Retry generation is terminal. Preserve both new and prior records, capture safe screenshots/snapshots, record every failure, scan new evidence for private credentials, report exact IDs/outcome/limits.

## Preserved prior failure

Conversation81b8e878-931c-436a-ac54-37f0bf38c1cd contains duplicate user rows7f16c00c-19b4-4add-9e2a-8b05ad5d3e18 and9be643bf-4a42-4f6f-951e-be2aded6a78f. Do not modify or coalesce it. Earlier evidence under output/playwright/cycle4-repair-verification-2026-09-16/native-multi remains immutable for this separate recheck.

## New evidence prefix

/private/tmp/cycle4-uat108-native-multi-

Use unique suffixes for plan, tracker, requests, snapshots, canonical rows, screenshots, console, final report and hash manifest. Product files/config/accounts remain unchanged except authorized new synthetic chat/note records.
