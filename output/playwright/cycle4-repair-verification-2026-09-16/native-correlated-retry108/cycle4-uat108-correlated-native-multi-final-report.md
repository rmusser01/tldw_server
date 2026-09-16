# Correlated native UAT108 — core PASS; optional Note backlink FAIL

## Runtime and scope

Commit4bd4e2dfda, preserved multi Alice session. Parent intentionally restarted isolated API18301 PID22744 and UI18381 PID22787. Normal navigation loaded current source after readiness. No profile reset, config/product edits, mocks, API seeding, or old-data cleanup. New ordinary saved conversation only. Sole Gemma9099 lease held during Retry and released immediately after terminal answer; later regrant belongs to separate117test.

## Actual failure → Retry → canonical reload

Synthetic request: Cycle4 correlated Retry108 20260916_0526: reply exactly CEDAR RECOVERED.

Conversation c244a779-afcd-4f3d-8d89-a63065971a9b. Initial existing Ollama/gemma3:1b request returned502 with exactly one user and metadata.tldw_client_message_id=pa_eeb3-27ac-b5b-513c. Canonical GET200 before Retry (normal canonical URL open followed by normalreload) contains system8f69af6a-30fb-47a5-a46d-b4d677842eb5 and exactly one user e69bfae5-f386-4b8e-8616-eeeefdcbdd06 with stored metadata.client_message_id matching the request. UI after reload shows one user and a local displayerror.

Selected exact configured LLaMa.cpp ../../../Working/Language_Models/gemma-4-26B-A4B/gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf; clicked actual Retry chat once. Actual200 request contains one original user, metadata.tldw_retry_failed_turn=true and same client correlation. No displayerrorJSON appears in model messages. Finalanswer CEDAR RECOVERED.

Final normalreload GET200 retains exactly three canonical rows: same system, same original user, and assistant1b4b6005-f5fb-439a-9a2b-2bf3e5b826f5 with CEDAR RECOVERED. The prior local-only error remains visible afterreload (fourUIrows vs threecanonical); this is retained evidence, not a duplicateuser or claimed clean UI. Screenshot final.png visually inspected.

Evidence: negative-requests.txt, pre-retry-open.txt, pre-retry-reload.txt, pre-retry-ui.txt, retry-requests.txt, retry-ui.txt, final-reload.txt, final-ui.txt, final.png. Filenames share this report's prefix. Captures use real browser response events without credentials or response substitution. The response-event collection includes multiple canonical reads during startup/reloads; counts alone are not a separate performance assertion.

## Optional Note/backlink

Saved successful assistant using actual Save to Notes. New note1eea11bb-c31d-448e-a83d-74004f7d540e appears in Notes list and editor, content CEDAR RECOVERED, Origin Saved from Chat, saved version1, correct conversation and canonical assistant metadata. No duplicated save attempted. The narrow notePOST observer returned no entries; creation proof is actual subsequently loaded Notes UI, not a claimed captured POST status.

FAIL: Notes More actions → Open conversation was clicked twice with settled snapshots. Both close the menu but remain http://127.0.0.1:18381/notes. No new browser tab appears. Finalconsole capture has0errors/0warnings. This optional source-navigation failure is reported separately from coreRetry pass; no source repair attempted. Evidence note-open.txt, note-actions.txt, backlink.txt, backlink-settled.txt, backlink-repeat.txt, backlink-final-state.txt, console.txt. Clicking the informational linked-conversation text before using the menu did not navigate and is not counted as the explicit action failure.

A first click on the Chat hover overflow timed out when the hover toolbar replaced it; using the fresh visible toolbar ref succeeded. An unsupported network CLI command produced only usage; no product inference from that tooling error. Existing data preserved; new data consists of this synthetic conversation/rows and oneNote. No native image/reasoning-only coverage or full UAT is claimed in this unit.
