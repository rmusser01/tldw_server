# UAT225 original native acceptance

Root observed the repaired administrator reads on committed e1ccad4be7cf5b5c1b4c1e3b7405741a3ad19f0d. All3669 backend hashes match before restart and after native observation. Owned API89545 received graceful SIGTERM; replacement10113 started11:32:22UTC and health200 at11:32:32UTC. The existing profile/data were preserved.

Normal Settings logout and private normal-UI login yielded auth/me200 for administrator1/uat_admin at11:35:24UTC. The first login-helper invocation was on the still-open Flashcards page and timed out waiting for a Username field; no credentials were entered there. Recovery used the visible Settings/Edit server/Logout path, then the actual login form. This was a harness precondition error, preserved in login-harness-note.json. No auth token/header/storage extraction or seeding.

Opened the existing note from the Notes list and Graph/Suggestions through visible controls. At11:37:16UTC original Graph, pending/accepting list, capabilities and active-run list all returned200. Original09:16 two503 responses remain in the separately committed original failure packet. The note is unchanged version1, title/content/timestamps/owner intact, with tag14version1. Graph still authorizes suggestions and renders its tag relationship.

Normal browser reload at11:38:30UTC returned the Notes list. Reopened the same note and Graph through visible UI. Fresh Graph/list/capability/run responses at11:39:34UTC all returned200, with the same source fingerprint and truthful unavailable reason. This is fresh request evidence after reopening, not a claim that Graph view selection persisted through reload.

The Suggestions panel discloses llama.cpp/Gemma and says “The suggestion worker is unavailable.” Generate is disabled and no suggestions are ready. No worker/provider setting was changed; no generation, inference, acceptance, note save or new product mutation was performed. Actual local lifecycle/decision/retirement acceptance is separately covered by the independent173+200 real-database tests with synthetic provider replies. Native generation is not claimed.

The safe observer recorded no pageerror and no failed response among its authorized Notes/Graph/Study/Chat domains. Console evidence separately preserves two Settings /openapi.json404 probes under read-only classification, normal logout warnings, development HMR messages and the library warning for configured graph wheel sensitivity. Those entries are not hidden or counted as Graph failures. No defective zoom interaction was observed.

This reused native PostgreSQL profile has the previously recorded privileged/BYPASSRLS runtime qualification. Fresh state, restricted-role multi-user coverage and all48 full-matrix rows remain separate and unstarted. Independent evidence audit pending.
