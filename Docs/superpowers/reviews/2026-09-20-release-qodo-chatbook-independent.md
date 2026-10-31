# Independent Chatbook compatibility review

Reviewed read-only in `/private/tmp/tldw-chatbook-release-compat`, cross-checked against server schemas and endpoints in `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/release-main-0.1.42`.

## Finding: clone key lifetime is not retained through the panel

At `tldw_chatbook/UI/Sharing_Panel.py:484-492`, the clone action supplies only share ID and name. `tldw_chatbook/Sharing/server_sharing_service.py:103` and `tldw_chatbook/Sharing_Interop/server_sharing_service.py:250` construct a new `CloneWorkspaceRequest` for every service call. Its key default factory generates a fresh UUID. Thus a server-accepted clone whose reply is lost followed by the user's panel retry sends a different admission key and can create another workspace. The direct-client replay test retains its request object and therefore does not cover this real panel/service path.

Reported to qodo_frontend and parent. qodo_frontend confirmed and is implementing panel-owned key retention through uncertain responses, explicit reset for new logical clones, and a mounted panel→scope→service→HTTP timeout/replay regression.

## Other checks

- Clone request serialization uses canonical `name`, excluding `idempotency_key` from JSON and supplying the header; the server forbids extra request-body fields and requires that header.
- Operation identity, queued/running progress, publication/result/readiness details, failure/error details, and recipient-owned polling route agree with current server schema and endpoint. Poll URLs are locally constructed from validated operation UUIDs rather than dispatched from arbitrary server-provided URLs.
- Canonical source `source_id` and `origin_url` fields are parsed with legacy aliases and compatibility accessors. The explicit page API preserves pagination, summary, and partial_errors. The list convenience traverses pages and rejects nonadvancing pagination. Both service families retain page metadata and source identity.
- Notes deletion forwards caller-selected dataset_id, expected_version, idempotency_key, and reason through both wrappers into the exact server query parameters. Missing/stale version errors stay visible; no read-latest fallback bypasses optimistic preconditions.

No additional actionable issue found in these areas. Key-lifetime finding remains open pending panel fix verification. No repository edits.

## Panel key-retention follow-up

The panel now owns clone keys in app-lifetime state and passes them through the existing scope/service wrappers. Same-input retries and panel remounts retain the key. The map preserves earlier intents when switching A/B/A input values, and neither successful nor failed receipts silently create a new admission identity. An explicit `Start another clone` action clears only the selected intent before the next admission. This closes the originally reported lost-response duplicate-clone path by inspection.

A small additional issue was reported: the new synchronous reset button parsed a free-text share ID without catching ValueError, allowing invalid input to escape Textual's event handler. qodo_frontend is adding containment and the parent's requested server/principal identity scoping before final independent test execution. Final verification pending those changes.

Independent mounted verification of the panel-owned lifetime repair:

`source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate && PYTHONPATH="$PWD" python -m pytest Tests/Sharing/test_clone_panel_replay.py -q --tb=short --basetemp=/private/tmp/qodo-chatbook-reviewer-panel`

**1 passed**. Log: `/tmp/qodo-chatbook-reviewer-panel.log`. This exercises the real panel through real scope/service wrappers and HTTP serialization: timeout after admission, panel removal/remount, retry with identical key, then explicit new-copy intent with a different key.

The originally reported lost-response duplicate-clone finding is closed. Additional identity-scoping and invalid-ID containment follow-up remains pending final review.

## Final closeout — identity scope and reset handling verified

The final panel key identity is `(active_server_id, base_url, authenticated_authority_id, share_id, normalized_name)`. The existing RuntimeServerContextProvider authority resolver derives the stable authenticated user identity (rather than storing or keying directly on changing credentials); its capture/current guard rejects account or server changes during identity lookup. The production app supplies this provider. Canonical name normalization uses `" ".join(name.split())`, matching server `core/Sharing/shared_workspace_clone_operations.py:normalize_clone_name`, with the same 255-character bound.

The retained-key limit rejects new admissions while preserving existing keys and allowing their retries. It performs no eviction. Explicit reset resolves the selected current identity and pops only that exact server/account/share/name entry. Outcomes do not implicitly reset keys. Reset stops event propagation, and both actions catch invalid share IDs and ServerContextError without mutating stored identities or sending a clone.

Fresh independent verification:

`source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate && PYTHONPATH="$PWD" python -m pytest Tests/Sharing/test_clone_panel_replay.py -q --tb=short --basetemp=/private/tmp/qodo-chatbook-reviewer-panel-final`

**2 passed**. Log: `/tmp/qodo-chatbook-reviewer-panel-final.log`. Cases exercise lost admission response/remount/replay, explicit new intent, canonical whitespace, account isolation, selected-context reset, preserved prior-account retries, server separation under the retention quota, no quota eviction, and invalid-ID reset containment.

**Final disposition: original key-lifetime finding and follow-up reset error path are closed. No remaining actionable issue found in reviewed compatibility scope.** App-lifetime retention intentionally does not persist across a full application restart; callers needing restart-persistent uncertain-request recovery must persist their request/receipt lifecycle, as documented by the compatibility ADR. No repository edits by reviewer.
