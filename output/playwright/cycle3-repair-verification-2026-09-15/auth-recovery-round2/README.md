# Cycle 3 targeted auth recovery checks — 2026-09-15 UTC

These observations extend the separate targeted repair evidence. They do not alter the frozen full UAT at d40e17dc81 or constitute another full fresh run.

## Results

- **UAT087 natural refresh:** Bob's access token expired naturally at16:22:26. Refresh succeeded at16:22:28, producing an effective expiry of16:52:28. Both existing tabs kept notifications active; subsequent status-only observations show notification200 in both. No token timestamps, browser clock or responses were changed.
- **UAT080 terminal control:** only the current isolated session was revoked at16:24:15. Notes moved to login and Settings showed Login Required. Private request counts remained37/36 from16:25:11 to16:26:22. Normal UI login restored access.
- **UAT088 Settings:** the same mounted Settings form became Logged In after normal login in the other tab at16:26:36. Cancelling a proposed auth-mode change also restored multi-user and Logged In at16:32:36.
- **UAT089 failure and repair:** despite form recovery, the old app owner left the Settings header absent. After the repair, another owned-session revocation at16:44:50 and normal UI login at16:46:53 restored Logged In, Companion Home and active notifications in that existing Settings tab without a reload during the control. Fresh subsequent notification requests returned200 in both tabs.
- **UAT090 confirmation:** the first mode-change confirmation emitted the static AntD Modal context warning. The context-backed dialog opened and cancelled at16:42–16:43 without that warning and retained active login. A browser advisory about missing login autofill hints was also corrected; the later signed-out Settings metadata records username/current-password attributes.

Reviewed repair commits:65b8914dbd (optional browser guard),a3542ea385 (Settings),af725e4330 (app shell). Notification transport was already at fb79cc565a. Frontend development updates were active between controls; there was no reload of Settings within either documented login-recovery control.

## Capture limits

JSON files contain only timestamps, route names, visible-state flags, input hint attributes and aggregate HTTP status counts. Browser performance buffers were explicitly cleared/resized for the settled observations; old historical counters are not interpreted as fresh failures. Snapshot text retains visible UI and CLI capture context. There are no request headers, request/response bodies, session IDs, tokens or credentials. Raw request capture was not used.

The Settings title is still empty (existing UAT058). All issue closure still requires the subsequent complete fresh single-user and multi-user workflow run.
