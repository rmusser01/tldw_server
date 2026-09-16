# UAT155 independent native acceptance audit

Verdict: the retained native evidence closes the Settings/Close acceptance gap for TASK-13260.93. The unsupported WebUI sidebar action remains visible but disabled, as approved.

- Independently inspected settings-disabled.txt: actual URL http://127.0.0.1:18580/settings/tldw, Server Settings page, and button "Switch to Sidebar" explicitly marked [disabled].
- Visually inspected settings-disabled.png: the button is dimmed, Close is available, Settings content renders, and the API key is masked. The textual receipt replaces the actual credential with [REDACTED]; the displayed demo-key help text is public product copy.
- Independently inspected closed.txt: the actual Close button click returns the current page to http://127.0.0.1:18580/admin/llamacpp, with the Llama.cpp Admin heading and populated page. The native return destination is Admin; it is not a claim about returning to a particular Chat.
- Source and mounted capability test hashes still exactly match the earlier independent review: fae250e03181e7d5d9abaf7a236faa9f7f2e6693145703c32557703abde9e284 (SettingsOptionLayout.tsx) and 4aae0c79221d873e67aa7621d7abf4817e4b4f09569b0166494533e1629535d0 (settings-layout-sidebar-capability.test.tsx).
- Earlier independently retained validation in ../followup155/ supplies 24/24 passing tests and committed-source replay with the one WebUI failure plus three passing extension/unsupported controls. These tests were not rerun for this artifact audit. Chrome/Firefox extension behavior and preference-write prevention remain mounted-test evidence, not newly performed native extension tests.

Limits: no forced click of the disabled button, no new native extension exercise, and no claim of a clean browser console; the Close receipt references new console entries whose content is outside this evidence set. No product/browser/runtime changes, inference, staging or commits were performed by this reviewer. Bandit is inapplicable to this evidence-only/TSX scope. Source-to-copy hashes and trailing-whitespace normalization are recorded in retention-manifest.json.
