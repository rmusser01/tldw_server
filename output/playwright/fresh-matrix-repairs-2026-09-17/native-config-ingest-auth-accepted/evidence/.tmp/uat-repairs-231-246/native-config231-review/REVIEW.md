# UAT231 native supplement

**Partial native evidence; AC3's original generic-configuration branch remains unverified.** Task TASK13260.173. This supplement is separate from UAT237 and does not change its verdict.

The retained `native-preparation/mcp251-console.txt` captures a fresh PostgreSQL single-user WebUI reload on the `mcp251-fresh-targeted-20260917` profile (API18704/Web18784), prepared from `a7d3155a567afb25982eb360ea24b973cc3249c9`. It reports **3 total messages, 0 errors, 1 warning**. The warning correctly directs the user to **Settings → tldw server** to configure a missing API key. These counts describe this capture only, not all startup messages or later wizard activity.

However, this is the separate `getMissingApiKeyMessage()` branch. The original frozen SQLite observation contained that same accurate missing-key warning **plus three** generic warnings saying the WebUI user should open Settings **“in the extension.”** The repair changes the generic `ensureConfigForRequest` branch, which runs when no configuration/server URL is available and neither hosted mode nor a runtime key supplies an alternative. The new capture contains **zero** generic “tldw server is not configured” warnings. It therefore cannot establish that the repaired generic branch ran natively, or explain why the earlier three warnings no longer appeared.

The corrected branch and focused test remain present in the prepared source. The test verifies warning and rejection text for HTTP/HTTPS and Chrome/Firefox extension surfaces, plus warning-free configured unauthenticated discovery. The prior independent Chat review supports AC1/AC2:327 focused tests passed;103 adjacent tests passed and2 speech-fixture failures reproduced against baseline. Those tests and static checks were not rerun for this supplement.

## Required remaining acceptance evidence

Retain a native observation that actually reaches the generic missing-configuration path on the repaired WebUI source, with the real browser's configuration state and request trigger documented. Verify the emitted text points to Settings → tldw server without extension wording, while the operation still rejects missing configuration. Do not manufacture the console warning or treat the already-correct missing-key branch as its substitute. Preserve the captured warning counts and the original-vs-new branch distinction in the task record.

The existing native MCP251 review independently binds the fresh profile, restricted official PostgreSQL fixture, source, initialization, dependencies and process launch. This supplement references that reviewed provenance and hashes its packet, the console observations, the copied client/test source and preparation manifest. It does not repeat live process verification, inspect credentials, modify storage, or rerun the wizard. Native PostgreSQL startup observations do not constitute an original SQLite native rerun, clean-OS installation, or full-matrix acceptance.
