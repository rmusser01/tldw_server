# UAT246 Character SSE delivery repair — frozen for review

Task: TASK13260.188. Base: 86458ab88ce3fa62e6518c9d813c3860254ddb2c. Exact three source/test paths and hashes: [source-freeze.json](source-freeze.json). No source/test edits after that freeze.

## Change and causal evidence

All three Character SSE response branches (legacy provider, unified provider, and text fallback) now share the endpoint-local headers `Cache-Control: no-cache, no-transform` and `X-Accel-Buffering: no`. The change prevents an intermediate compressor from buffering frames. Provider selection, authentication, timeouts, persistence, and transport orchestration are unchanged.

Actual targeted native observations proved gzip on the Next-proxied SSE response and delayed first body delivery, despite early headers. This narrows the previous UAT246 attribution; it does not prove every original failure had this cause. See the separate immutable stream246-native-audit report. Native acceptance of this patch remains pending.

## Tests and results

Exact reproducible commands are in [commands.json](commands.json).

- Backend causal RED: **12 failed, zero skips**, each actual response missing no-transform. Covers SQLite and required official PostgreSQL, legacy/unified, complete/disconnect/text fallback.
- Installed Next causal RED: **1 failed / 2 passed, zero skips**. With terminal deliberately held, the early gzip body was unavailable; identity and the explicit adverse buffering control passed. An earlier draft stopped at the encoding assertion before the body read; that receipt is retained separately in next-header-red.log. The corrected causal early-reader RED is next-red.log.
- Backend final GREEN: **35 passed, 5 warnings, zero skips**, official required-PG runner **exit 0**. Twelve header/stream controls plus 23 existing Character stream/persistence cases. The runner briefly remained alive after the pytest summary; its preserved handle subsequently exited normally, without termination or replacement. Final redacted log and receipt are retained.
- Installed Next final GREEN: **8 passed, zero skips, exit 0**. Existing five forwarding/status/cancellation/30-second controls remain. New gzip-negotiated no-transform and identity cases delivered the role frame in **3ms while terminal was still held**. Legacy gzip remained pending through **305ms**, then delivered exact content after terminal release. Every case releases its gate and reader in finally.

The controlled provider/HTTP fixtures exercise actual response construction, local adapter delivery, installed Next rewrite and compression. They are not new native model calls. Backend tests assert canonical user preservation, cleanup and one provider request where applicable. Fallback exercises actual SSE fallback serialization and emits DONE once.

## Static verification

- Ruff: **4 existing findings, zero added/removed**, compared with both original Python files using their real filenames.
- Installed ESLint/root config: **0 errors / 0 warnings**, baseline and current Next test. Existing root pages-directory configuration notice retained.
- Production Python Bandit: **0 findings / 0 errors, exit 0**. Additional all-touched scan reports test assert checks and cannot parse MJS as Python; it supplies no JavaScript security proof.
- Node syntax check and scoped git diff whitespace check: pass.
- No full compiler, native browser run, provider call, runtime change, held profile/database mutation, task edit, staging or commit by this unit.

## Review and limitations

Independent review requested from retry031_repair. Source freeze is the review target; parent owns native acceptance and integration. Installed-Next adverse observation is bounded to a held 300ms window rather than a claim about all compression configurations. The header contract prevents the demonstrated default Next compression behavior while retaining no-cache.
