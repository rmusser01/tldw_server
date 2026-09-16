# UAT141 native timestamp acceptance

TASK-13260.80. Independent native audit passes cold failure with no invented last-success label, successful inbox read followed by later failure retaining the real age, and final controlled recovery. See independent-review.md for timestamps and limits.

Source: parent records HEAD ec28c34b7c plus separately frozen UAT164 guidance changes; not a clean full-HEAD run. Notification route SHA256 2c66d004e46d98c87523e57b35b3ff13577c75c15f142bb724540914f56262a9 matches its prior reviewed freeze. The fault control aborts actual notification requests before navigation; it does not invent successful payloads or mutate the product clock. Observer timestamps follow body reads and support minute-level freshness, not exact internal-state timestamps. Stream200 is recorded without raw SSE delivery claims. This is a notification-only outage, not whole-server UAT or separate138/139 acceptance.

## Retention integrity

Evidence is copied through an explicit allowlist. Original .tmp files are preserved. Only trailing spaces/tabs at text line ends and extra blank lines at end of file are normalized to one final newline; screenshots remain byte-exact. Audit tables quoting original hashes remain unchanged; evidence-manifest.json records both original and retained hashes. secret-scan.json records a zero-match scan against the private runtime API keys, JWT/hash secrets, account passwords, vision key and PostgreSQL passwords, plus JWT/private-key patterns. Exact-byte screenshot scans do not claim OCR secret detection; retained screenshots were visually inspected in the native audits.
