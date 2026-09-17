# Audit input discovery note

The first verification invocation stopped before inspecting events because its discovery assertion expected the initial 23 `alice211-*.txt` receipts. A new `alice211-editor-cancel.txt` receipt arrived while the read-only audit was running, yielding 24 files. The verifier now uses an explicit 24-file allowlist, including that ordinary editor-cancel receipt. This was an audit harness inventory mismatch, not a product failure or altered native result.
