# TASK13260.166 — fresh-matrix browser wrapper

Reuse the installed Playwright CLI and the recovery wrapper's private raw evidence pattern. Select the initialized matrix profile by run/cell and use its recorded browser session and frozen source root. Redact known credentials (including JSON/URL encodings) and JWTs from displayed output and retained text snapshots. Keep raw evidence mode0600 and confined to the selected profile. Reject session overrides, invalid profile ownership, mismatched initialization and escaped snapshot paths. Preserve subprocess failure status; suppress partial output on subprocess infrastructure errors.

Tests use temporary synthetic profiles and an actual harmless Node subprocess, never a browser, application, database or model. Existing14 reviewed launcher files remain untouched. Full UAT and real browser use remain gated on225/226.

Stages: causal wrapper tests → bounded implementation → independent review → actual use only after the matrix gate. No full-matrix source, dependency, profile or browser creation is authorized by this preparation script itself.
