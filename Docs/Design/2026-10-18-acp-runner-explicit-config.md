# Explicit configuration for an isolated ACP runner

Task: TASK-13260.278.18.83.45. C04 needs a real installed agent with a public disposable workspace and existing local provider. The Go runner currently always loads the host HOME configuration. Rewriting HOME or changing unrelated host agent settings is outside this workflow.

Add only `--config PATH` to the ACP executable. Reuse the existing single-read YAML loader; an explicit empty path, missing or invalid file must fail startup instead of silently choosing default agent settings. Omitting the flag preserves the existing default and fallback behavior. Keep workspace roots, tool approvals, credentials and server security policies unchanged. Do not add a new loader, dependency or environment reconstruction.

Test explicit selection, missing/invalid files and legacy default with ordinary temporary unit fixtures. Build/test offline with installed Go, then use the real first-party runner and installed OpenCode1.15.7/local llama.cpp8010 in the supported C04 workflow. Real task/result/cancellation/ownership gates remain separate.

ADR assessment: no new durable architecture rule; this exposes existing LoadFrom at one executable entrypoint and preserves governing ADR007/012/020 workspace/resource/ownership decisions.
