# tldw-agent In tldw_server2

`tools/tldw-agent/` is the in-repo source of truth for the first-party agent code
used by the `vz_linux` helper/guest roadmap.

The migration starts from the upstream `tldw-agent` snapshot recorded in
[MIGRATION.md](./MIGRATION.md) and intentionally preserves existing behavior
before adding:

- `vz_linux` guest mode
- guest protocol handling
- helper/VM integration

Until those follow-up tasks land, treat this directory as a preserved source
import that must continue to build and pass its current tests.

## Isolated ACP runner configuration

The ACP executable accepts `--config /absolute/path/to/runner.yaml`. This selects an existing runner YAML file without changing HOME or the host's default agent settings. An explicit missing, unreadable or invalid YAML file fails startup. With no flag, the existing `~/.tldw-agent/config.yaml` default and optional-file behavior remain.

Point the server's ACP runner command at the built executable and pass the flag through `runner_args`. Workspace allowlists, agent setup, tool permissions and provider configuration still need their normal explicit setup; this flag does not certify execution readiness.
