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
