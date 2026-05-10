# backlog-py

Python compatibility clone of Backlog.md.

This package is experimental. Do not put it on PATH as `backlog` and do not use
it to mutate live repository data until the cutover gates in the design spec
pass.

## Oracle Fixtures

Compatibility fixtures are pinned to explicit upstream Backlog.md release
metadata. The initial oracle manifest records `backlog.md@1.44.0`, source kind,
source reference, package metadata hash, generation date, and the agent-critical
commands/resources/tools that future golden fixtures must cover.

Upstream Backlog.md and its Node/Bun toolchain are only allowed in fixture
generation or refresh jobs. Normal `backlog-py` runtime, regular tests, and
future repository cutover paths must remain Node/Bun-free.
