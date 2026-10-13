# Settings fixture contracts — TASK-13639

The numeric and model-prompt suites now declare the physical native storage API before constructing real SDK storage. The prompt suite uses actual Actor scene reads and a stable controller/account receipt against an actual fresh draft. Existing model catalog and Actor Save boundaries remain controlled I/O; the numeric settings hook and storage persistence remain real.

Account invalidation clears the actual draft address. Cache reopen cases create a fresh owned draft after that event. The stale-form case first requires Save to be absent, no Actor write, no close and no model prompt; only a new source and completed fresh query admit the later blank Save. Original prompt, numeric, cancellation, retry, scope and cache assertions remain.

An unchanged isolated cached reset measured5.059s against5s, while its companion measured4.829s. This real two-mount group receives the same10s allowance as adjacent generation cache tests; no global timeout changes. Representative initial3passed; full initial native61passed4failed; followup5passed. Final native qualification passes202tests across5suites. WebUI qualification passes249tests across4suites, including both repaired suites and actual Actor read/ownership services. Both normal client compilers pass; they do not provide a dedicated test-file type check.

Independent bounded review found no Critical or Important source issue. These component tests do not qualify actual durable Actor Save because its ACK remains mocked; connected real scene tests cover that separately. ADR required: no; restores existing test dependencies. Bandit N/A for TypeScript tests. Whole-PR review, remaining findings, final rebase and merge remain pending.

PR: https://github.com/rmusser01/tldw_server/pull/3229
