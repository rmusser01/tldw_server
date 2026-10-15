# Current chat shell test contracts — TASK-13628

The quick prompt root fixture explicitly uses the WebUI storage hook alias and now lives in the frontend package. Its full behavioral body is unchanged except one relative import becoming the existing shared alias and one ownership comment. The original and relocated suites each pass23 cases.

The model-settings source guard now requires the existing stronger open-and-not-Scene-only catalog condition. The sidepanel adapter import guard accepts either quote style while retaining the exact imported component/type and module path. Scoped saving, provider handling, adapter projection and separate Quick Chat pop-out distinctions remain asserted. No production source or configuration was changed.

Final WebUI qualification passes71 tests across quick prompt, two shell guards and two initial setup suites. The shared native runner passes both guard suites9/9. Both normal client source type checks pass; these are not dedicated test-file type checks. Independent bounded review is clear. Normal hooks are recorded in the task.

ADR required: no; these align tests with current runtime/source contracts. Bandit N/A TypeScript test-only scope. Remaining whole-PR CI/UX/H2 work, final review and merge remain open. Remote dev has advanced and the PR is conflicting; four passing license/review-bot checks do not constitute the required CI gates.

PR: https://github.com/rmusser01/tldw_server/pull/3229
