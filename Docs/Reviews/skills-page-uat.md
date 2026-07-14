# Skills Page UAT Checklist

Manual checklist for validating `/skills` in the WebUI and browser extension after automated Playwright coverage passes.

## Setup

- Use a current `dev` build of `tldw_server` with the Skills API enabled.
- Test both WebUI desktop and extension-width layout when possible.
- Authenticate in single-user mode with a valid API key.
- Start with one clean profile that has no custom skills, then one seeded profile with at least 25 skills.
- Keep browser devtools open for console and network evidence.

## Manual Scenarios

| ID | Scenario | Coverage | Pass criteria | Evidence |
| --- | --- | --- | --- | --- |
| SK-BEG-01 | First visit with no skills | Beginner orientation | Empty state explains what skills are, offers Seed Built-ins, Import, and New Skill, and does not show a dead table. | Screenshot of empty state. |
| SK-BEG-02 | Seed built-ins | Beginner first success | Seed action completes once, shows success feedback, lists the seeded skill, and survives refresh. | Screenshot before/after refresh. |
| SK-BEG-03 | Copy invocation | Beginner confidence | Copy `/skill summarize` reports success and pasted clipboard content matches the invocation. | Clipboard paste into scratch field. |
| SK-BEG-04 | Test run render-only | Beginner safe trial | Test run accepts arguments, render-only returns a prompt preview, and no destructive action occurs. | Dialog screenshot with rendered prompt. |
| SK-BEG-05 | Create then cancel | Beginner error avoidance | New Skill opens with labeled fields; Cancel closes without creating a draft row or dirty state warning. | Screenshot plus no new row. |
| SK-PWR-01 | Search large library | Power-user retrieval | A skill outside page one is found by name/description within one query and the request includes `q`. | Network request and row screenshot. |
| SK-PWR-02 | Filter by mode/tools/model | Power-user narrowing | Filters can be combined, visible chips/buttons reflect active filters, and results match the selected constraints. | Screenshot and request params. |
| SK-PWR-03 | Sort and pagination | Power-user predictability | Name/mode sorting changes request params and visible order; pagination keeps active filters. | Network request sequence. |
| SK-PWR-04 | Bulk delete preflight | Power-user control | Selecting multiple skills shows selected count; Delete selected opens confirmation and does not delete until confirmed. | Dialog screenshot. |
| SK-PWR-05 | Dense/compact table | Power-user scanning | Compact density shows more rows without clipping action buttons, names, or badges. | Desktop and extension-width screenshots. |
| SK-A11Y-01 | Keyboard navigation | Accessibility | Tab reaches search, filters, row actions, dialogs, and confirmation controls in logical order; Escape returns focus to opener. | Notes from keyboard-only pass. |
| SK-A11Y-02 | Labels and announcements | Accessibility | Search, filters, checkboxes, dialogs, loading, errors, and success states have accessible names or live-region feedback. | Accessibility tree or Playwright snapshot. |
| SK-A11Y-03 | Contrast and focus | Accessibility | Text and focus rings remain visible in light/dark themes, including destructive buttons and disabled states. | Screenshots in both themes. |
| SK-RSP-01 | Extension width | Responsive | At 390 px width, primary actions remain reachable, dialogs fit viewport, and button text does not clip. | Extension-width screenshot. |
| SK-FAIL-01 | Skills unsupported | Failure recovery | Server without Skills API shows Skills-specific unavailable copy and Refresh capabilities, not a generic crash. | Mock/older-server screenshot. |
| SK-FAIL-02 | Import validation failure | Failure recovery | Invalid `SKILL.md` preview lists validation errors and import is not submitted. | Dialog screenshot and network evidence. |
| SK-FAIL-03 | Execution failure | Failure recovery | Test run API failure shows the backend detail and keeps the dialog open for correction/retry. | Dialog alert screenshot. |
| SK-FAIL-04 | Stale delete version | Failure recovery | Conflict response explains the skill changed elsewhere and tells the user to reload before retrying. | Error toast/callout screenshot. |
| SK-FAIL-05 | Slow list loading | Feedback | Loading state is announced until the list resolves and duplicate risky actions are unavailable or harmless. | Loading screenshot and network timing. |

## Success Metrics

These are manual/product-analysis measures for release validation. This task does not add telemetry.

| Metric | Target | How to measure |
| --- | --- | --- |
| Beginner task completion rate | 90% complete seed, copy invocation, and render-only test run without help. | Moderated UAT or scripted manual run. |
| Time to first successful skill use | Median under 2 minutes from first `/skills` visit to rendered test prompt. | Stopwatch during beginner scenario. |
| Error rate | No uncaught console errors and no failed API calls outside intentionally mocked failure cases. | Devtools console and network log. |
| Search/filter success | 95% of large-library lookup attempts find the target skill in one query/filter sequence. | Power-user scenario log. |
| Configuration recovery success | 90% recover from unsupported, stale delete, invalid import, or execution failure using visible guidance. | Failure scenario checklist. |
| User confidence rating | Average 4/5 after beginner flow and 4/5 after power-user flow. | One-question post-task rating. |

## Release Gate

- Automated mocked Skills UAT passes in Playwright.
- Manual scenarios above have pass/fail evidence attached to the PR or release checklist.
- Any failed High-risk scenario has an issue linked before release.
