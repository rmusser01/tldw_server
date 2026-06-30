---
id: TASK-10007
title: Harden Templating renderer review findings
status: Done
assignee: []
created_date: '2026-06-23 20:57'
updated_date: '2026-06-23 21:50'
labels:
  - templating
  - security
  - review-fix
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix validated review findings in `tldw_Server_API/app/core/Templating`: harden resource limits, preserve fail-safe renderer behavior for runtime errors, prevent arbitrary callable exposure through template context extras, make timezone defaults effective, and remove unused API surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Renderer returns the original template instead of raising for arithmetic and oversized range runtime failures
- [x] #2 Renderer rejects expensive constructs before allocating large outputs
- [x] #3 Templates cannot call arbitrary methods from `ctx.extra`; only approved helpers/facades remain callable
- [x] #4 `TemplateEnv.timezone` drives default date helpers
- [x] #5 Unused external-call option and unused render error class are removed
- [x] #6 Focused tests, Bandit, and diff checks are recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
- Add failing regression tests for runtime arithmetic failures, oversized range failures, expensive string multiplication, callable object exposure, timezone defaults, and removed API surface.
- Implement renderer validation and sandbox changes in `template_renderer.py`.
- Update consumers/tests for the removed `allow_external_calls` and `TemplateRenderError` API surface.
- Run focused templating/chat dictionary/chatbook tests, Bandit on the renderer, and whitespace diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added regression tests for renderer runtime arithmetic/range failures, expensive string multiplication, arbitrary method-call blocking, timezone defaults, and removed API surface.
- Added template AST call/operator guardrails in the renderer and dictionary validator.
- Added an explicit safe-callable sandbox and marked only renderer helpers, seeded random helpers, `user()`, and the regex match facade as callable from templates.
- Removed the no-op external-call option and unused render error class.
- Updated runtime/user docs to remove the stale external-call option and describe the stricter safe-call behavior.
- Added narrow Bandit suppressions for existing non-cryptographic random selection/helper paths.
- Addressed PR review comments by adding module/sandbox docstrings, marking touched unit tests, typing new test helper methods, narrowing arithmetic fallback handling to render-time only, and avoiding per-call safe-method set allocation.
<!-- SECTION:NOTES:END -->

## Verification

<!-- SECTION:VERIFICATION:BEGIN -->
- Red step confirmed: renderer tests initially failed for uncaught `ZeroDivisionError`, uncaught oversized `range` `OverflowError`, string multiplication rendering, arbitrary `ctx.extra` method calls, timezone default behavior, and stale API surface.
- Red step confirmed: dictionary validator tests initially failed to flag string multiplication and arbitrary method calls.
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat_NEW/unit/test_template_renderer.py tldw_Server_API/tests/Chat_NEW/unit/test_chat_dictionary_templates.py tldw_Server_API/tests/Chat_NEW/unit/test_dictionary_validator.py tldw_Server_API/tests/Chatbooks/test_chatbooks_template_mode_and_dict_strict.py -q` - 39 passed, 91 warnings.
- `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Templating/template_renderer.py tldw_Server_API/app/core/Chat/chat_dictionary.py tldw_Server_API/app/core/Chat/validate_dictionary.py -f json -o /tmp/bandit_templating_renderer_10007.json` - 0 results, 0 errors.
- `git diff --check -- tldw_Server_API/app/core/Templating/template_renderer.py tldw_Server_API/app/core/Chat/chat_dictionary.py tldw_Server_API/app/core/Chat/validate_dictionary.py tldw_Server_API/app/core/Templating/README.md Docs/User_Guides/WebUI_Extension/Chatbook_Tools_Getting_Started.md Docs/Published/User_Guides/WebUI_Extension/Chatbook_Tools_Getting_Started.md tldw_Server_API/tests/Chat_NEW/unit/test_template_renderer.py tldw_Server_API/tests/Chat_NEW/unit/test_dictionary_validator.py` - passed.
- `rg -n "[ \t]+$" IMPLEMENTATION_PLAN_templating_renderer_hardening_10007.md "backlog/tasks/task-10007 - Harden-Templating-renderer-review-findings.md"` - no matches.
- PR comment pass: `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat_NEW/unit/test_template_renderer.py tldw_Server_API/tests/Chat_NEW/unit/test_chat_dictionary_templates.py tldw_Server_API/tests/Chat_NEW/unit/test_dictionary_validator.py tldw_Server_API/tests/Chatbooks/test_chatbooks_template_mode_and_dict_strict.py -q` - 39 passed, 90 warnings.
- PR comment pass: `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Templating/template_renderer.py tldw_Server_API/app/core/Chat/chat_dictionary.py tldw_Server_API/app/core/Chat/validate_dictionary.py -f json -o /tmp/bandit_templating_renderer_10007_final.json` - 0 results, 0 errors.
<!-- SECTION:VERIFICATION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the Templating renderer by failing safely for runtime arithmetic/range errors, rejecting expensive operators and unapproved calls during validation, restricting callable access to explicit safe helpers/facades, applying `TemplateEnv.timezone` to default date helpers, and removing unused external-call/error-class API surface. Aligned dictionary validation and docs with the renderer behavior and verified the focused chat/chatbook template slice plus Bandit.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
