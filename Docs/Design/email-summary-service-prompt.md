# Synchronous email summarization Service Prompt

Approved scope: TASK-13178. Add `media.email.summarization`, one literal
system-instructions part, to the existing registry and shared Settings.

The authenticated `/media/process-emails` adapter resolves one owner-specific
prompt before saving uploads, only when analysis and a provider are enabled and
no explicit system prompt was supplied. Preserve explicit empty multipart text
in the form dependency before validation. Reuse the resolved string across EML,
ZIP, MBOX and enabled PST/OST processing and recursive summary passes. Close
the prompt connection on its lookup worker, including failures. Invalid saved
overrides fail closed before input processing. Unset/reset use deployment
summary defaults through the existing resolver.

Repair the form's missing `api_provider`, legacy `api_name`, and
`summarize_recursively` fields. Canonical provider wins, matching EPUB/PDF.
Remove the email processor's explicit-key requirement: the shared analyzer
already resolves configured credentials and supports keyless providers. Keep
analysis opt-in, user prompts, chunking, container feature flags, and nested
attachments' no-analysis policy. Direct core callers remain DB-free; enabling
their explicitly requested analysis without a supplied key is intentional.
Queued/persisted prompt lookup is outside this slice.

No new storage, endpoint, settings subsystem, credential resolver or abstraction.
Test real HTTP/form handling, owner databases, email/container extraction and
recursive analysis, replacing only the external model adapter. Cover prompt
precedence, default/reset, owner isolation, batch snapshots, no-read paths,
invalid overrides, provider wiring and nested attachments. Extend generic API
and shared Settings tests; verify OpenAPI, Ruff, compilation and Bandit.

Baseline: 117 backend tests passed, 2 existing skips (119 collected).

## Verification and review

The initial test-first run produced 14 expected email failures, 4 expected
registry/API failures, and 1 expected Settings failure. All were addressed by
the approved changes. Self-review also reproduced loss of JSON-shaped bodies
(object, list and string) in 3 failing tests. Passing the existing
`input_is_literal_text=True` option preserves the already-parsed body; all 3
regressions passed afterward.

Independent review found no blocking defects. Its three coverage suggestions
are implemented: real configured/keyless credential resolution with only the
external chat adapter replaced, enabled PST/OST traversal with libpff replaced,
and real lookup/close calls observed on the same worker for success/corruption.
All 6 targeted checks passed; follow-up review found no remaining issues.

- Final combined backend: 200 passed, 2 existing PST skips, including all 25
  email prompt tests, adjacent media prompt regressions, email parser/routes,
  generic Service Prompts and analyzer credential/adapter tests.
- Shared frontend: 197 Settings/service/domain tests passed.
- WebUI configuration: 5 targeted Settings tests passed.
- Official OpenAPI export, type generation and fingerprint verification passed.
  Removing only the three newly exposed email form fields reproduces the base
  fingerprint exactly; path/schema counts remain unchanged.
- Compilation passed. Bandit found zero issues in all four touched production
  Python files (`/tmp/bandit_email_service_prompt.json`).
- Ruff lint/format passed for the adapter, form, registry and touched tests.
  The existing email library has unchanged I001 and SIM103 findings, verified
  against base `dev`; its two-line analysis correction adds no findings.

Full repository tests, full frontend typechecking, live browser/provider runs,
and real libpff binary integration were not performed. The two existing PST
integration skips require libpff and/or a supplied real fixture. New container
tests exercise application traversal/conversion, not libpff's binary parser.

## PR #2887 review follow-up

Rebased cleanly onto `dc0b7455f2` (documentation-only changes); range-diff
confirmed the implementation commits were unchanged. Qodo reported zero bugs
and four rule findings. Added the endpoint's `JSONResponse` return annotation
and documented endpoint inputs/results/errors and form provider precedence,
recursive behavior, explicit prompt presence, validation and return type.
Regenerated the public OpenAPI description and fingerprint with official tools.

Retained API-owned authenticated database acquisition and worker-local cleanup.
The existing core resolver already owns saved-part decoding, validation and
deployment-default selection. Moving HTTP dependencies into core would add
coupling; adding a callback/service wrapper would not move any prompt semantics.
This preserves the established lazy request snapshot and DB-free core callers.

Post-rebase and post-review validation each passed 112 focused backend tests.
Ruff lint/format, compilation, Bandit (zero findings) and official OpenAPI
export/type generation/fingerprint validation passed. No runtime behavior
changed during this review correction; frontend code remains unchanged.
