# Moderation Backend Review Workspace

Stage order: `1 -> 2 -> 3 -> 4 -> 5`

Stage reports:
- [Stage 1](./2026-04-07-stage1-baseline-and-inventory.md)
- [Stage 2](./2026-04-07-stage2-policy-and-rule-parsing.md)
- [Stage 3](./2026-04-07-stage3-endpoints-caller-and-permissions.md)
- [Stage 4](./2026-04-07-stage4-persistence-concurrency-and-verification.md)
- [Stage 5](./2026-04-07-stage5-test-gaps-and-final-synthesis.md)

Rules for this review:
- Write confirmed findings before probable risks and improvements.
- Label uncertain items as probable risks or open questions instead of overstating them as confirmed defects.
- Persistence and concurrency claims require targeted verification, or the report must explicitly downgrade confidence.
- `tldw_Server_API/app/api/v1/endpoints/chat.py` is inspected only at moderation call sites, not as a general chat review.
- Keep the final report findings-first and evidence-backed for the backend moderation surface only.

Canonical final output:
```markdown
## Findings
### Confirmed findings
- severity, confidence, file references, impact, and fix direction when clear

### Probable risks
- material issues not fully proven, with explicit confidence limits

## Open Questions
- only unresolved ambiguities that materially affect confidence

## Improvements
- lower-priority hardening or maintainability suggestions

## Verification
- files inspected, tests run, and what remains unverified
```

Stage 1 captures the scaffold, baseline command output, and scoped inventory. Later stages should build on that fixed starting point rather than re-litigating the workspace state.
