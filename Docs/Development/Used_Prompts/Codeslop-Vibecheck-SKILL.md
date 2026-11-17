---
name: codeslop-vibecheck
description: Use when reviewing code for design-quality issues involving duplication, weak encapsulation, temporal or sequential coupling, hidden ordering dependencies, leaky abstractions, or maintainability risks in changed code.
---

# Codeslop Vibecheck

## Overview

Review code for maintainability hazards along three axes: code duplication, encapsulation, and sequential coupling. Focus on concrete risks in the code, not taste, style, or author intent.

## Core Rule

Find high-signal design problems that would make future changes harder, riskier, or more bug-prone. Prefer a few defensible findings over exhaustive commentary.

## Review Scope

- Review only the provided diff, changed files, or explicitly requested code.
- Treat surrounding code as context, not as a mandate to audit the whole repository.
- Flag pre-existing issues only when the change depends on them, worsens them, or makes them newly relevant.
- Do not flag formatting, naming, performance, security, or test coverage unless they directly intersect one of the three axes.
- Critique the code and design shape, never the developer.

## Quality Axes

### Code Duplication

Look for repeated knowledge that will create change amplification.

Flag:
- Repeated business rules, validation logic, branching, query construction, mapping, parsing, or error handling.
- Copy-pasted control flow with only small parameter changes.
- Multiple sources of truth for the same policy, schema, default, or state transition.
- Repeated setup or cleanup sequences that should be packaged behind one interface.

Do not flag:
- Small local repetition that improves readability.
- Similar code that reflects different domain rules.
- Duplication where abstraction would create a worse dependency or unclear API.

### Encapsulation

Look for boundaries that leak implementation details or force callers to know too much.

Flag:
- Callers reaching into internal state, private fields, raw storage shapes, or transport-specific details.
- Wide APIs that expose too many knobs instead of a coherent operation.
- Feature envy, where one module repeatedly assembles another module's internals.
- Mutable state escaping without guardrails.
- Business logic placed in the wrong layer, such as endpoint code owning persistence details.
- Hidden dependencies that are not explicit in function signatures, constructors, or configuration.

Do not flag:
- Explicit dependency injection.
- Thin adapters whose purpose is to translate across boundaries.
- Public data transfer objects when the boundary intentionally uses them.

### Sequential Coupling

Look for temporal coupling: correctness depends on doing steps in the right order.

Flag:
- APIs that require `initialize -> configure -> call -> cleanup` sequences without enforcing that sequence.
- Objects that are valid only after undocumented setup.
- Boolean flags, status fields, or caches that callers must manually coordinate.
- Repeated call choreography across files.
- Cleanup, rollback, commit, close, or refresh steps that can be skipped accidentally.
- State machines encoded as scattered conditionals rather than explicit transitions.

Prefer fixes that make invalid orderings impossible or harder to express.

## Review Procedure

1. Identify the changed behavior and touched boundaries.
2. Scan once for each axis: duplication, encapsulation, sequential coupling.
3. For each possible finding, ask: "Would this materially increase maintenance cost or defect risk?"
4. Keep only findings that are specific, actionable, and line-referenceable.
5. Choose one primary axis per finding, even if the issue overlaps multiple axes.
6. Suggest the smallest credible refactoring direction, not a broad rewrite.

## Output Format

Start with findings, ordered by severity. If there are no material findings, say that directly.

```text
Findings
- [severity: high|medium|low] [axis: duplication|encapsulation|sequential coupling] [file:line or symbol]
  Issue: <one-sentence title>
  Why it matters: <specific maintainability or correctness risk>
  Suggested direction: <smallest credible improvement>

Overall assessment
- <Brief statement on whether the change improves, worsens, or preserves design quality across the three axes.>
```

If no issues are found:

```text
No material issues found on duplication, encapsulation, or sequential coupling.
```

## Severity Guide

- High: likely correctness bug, fragile ordering hazard, or design issue that will predictably break future changes.
- Medium: meaningful maintainability drag, boundary leak, or duplicated policy likely to diverge.
- Low: localized design smell worth fixing opportunistically, but not blocking.

## Common Mistakes

- Do not report "duplication" just because two blocks look similar; identify the shared knowledge that will diverge.
- Do not demand encapsulation that hides useful domain concepts from legitimate callers.
- Do not call every multi-step workflow sequential coupling; flag it when the API relies on caller memory instead of enforcing safe usage.
- Do not produce architecture essays. Tie every finding to code and a practical next step.
- Do not invent missing context. If the diff is insufficient, mark the concern as tentative or ask for the needed file.
