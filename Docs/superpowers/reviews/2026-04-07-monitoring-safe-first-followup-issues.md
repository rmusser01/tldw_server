# Monitoring Safe-First Follow-Up Issues

## Umbrella Issue

Title: Monitoring follow-ups: tighten alert identity, lifecycle, and digest semantics

Body:

- Background: the safe-first remediation intentionally preserved current Monitoring API behavior while fixing internal defects, adding regression coverage, and documenting the current contract.
- Goal: track the contract and design changes that should not ship silently in a compatibility-focused batch.
- Scope:
  - stricter overlay identity validation or a first-class overlay-only contract
  - public monitoring alert lifecycle and response redesign
  - real digest delivery semantics if desired
  - admin/public permission-model clarification if later needed

## Suggested Subtasks

### Subtask 1

Title: Define the authoritative contract for overlay-only monitoring identities

### Subtask 2

Title: Redesign public monitoring alert mutation responses around merged state

### Subtask 3

Title: Decide whether Monitoring digest mode should send compiled deliveries

### Subtask 4

Title: Clarify long-term Monitoring admin/public permission boundaries
