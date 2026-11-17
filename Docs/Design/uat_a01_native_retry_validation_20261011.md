# A01 native unanswered-input retry validation

Task: TASK-13260.278.18.83.45. Plan: `Docs/Plans/IMPLEMENTATION_PLAN_uat_a01_first_setup_chat_20261009.md`.
ADR assessment: no new durable decision. This repairs the implementation of [ADR-049](../ADR/049-chat-history-selection-ownership.md) and the existing H1 admission contract; it does not redefine either revision domain.

## Problem

Native admission records the physical input message version, while the full history capture revision hashes content, protected authority, metadata and assets. Comparing those tokens rejects an unchanged native question. Local admission uses its local semantic revision and keeps its existing comparison.

## Narrow opt-in contract

- Add optional `accepted_input_admission` to the existing capture request. It is permitted only for `send` with the cursor after that accepted input. Existing ordinary capture and existing-input regeneration requests retain their wire behavior.
- On this opt-in read, the authenticated persistence owner validates the supplied immutable receipt and current input through the existing protected-parent validation, inside the same transaction as capture. A current capture/source check binds that proof to the returned semantic input revision. No message or new admission is created.
- Only successful opted-in capture returns `validated_admission: { admission, input_revision }`. The receipt is unchanged; `input_revision` names the full semantic revision in this capture.
- Native retry requires matching acknowledged receipt and captured semantic revision, exact original text/images, no existing assistant child and all existing owner/lease/prompt/context/fresh-recovery checks. Missing/unsupported/mismatched proof fails closed. Local retry is unchanged.
- Normal server settlement revalidates the original protected admission; capture proof grants no durable mutation authority.

## Verification

Real storage regressions on SQLite and official isolated PostgreSQL cover unchanged physical-versus-semantic tokens, receipt/owner tamper, deleted/edited inputs, metadata and asset drift without physical-version changes, changed scope and capture drift. API cases cover opt-in proof, normal-wire preservation and no new message. Native mounted-client fixtures use distinct physical and semantic revisions; missing/tampered proof refuses dispatch. Existing navigation/account/input/prompt/no-duplicate-user protections remain.

A01 live validation uses a new repaired-source negative request and one reviewed working-model retry only after source/tests/security/review/canonical types/build qualify. Earlier failed attempts remain consumed. Other UAT workflows and publication remain out of scope until A01 is accepted.
