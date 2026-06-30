# Prototype Workspaces User Guide

## Overview

Prototype Workspaces let an owner share a working app prototype with outside collaborators without letting every collaborator change the shared canonical version directly.

The main roles are:

- Owner: creates the workspace, shares private links, reviews promotion requests, and controls the canonical prototype.
- Collaborator: opens a private link, works in an isolated branch session, saves candidate snapshots, and asks the owner to promote a candidate.
- Designated promoter: an internal user allowed by the owner to review or promote candidates.

## Owner Flow

1. Create a prototype workspace from the prototype workspace page.
2. Review the seed canonical snapshot and current canonical preview state.
3. Start an owner branch session when you want to make changes without immediately changing the canonical prototype.
4. Create a private share link for outside collaborators.
5. Choose whether the link is password-protected, whether browser resume is allowed, and how many times the link can be used.
6. Review incoming promotion requests from collaborator branches.
7. Approve only candidates that should become the shared canonical prototype. Reject candidates that need more work.

The canonical prototype changes only after an owner or designated promoter approves a candidate and validation succeeds.

## Collaborator Flow

1. Open the private `/share/:token` link from the owner.
2. Enter the password if the link is password-protected.
3. Enter a display name on first use so the owner can identify the branch and promotion request.
4. Work in the isolated collaborator branch session.
5. Save a candidate snapshot when the branch has a useful change.
6. Submit a promotion request for owner review.

Collaborators do not become full user accounts. The system creates a scoped shared actor for the workspace and link.

## Link And Resume Behavior

Password-protected links require the password unless the same browser has a valid resume cookie for the same active link and collaborator. The resume cookie is only a browser-binding hint; it is not a durable password or session token that should be copied between devices.

Single-use links can be used once for a first-time collaborator. If the same browser has a valid resume cookie, that collaborator may resume even after the first use. A different browser or user sees an exhausted link once the use count is spent.

If a link is expired, revoked, exhausted, invalid, or points to a missing workspace, the collaborator sees a generic unavailable-link state. This is intentional so public link responses do not reveal which links exist.

If the workspace is archived, the collaborator sees that the workspace is unavailable. The owner must unarchive it or create a new workspace before collaboration can continue.

## Common Examples

### Password-protected link

The owner creates a private link with a password and sends the link and password separately. The collaborator opens the link, sees a password prompt, enters the password, and then starts their branch. If the password is wrong, the prompt remains visible and can be retried.

### Single-use and exhausted link

The owner creates a single-use link for one stakeholder. The first successful collaborator entry consumes the link. If another person opens it later in a different browser, they see an exhausted link as a generic unavailable-link state. The original collaborator can still resume from the same browser if resume is allowed and the resume cookie is valid.

### Resume cookie

A collaborator closes the browser tab after starting work. Later, they open the same private link in the same browser. If the owner has not revoked the link or workspace access and the session is still active, the resume cookie lets the collaborator continue without re-entering the password.

### Revoked link

The owner revokes a link after sending it. New visitors see an unavailable link. A collaborator who already exchanged the link may see their session become inactive and should ask the owner for a fresh link if work should continue.

### Archived workspace

The owner archives the workspace while a collaborator still has the link. The collaborator cannot enter or continue the workspace. The owner should unarchive the workspace or create a new share link from an active workspace.

### Promotion conflict

A collaborator submits a candidate after the canonical prototype changed. The owner may see a stale or promotion conflict state instead of a successful promotion. The collaborator should start from the current canonical prototype, reapply the useful changes, and submit a new request.

### Validation failure

The owner approves a candidate, but validation fails. The canonical prototype does not change. The collaborator should inspect the candidate branch, make a safer change, save a new snapshot, and submit another promotion request.

## Owner Review Outcomes

Promotion requests can end in these user-visible outcomes:

- Approved and promoted: the candidate becomes canonical after validation.
- Rejected: the owner records review notes and the canonical prototype does not change.
- Stale: the candidate was based on an older canonical snapshot.
- Promotion conflict: validation detected a conflict that needs a new candidate or owner decision.
- Validation failure: the candidate did not pass publish validation and the canonical prototype remains unchanged.

## Safety Expectations

- Do not save share tokens, session tokens, passwords, preview grants, or resume cookies into notes, screenshots, or bug reports.
- Share passwords separately from the private link.
- Ask the owner for a fresh link when a link or session is unavailable and the work should continue.
- Treat collaborator branches as drafts until the owner promotes a candidate.

## Gate 8 Handoff

Backend/Core owns the operational contracts, status fields, and security behavior. Frontend/Product owns the release evidence that these user flows render correctly and stay understandable for owners and collaborators.

Gate 8 should include product review evidence for password-protected entry, single-use and exhausted links, resume cookie continuation, revoked link handling, archived workspace handling, promotion conflict handling, and validation failure recovery.
