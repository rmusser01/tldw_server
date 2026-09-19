# UAT322: Evidence identity and navigation

Backlog: TASK13260.260. Add public title, canonical source type, and record ID at existing retrieval boundaries. The existing streaming allowlist already forwards these fields. Route source actions through the existing Character page and Chat thread path builder; never interpret arbitrary source URLs as internal paths.

## Stage 1: Reproduce
**Goal**: Verify missing identity through actual retrieval and missing source actions in the UI.
**Success Criteria**: SQLite/PostgreSQL pipeline and frontend controls fail causally.
**Tests**: Four real pipeline-to-stream cases and two source navigation cases.
**Status**: Complete

## Stage 2: Repair and review
**Goal**: Preserve truthful source identity and supported source actions.
**Success Criteria**: Source IDs, titles and types survive stream/nonstream serialization; links use existing routes.
**Tests**: Focused and adjacent backend/frontend tests, static checks, review.
**Status**: Complete

## Stage 3: Native acceptance
**Goal**: Verify source inspection on fresh SQLite/PostgreSQL alongside UAT321/323.
**Success Criteria**: Owner opens the correct source; Bob cannot retrieve foreign evidence.
**Tests**: Native source actions, source parity and owned runtime cleanup.
**Status**: In Progress
