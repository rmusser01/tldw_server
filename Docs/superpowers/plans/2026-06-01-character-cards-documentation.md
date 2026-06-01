# Character Cards Documentation Implementation Plan

## Stage 1: Source Review

**Goal**: Read the current Character Cards and Character Chat API/core files before writing docs.

**Success Criteria**: Documentation statements are grounded in current route prefixes, schema fields, import/export behavior, prompt assembly, persistence, and limits.

**Tests**: Local source inspection with `rg` and targeted file reads.

**Status**: Complete

## Stage 2: User Documentation

**Goal**: Add a source user guide that explains practical Character Cards and Character Chat use.

**Success Criteria**: Guide covers card fields, import/export, chat session flow, messages, completions, greetings, author notes, world books, dictionaries, safety/privacy boundaries, and common errors.

**Tests**: Markdown/source-link checks and manual consistency review against endpoint/schema files.

**Status**: Complete

## Stage 3: Module README

**Goal**: Replace stale `Character_Chat` module README content with a current developer map.

**Success Criteria**: README describes module files, data flow, endpoint integration, limits, extension rules, and relevant test locations without stale line references.

**Tests**: Verify referenced files/directories exist and run markdown whitespace checks.

**Status**: Complete

## Stage 4: Verification And Tracking

**Goal**: Verify docs-only changes and close the Backlog task.

**Success Criteria**: `git diff --check` passes, generated `Docs/Published` is untouched, docs avoid placeholder markers, Bandit skip is recorded as docs-only, and Backlog task is updated.

**Tests**: `git diff --check`, generated-doc status check, path checks, and targeted text scans.

**Status**: Complete
