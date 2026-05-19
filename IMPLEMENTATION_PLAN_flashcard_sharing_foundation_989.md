# Flashcard Sharing Foundation

## Stage 1: Sharing Foundation

**Goal**: Add durable deck visibility and per-user share metadata without changing study/review access semantics yet.
**Success Criteria**: Decks persist a `visibility` value, deck shares can be upserted/listed/removed, deleted decks drop share records, and existing deck responses remain compatible.
**Tests**: Focused ChaChaNotes DB tests plus flashcards endpoint integration tests.
**Status**: Complete

## Stage 2: Shared Deck Access

**Goal**: Teach API reads how to resolve shared deck access across owner databases.
**Success Criteria**: A recipient can see decks shared with them in a separate section without mutating owner deck data.
**Tests**: Multi-user endpoint tests using owner/recipient DB fixtures.
**Status**: Not Started

## Stage 3: Per-User Review State

**Goal**: Separate shared-card content from each recipient's scheduling metadata.
**Success Criteria**: Reviews by recipients do not modify owner SRS state or other recipients' progress.
**Tests**: DB and API tests for owner and recipient review sessions on the same shared card content.
**Status**: Not Started
