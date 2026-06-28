# Fish S2 Reference Import Implementation Plan

**Goal:** Extend the existing Fish S2 commercial TTS PR with JSON and Markdown import support for managed Fish S2 voice references.

**Architecture:** Add a parser for import files that normalizes JSON objects, JSON arrays, and Markdown files with optional frontmatter into the existing `create_fish_s2_reference` service flow. Keep Fish reference persistence in local voice metadata and reuse the current remote-reference sync path.

**Tech Stack:** FastAPI multipart uploads, pytest, existing TTS service and voice manager abstractions.

## Stage 1: Import File Contract
**Goal**: Define the supported JSON and Markdown shapes for Fish S2 reference imports.
**Success Criteria**: JSON object/array imports and Markdown frontmatter/body imports normalize into the same internal item shape.
**Tests**: Parser tests for object, array, embedded audio, Markdown frontmatter, and invalid payloads.
**Status**: Complete

## Stage 2: API Endpoint
**Goal**: Add a Fish S2 reference import endpoint to the existing audio voice router.
**Success Criteria**: Endpoint accepts `.json`, `.md`, and `.markdown` uploads, forwards normalized items to `create_fish_s2_reference`, and returns per-item results.
**Tests**: Integration tests for JSON and Markdown imports using fake TTS service overrides.
**Status**: Complete

## Stage 3: Documentation And Verification
**Goal**: Document the import file format and verify touched code.
**Success Criteria**: TTS setup docs mention JSON and Markdown import examples; focused pytest and Bandit checks pass.
**Tests**: Focused TTS endpoint/parser tests plus Bandit on touched production Python files.
**Status**: Complete
