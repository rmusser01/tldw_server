## Stage 1: Parser Validation
**Goal**: Reject or normalize invalid tag-derived chapter, speed, and timestamp data before it reaches worker orchestration.
**Success Criteria**: Duplicate chapter IDs are made unique with warnings; invalid speeds and timestamps are dropped with warnings.
**Tests**: Focused unit tests in `test_audiobook_tag_parser.py`.
**Status**: Complete

## Stage 2: Alignment Timeline Safety
**Goal**: Prevent alignment anchors from producing non-monotonic adjusted word timings.
**Success Criteria**: Later anchors before the last adjusted cue end are ignored and subtitle timelines remain ordered.
**Tests**: Focused unit tests in `test_audiobook_alignment_anchors.py`.
**Status**: Complete

## Stage 3: Subtitle Text Safety
**Goal**: Escape or sanitize cue text for SRT, VTT, and ASS output without breaking expected line wrapping.
**Success Criteria**: Cue text cannot inject subtitle control markup or extra cue blocks.
**Tests**: Focused unit tests in `test_audiobook_subtitle_generator.py`.
**Status**: Complete

## Stage 4: Subtitle Parser Tightening
**Goal**: Only remove actual SRT/VTT timing lines while preserving ordinary dialogue containing arrows.
**Success Criteria**: Dialogue text containing `-->` survives normalization, while timing metadata is still removed.
**Tests**: Focused unit tests in `test_audiobook_parse_utils.py`.
**Status**: Complete

## Stage 5: Verification
**Goal**: Run targeted tests and security scan for the touched audiobook core scope.
**Success Criteria**: Targeted tests pass, Bandit reports no new actionable findings, and the backlog task records the result.
**Tests**: `python -m pytest` for touched Audiobook unit tests plus Bandit on touched core files.
**Status**: Complete
