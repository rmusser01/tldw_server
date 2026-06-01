# Audiobooks

Audiobooks contains the text parsing, chapter/tag interpretation, subtitle generation, subtitle parsing, and alignment helpers used by the audiobook creation API. It turns structured or tagged source text into chapter-aware plans, generates SRT/VTT/ASS subtitle cues, and applies alignment anchors to TTS output timing data.

## Start Here

- `tag_parser.py` parses audiobook markup, chapter markers, scalar controls, speed controls, and timestamp tags.
- `subtitle_generator.py` creates subtitle cues and exports SRT, VTT, and ASS output.
- `subtitle_parser.py` normalizes existing subtitle or plain-text inputs.
- `alignment_utils.py` applies timing anchors, scales payloads for playback speed, and stitches alignment payloads.
- Related API surface: `tldw_Server_API/app/api/v1/endpoints/audio/audiobooks.py`, declared under `/audiobooks`.
- Related schemas: `tldw_Server_API/app/api/v1/schemas/audiobook_schemas.py`.
- Related tests: `tldw_Server_API/tests/Audiobooks/`.

## Responsibilities

- Parse tagged audiobook source text into chapters, narration text, and voice or speed directives.
- Build chapter structures from explicit markers.
- Generate subtitle cues from text and timing metadata.
- Export subtitles in SRT, VTT, and ASS formats.
- Parse imported SRT, VTT, ASS, and text sources into normalized cues.
- Apply alignment anchors and speed scaling to TTS timing payloads.
- Support audiobook API flows for parsing, jobs, projects, chapters, artifacts, voice profiles, and subtitle export.

## Module Map

- `tag_parser.py`: audiobook markup and chapter parser.
- `subtitle_generator.py`: cue generation, sentence splitting, line chunking, and subtitle format serialization.
- `subtitle_parser.py`: import-side subtitle normalization.
- `alignment_utils.py`: alignment anchors, speed scaling, and payload stitching.
- `__init__.py`: package marker.

## How It Connects

- `audio/audiobooks.py` exposes parse, job, project, chapter, artifact, voice profile, and subtitle export routes.
- `audiobook_schemas.py` defines source references, chapter selection, voice overrides, output options, subtitle options, queue options, alignment payloads, jobs, projects, artifacts, and voice profiles.
- The endpoint connects to the Jobs module for background audiobook work and to Collections DB for project and artifact state.
- Audiobook flows connect to uploads, notes export, voice profile storage, and TTS provider hints through the endpoint layer.
- Subtitle persistence and cache behavior are controlled by Audiobooks configuration keys used by the endpoint and subtitle generation flow.

## Extension Points

- Add new source tags or chapter directives in `tag_parser.py`.
- Add a subtitle output option in `subtitle_generator.py` and `audiobook_schemas.py`.
- Add subtitle import behavior in `subtitle_parser.py`.
- Change timing behavior in `alignment_utils.py` before touching endpoint or job orchestration.
- Extend voice profile or project API behavior in `audio/audiobooks.py` and matching schema tests.

## Testing

- Direct tests live under `tldw_Server_API/tests/Audiobooks/`.
- Integration coverage includes worker pipeline, jobs endpoints, parse endpoint, alignment flow, subtitle export endpoint, and voice profile tests.
- Unit coverage includes tag parser, subtitle generator, subtitle parser, alignment anchor, worker plan tag, and schema tests.

## Gotchas

- Sentence splitting can use spaCy when enabled in Audiobooks configuration; behavior can differ when the optional model is absent.
- Subtitle persistence and cache lifetime depend on Audiobooks configuration.
- Speed scaling changes alignment timing, so subtitle and artifact tests should be checked together when timing logic changes.
