# Chat Source URL Eligibility

Tracking: TASK-13398.22; related navigation acceptance TASK-13398.18.
Status: approved by the human requester on 2026-09-30; implementation in progress.
Original verified baseline: dev2256bc82afa154891c635df3ef955ed7a6bc61b3.
Current baseline: devf3f1b4fdbe3fe461b371ece30887c5fff8476d9d plus preserved candidate.

## Verified Problem

Real document ingestion records the uploaded filename as provenance. RAG carries
that value in metadata.url; ragMode forwards it into structured message sources.
MessageSource applies the generic safeExternalUrl guard, which deliberately
supports relative application paths. A filename consequently becomes a link
relative to the WebUI origin. Actual Chrome opened lumen-field-memo.md at a
missing Next route; an actual GET confirmed HTTP404. Canonical owned Browse
preview independently returns the uploaded content200.

## Bounded Proposal

Qualify navigation in the existing shared source-card renderer, not each chat
caller. Open source should require an explicit validated absolute HTTP(S) URL;
an uploaded filename remains provenance/evidence, not an invented hosted URL.
Reuse the existing URL validation/parsing helpers. Keep the generic relative-URL
guard unchanged for internal routes. Keep source metadata, citation evidence,
feedback, tracking and canonical scoped Browse unchanged. Do not rewrite stored
provenance, create a synthetic route, infer a media ID from generated text or
introduce a new navigation service.

The requester explicitly approved this source-card eligibility boundary. Native
acceptance remains required separately from the preceding missing-prop repair.

## Verification

RED/GREEN: actual shared renderer with an uploaded basename, absolute safe URL,
unsafe/obfuscated URL and its existing source/feedback wrappers. Preserve details
and default unrelated commands. Run focused sibling tests, TypeScript and scoped
lint. Bandit is not applicable if the implementation is TypeScript-only.

Native Chrome: actual uploaded-document RAG, all returned chunks preserved, no
invented filename link, actual canonical Browse content200. Verify a genuine
returned absolute source URL when available; never create a fake route or replace
API/model/auth/storage responses to make navigation pass. Record unavailable
coverage separately from acceptance. Full checkpoint UAT remains separate.

Evidence: /private/tmp/chat-workspace-approved-six-20260930/
native-uploaded-citation-navigation.json and its screenshot. Historical failing
probes and their assertions remain unchanged.
