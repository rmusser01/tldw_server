# Skills Export Metadata Feedback Design

## Context

`/api/v1/skills/{skill_name}/export` already returns zip bytes with a
`Content-Disposition` filename. The frontend currently discards response
headers, guesses `${name}.zip`, and only notifies users on failure. That works
for the current simple case, but it makes export behavior brittle if the server
later changes naming, adds conflict-safe filenames, or returns encoded names.

## Goal

Preserve server-provided export filename metadata through the WebUI client and
use it for browser downloads and user feedback.

## Scope

- Change the Skills frontend API helper to return `{ blob, filename }`.
- Parse `filename=` and `filename*=UTF-8''...` from `Content-Disposition`.
- Fall back to a safe `${skillName}.zip` filename when metadata is absent,
  invalid, or unsafe.
- Use the returned filename for the anchor download name.
- Show a success notification naming the downloaded file.
- Preserve existing sanitized error feedback.

Out of scope: backend API shape changes, bulk export, import review, permission
or model metadata panels, visual restyling, and route-level redesign.

## Design

The client contract should mirror existing export helpers in the codebase:
`exportSkill(name): Promise<{ blob: Blob; filename: string }>`. The helper will
request `arrayBuffer` with `returnResponse: true`, validate non-OK responses,
build the blob from `response.data`, and derive the filename from response
headers.

Filename parsing should prefer `filename*` because it supports encoded UTF-8
names. Plain `filename` remains supported for the current backend response.
After parsing, the client should reject path-like or empty filenames and fall
back to `${skillName}.zip`. This keeps the UI stable even when proxies strip
headers or a server returns malformed metadata.

`SkillsManager.handleExport()` will use the returned filename for the hidden
anchor download, then show a compact success notification. The notification is
post-click feedback that the browser download was initiated, not a guarantee
that the user saved the file. Failure feedback remains sanitized through the
existing `getErrorDescription()` path.

## Testing

- Service tests cover header filename parsing and safe fallback behavior.
- Manager tests cover download filename usage and success notification.
- Manager tests cover sanitized export failure notification.
- Existing Skills manager tests remain focused; no browser E2E is required for
  this contract-only slice.
