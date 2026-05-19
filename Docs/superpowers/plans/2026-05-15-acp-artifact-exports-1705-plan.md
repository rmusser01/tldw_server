## Stage 1: Export Contract And Schemas
**Goal**: Add the backend request/response contract for accepted artifact version exports.
**Success Criteria**: Markdown, HTML, and JSON are explicit accepted formats; responses expose artifact id, artifact version id, workspace id, review state, export format, generated timestamp, metadata, content, byte count, and persisted export reference.
**Tests**: Focused workspace API export tests fail before implementation and pass after implementation.
**Status**: Complete

## Stage 2: Accepted-Version Export Service
**Goal**: Render accepted workspace artifact versions into Markdown, HTML, and JSON while preserving traceability metadata.
**Success Criteria**: Exported payloads embed or expose source lineage, producer metadata, review metadata, version metadata, redaction posture, and stable artifact identity. Non-accepted review states fail closed.
**Tests**: API tests cover all three formats and non-accepted rejection.
**Status**: Complete

## Stage 3: Export Reference Persistence
**Goal**: Record export references back onto the workspace artifact without creating a new content version or losing existing references.
**Success Criteria**: Existing export refs remain intact; each export adds a ref tied to the source artifact version.
**Tests**: API tests fetch the artifact after export and verify legacy plus new refs.
**Status**: Complete

## Stage 4: Documentation And Verification
**Goal**: Document the implemented contract and run focused verification for the touched backend scope.
**Success Criteria**: Product docs name the concrete endpoint and remaining richer-export boundaries; pytest, syntax checks, Bandit, and diff checks are recorded.
**Tests**: Focused workspace API/DB tests plus touched-scope security checks.
**Status**: Complete
