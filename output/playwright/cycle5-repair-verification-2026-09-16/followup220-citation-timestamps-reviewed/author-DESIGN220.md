# UAT220 / TASK13260.158

Approved narrow repair: FlashcardCitationResponse.created_at/last_modified before-validator in schemas/study_packs.py. Shared source remains held until independent219 reviewer releases its hash verification. Baseline includes only219's accepted candidate Summary validator.

Actual official PG probe: GET /flashcards/{card}/assistant with a real citation and no StudyPack membership raises four FastAPI response validation errors at citations[0] and primary_citation created_at/last_modified; each input_type=datetime. SQLite returns200. FlashcardProvenanceStore.list_citations copies driver timestamps unchanged; FlashcardCitationResponse declares Optional[str].

Use the same datetime-only isoformat before-validator as Deck/Study/Review/StudyPackSummary. Do not change service, storage, ownership, endpoint, parser, coercion rules or public string contract. Existing strings/null stay exact; offsets and naive datetime follow existing conventions; other input types fail validation. The shared datetime import already exists after219, so220 requires only one class-local method.

Permanent tests: actual official PG and separate SQLite assistant GET with persisted note/card/citations, both with and without a StudyPack; verify citation list and primary timestamps, source identities, ownership/version and persisted rows unchanged. Legacy empty citation behavior remains200. A foreign owner uses its real selected DB and receives404 without modifying owner data. Pure schema controls cover UTC/offset/naive/string/null and invalid number/object/list/date-only types, idempotent model output and unchanged non-time data.

No original native card/job mutation, provider call, service restart or browser action. Root owns same cited-card native acceptance after combined restart.219 and220 attribution remains separate despite one shared schema file.
