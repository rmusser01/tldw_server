# UAT219 / TASK13260.157

Approved bounded repair: schemas/study_packs.py StudyPackSummaryResponse only, plus a new actual-route regression.

The completed-job route reads real persisted pack data then calls _serialize_study_pack → StudyPackSummaryResponse.model_validate. The direct pack-detail route shares this serializer. PG TIMESTAMPTZ becomes datetime, while the response declares Optional[str] for created_at/last_modified. A pure schema probe produces exactly two string_type validation errors for those fields. Existing completed-job/detail tests use SQLite string values.

Reuse the established datetime-only before-validator in DeckResponse, FlashcardReviewResponse and StudyAssistantThreadSummary: value.isoformat() if isinstance(value, datetime), otherwise value unchanged. This preserves public strings, timezone offsets, naive values, null and invalid-type rejection. No endpoint/DB/worker behavior change, no timestamp parsing or coercion of unrelated types. An endpoint-local conversion would duplicate the contract and miss nested uses; changing fields to datetime would alter existing string validation/normalization semantics.

Tests: official real PG + normal SQLite content databases, actual JobManager completion and real FastAPI completed-job/detail routes; compare full persisted rows before/after reads, including owner/version/source/deck. Test foreign job/missing-result/missing-detail controls without weakening authorization. Schema controls cover UTC, offset, naive, SQLite strings, null and invalid number/object/list/date inputs. Jobs are explicitly SQLite-backed for this API read boundary; adjacent existing job tests retain their separate required-PG controls.

Native job5 is preserved and remains parent-owned; no regenerate, native read or mutation is part of this implementation. Adjacent FlashcardCitationResponse timestamp risk was reported separately as source-only and is excluded from this production change.
