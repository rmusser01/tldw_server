import type { RagSettings } from "@/services/rag/unified-rag"

export type ConferenceCollectionKnowledgeQaOptions = Pick<
  Partial<RagSettings>,
  "collection_id" | "sources"
>

const normalizeCollectionId = (collectionId: number | string): number => {
  const numeric =
    typeof collectionId === "number"
      ? collectionId
      : Number(collectionId.trim())
  if (!Number.isInteger(numeric) || numeric <= 0) {
    throw new Error("collection_id_required")
  }
  return numeric
}

export const buildConferenceCollectionKnowledgeQaOptions = (
  collectionId: number | string
): ConferenceCollectionKnowledgeQaOptions => ({
  collection_id: normalizeCollectionId(collectionId),
  sources: ["media_db"],
})
