/**
 * Whether the composer holds anything to send (CC-02, #3107). Send stays
 * disabled until it does, instead of looking ready and then doing nothing.
 * Mirrors the empty-draft check in usePlaygroundSubmit: text, an image,
 * attached tabs or files, imported page context, or pinned search results
 * (which are added to the message unless file retrieval is on).
 */
export const isComposerDraftEmpty = ({
  message,
  image,
  documentCount,
  fileCount,
  hasImportedContext,
  pinnedResultCount,
  fileRetrievalEnabled
}: {
  message: string | null | undefined
  image: string | null | undefined
  documentCount: number
  fileCount: number
  hasImportedContext: boolean
  pinnedResultCount: number
  fileRetrievalEnabled: boolean
}): boolean =>
  !(message ?? "").trim() &&
  !(image ?? "").length &&
  documentCount === 0 &&
  fileCount === 0 &&
  !hasImportedContext &&
  (fileRetrievalEnabled || pinnedResultCount === 0)
