/**
 * Whether the composer holds anything a send would deliver: text, an image,
 * attached documents or files, page context handed over from the side panel,
 * or pinned search results (which submit appends unless file retrieval is on,
 * so callers pass 0 then). Mirrors the empty-draft check in the submit path.
 */
export const hasSendableDraft = ({
  message,
  image,
  attachmentCount = 0,
  hasImportedContext = false,
  pinnedResultCount = 0
}: {
  message?: string | null
  image?: string | null
  attachmentCount?: number
  hasImportedContext?: boolean
  pinnedResultCount?: number
}): boolean =>
  Boolean(message?.trim()) ||
  Boolean(image) ||
  attachmentCount > 0 ||
  hasImportedContext ||
  pinnedResultCount > 0
