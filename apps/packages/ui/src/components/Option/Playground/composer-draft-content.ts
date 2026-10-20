/**
 * Whether the composer holds anything a send would deliver: text, an image,
 * attached documents or files, or page context handed over from the side
 * panel. Mirrors the empty-draft check in the submit path.
 */
export const hasSendableDraft = ({
  message,
  image,
  attachmentCount = 0,
  hasImportedContext = false
}: {
  message?: string | null
  image?: string | null
  attachmentCount?: number
  hasImportedContext?: boolean
}): boolean =>
  Boolean(message?.trim()) ||
  Boolean(image) ||
  attachmentCount > 0 ||
  hasImportedContext
