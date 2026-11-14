import { useComposerAttachments } from "@/components/Chat/composer/hooks/useComposerAttachments"

// ---------------------------------------------------------------------------
// Deps interface
// ---------------------------------------------------------------------------

export interface UsePlaygroundAttachmentsDeps {
  /** Chat mode - images disabled in RAG mode */
  chatMode: string
  captureImageOwnership: () => () => boolean
  /** Form helpers */
  setFieldValue: (field: string, value: any) => void
  /** File upload handler from useMessageOption */
  handleFileUpload: (file: File) => Promise<unknown>
  /** Notification for disabled image */
  notifyImageAttachmentDisabled: () => void
  notifyImageReadError?: () => void
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Playground attachment wiring. Thin wrapper over the shared
 * `useComposerAttachments` primitive — Playground accepts both images AND
 * documents, and blocks image attachments in RAG mode. No toast feedback;
 * unsupported-type warnings fall through to the shared primitive's
 * `onUnsupportedType`, which we log here for parity with the old inline
 * behavior.
 */
export function usePlaygroundAttachments(deps: UsePlaygroundAttachmentsDeps) {
  const {
    chatMode,
    captureImageOwnership,
    setFieldValue,
    handleFileUpload,
    notifyImageAttachmentDisabled,
    notifyImageReadError
  } = deps

  return useComposerAttachments({
    chatMode,
    captureImageOwnership,
    setImageField: (base64) => setFieldValue("image", base64),
    onImageAccepted: (file) => setFieldValue("imageFilename", file.name),
    onDocumentUpload: async (file) => {
      await handleFileUpload(file)
    },
    ragBlocksImages: true,
    onImageBlockedInRagMode: notifyImageAttachmentDisabled,
    onImageReadError: notifyImageReadError,
    onUnsupportedType: (file) => {
      console.error("File type not supported:", file.type)
    }
  })
}
