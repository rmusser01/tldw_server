import React from "react"
import { toBase64 } from "~/libs/to-base64"
import {
  inferImageAttachmentMimeType,
  normalizeImageDataUrlMime
} from "@/utils/image-utils"
import { otherUnsupportedTypes } from "@/components/Option/Knowledge/utils/unsupported-types"

/**
 * Shared attachment handler consumed by both composer surfaces.
 *
 * Playground (images + documents, no toasts, RAG-mode guard) and Sidepanel
 * (images only, toast feedback, no RAG guard) diverged on their inline
 * implementations. This primitive unifies the decision tree:
 *
 *   1. If the file type is in the unsupported list → `onUnsupportedType`.
 *   2. If it's an image:
 *        a. if `ragBlocksImages` + chatMode === "rag" → `onImageBlockedInRagMode`.
 *        b. otherwise read to base64, call `setImageField`, `onImageAccepted`.
 *        c. if read fails → `onImageReadError`.
 *   3. If it's a non-image:
 *        a. surfaces that accept documents pass `onDocumentUpload`.
 *        b. image-only surfaces (Sidepanel) pass `onNonImageRejected` instead.
 *
 * Every branch is a callback so surfaces decide whether to toast, log, or
 * ignore. Keeping the decision tree central guarantees both surfaces stay
 * in sync if (when) we add new attachment types.
 */

export interface UseComposerAttachmentsOptions {
  /** Current chat mode — used with `ragBlocksImages`. */
  chatMode: string

  /** Called with base64 data after an image is successfully read. */
  setImageField: (base64: string) => void

  /**
   * Called for non-image files. Omit for image-only surfaces (Sidepanel);
   * those use `onNonImageRejected` instead.
   */
  onDocumentUpload?: (file: File) => Promise<void>

  /**
   * If true and `chatMode === "rag"`, image attachments are blocked and
   * `onImageBlockedInRagMode` fires. Default false.
   */
  ragBlocksImages?: boolean
  onImageBlockedInRagMode?: () => void

  /** Fired once an image is successfully base64-encoded. Useful for toasts. */
  onImageAccepted?: (file: File) => void

  /** Fired when the file's MIME type is in the unsupported-types list. */
  onUnsupportedType?: (file: File) => void

  /**
   * Fired when a non-image file is handed to an image-only surface.
   * Only called when `onDocumentUpload` is NOT set.
   */
  onNonImageRejected?: (file: File) => void

  /** Fired if base64 conversion throws. */
  onImageReadError?: (error: unknown) => void
}

export interface UseComposerAttachmentsResult {
  /** Ref for the hidden <input type="file" accept="image/*"> trigger. */
  inputRef: React.RefObject<HTMLInputElement | null>
  /** Ref for the hidden <input type="file"> document trigger. */
  fileInputRef: React.RefObject<HTMLInputElement | null>
  /** Onchange handler for `<input type="file">`. Routes through `onInputChange`. */
  onFileInputChange: (
    e: React.ChangeEvent<HTMLInputElement>
  ) => Promise<void>
  /**
   * Polymorphic entry — accepts either a DOM event or a File directly.
   * Drop-targets and paste handlers pass the File object.
   */
  onInputChange: (
    e: React.ChangeEvent<HTMLInputElement> | File
  ) => Promise<void>
  /** Clicks the image input ref. */
  handleImageUpload: () => void
  /** Clicks the file input ref. */
  handleDocumentUpload: () => void
  /** Effect-style hook to process dropped files once. */
  useDroppedFiles: (droppedFiles: File[]) => void
}

export function useComposerAttachments(
  options: UseComposerAttachmentsOptions
): UseComposerAttachmentsResult {
  const {
    chatMode,
    setImageField,
    onDocumentUpload,
    ragBlocksImages = false,
    onImageBlockedInRagMode,
    onImageAccepted,
    onUnsupportedType,
    onNonImageRejected,
    onImageReadError,
  } = options

  const inputRef = React.useRef<HTMLInputElement>(null)
  const fileInputRef = React.useRef<HTMLInputElement>(null)
  const processedFilesRef = React.useRef<WeakSet<File>>(new WeakSet())

  const processFile = React.useCallback(
    async (file: File) => {
      const imageMimeType = inferImageAttachmentMimeType(file)

      if (!imageMimeType && otherUnsupportedTypes.includes(file.type)) {
        onUnsupportedType?.(file)
        return
      }

      if (imageMimeType) {
        if (ragBlocksImages && chatMode === "rag") {
          onImageBlockedInRagMode?.()
          return
        }
        try {
          const base64 = await toBase64(file)
          setImageField(normalizeImageDataUrlMime(base64, imageMimeType))
          onImageAccepted?.(file)
        } catch (error) {
          onImageReadError?.(error)
        }
        return
      }

      // Non-image file
      if (onDocumentUpload) {
        await onDocumentUpload(file)
      } else {
        onNonImageRejected?.(file)
      }
    },
    [
      chatMode,
      onDocumentUpload,
      onImageAccepted,
      onImageBlockedInRagMode,
      onImageReadError,
      onNonImageRejected,
      onUnsupportedType,
      ragBlocksImages,
      setImageField,
    ]
  )

  const onFileInputChange = React.useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      try {
        if (e.target.files && e.target.files[0]) {
          await processFile(e.target.files[0])
        }
      } finally {
        e.target.value = ""
      }
    },
    [processFile]
  )

  const onInputChange = React.useCallback(
    async (e: React.ChangeEvent<HTMLInputElement> | File) => {
      if (e instanceof File) {
        await processFile(e)
        return
      }
      if (e.target.files) {
        await onFileInputChange(e)
      }
    },
    [onFileInputChange, processFile]
  )

  const handleImageUpload = React.useCallback(() => {
    inputRef.current?.click()
  }, [])

  const handleDocumentUpload = React.useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  const useDroppedFiles = (droppedFiles: File[]) => {
    React.useEffect(() => {
      if (droppedFiles.length === 0) return
      let cancelled = false
      const run = async () => {
        for (const file of droppedFiles) {
          if (cancelled) return
          if (processedFilesRef.current.has(file)) continue
          try {
            processedFilesRef.current.add(file)
            await processFile(file)
          } catch (error) {
            processedFilesRef.current.delete(file)
            console.error("Failed to process dropped file:", file.name, error)
          }
        }
      }
      void run()
      return () => {
        cancelled = true
      }
    }, [droppedFiles])
  }

  return {
    inputRef,
    fileInputRef,
    onFileInputChange,
    onInputChange,
    handleImageUpload,
    handleDocumentUpload,
    useDroppedFiles,
  }
}
