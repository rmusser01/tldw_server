import React from "react"
import { Modal } from "antd"
import { useTranslation } from "react-i18next"
import type { FlashcardTemplateCreate } from "@/services/flashcards"
import { FlashcardTemplateForm } from "./FlashcardTemplateForm"

interface FlashcardSaveTemplateModalProps {
  open: boolean
  onClose: () => void
  initialValues?: Partial<FlashcardTemplateCreate> | null
  onSave: (values: FlashcardTemplateCreate) => Promise<void>
  isSaving?: boolean
}

export const FlashcardSaveTemplateModal: React.FC<FlashcardSaveTemplateModalProps> = ({
  open,
  onClose,
  initialValues,
  onSave,
  isSaving = false
}) => {
  const { t } = useTranslation(["option"])

  const handleSubmit = React.useCallback(
    async (values: FlashcardTemplateCreate) => {
      try {
        await onSave(values)
        onClose()
      } catch {
        // Keep the modal open so the user can correct or retry after a failed save.
      }
    },
    [onClose, onSave]
  )

  return (
    <Modal
      open={open}
      onCancel={onClose}
      destroyOnHidden
      footer={null}
      title={t("option:flashcards.saveAsTemplate", {
        defaultValue: "Save as template"
      })}
      width={760}
    >
      <FlashcardTemplateForm
        mode="create"
        initialValues={initialValues}
        onSubmit={handleSubmit}
        onCancel={onClose}
        submitting={isSaving}
      />
    </Modal>
  )
}

export default FlashcardSaveTemplateModal
