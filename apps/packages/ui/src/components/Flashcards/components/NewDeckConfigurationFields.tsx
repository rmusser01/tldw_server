import React from "react"
import { Input, Typography } from "antd"
import { useTranslation } from "react-i18next"

import { DeckStudyDefaultsFields } from "./DeckStudyDefaultsFields"
import { DeckSchedulerSettingsEditor } from "./DeckSchedulerSettingsEditor"
import type { DeckReviewPromptSide } from "@/services/flashcards"
import type { DeckSchedulerDraftState } from "../hooks/useDeckSchedulerDraft"

const { Text } = Typography

type NewDeckConfigurationFieldsProps = {
  deckName: string
  onDeckNameChange: (value: string) => void
  schedulerDraft: DeckSchedulerDraftState
  nameTestId: string
  hint?: string | null
  reviewPromptSide?: DeckReviewPromptSide
  onReviewPromptSideChange?: (value: DeckReviewPromptSide) => void
}

export const NewDeckConfigurationFields: React.FC<NewDeckConfigurationFieldsProps> = ({
  deckName,
  onDeckNameChange,
  schedulerDraft,
  nameTestId,
  hint = null,
  reviewPromptSide,
  onReviewPromptSideChange
}) => {
  const { t } = useTranslation(["option"])
  const [localReviewPromptSide, setLocalReviewPromptSide] = React.useState<DeckReviewPromptSide>(
    reviewPromptSide ?? "front"
  )

  React.useEffect(() => {
    if (reviewPromptSide) {
      setLocalReviewPromptSide(reviewPromptSide)
    }
  }, [reviewPromptSide])

  const effectiveReviewPromptSide = reviewPromptSide ?? localReviewPromptSide
  const handleReviewPromptSideChange =
    onReviewPromptSideChange ?? setLocalReviewPromptSide

  return (
    <div className="space-y-3 rounded border border-border bg-muted/10 p-3">
      <label className="flex flex-col gap-1">
        <Text strong>
          {t("option:flashcards.newDeckNameLabel", {
            defaultValue: "New deck name"
          })}
        </Text>
        <Input
          value={deckName}
          onChange={(event) => onDeckNameChange(event.target.value)}
          placeholder={t("option:flashcards.newDeckNamePlaceholder", {
            defaultValue: "New deck name"
          })}
          data-testid={nameTestId}
        />
      </label>

      <DeckStudyDefaultsFields
        reviewPromptSide={effectiveReviewPromptSide}
        onReviewPromptSideChange={handleReviewPromptSideChange}
      />

      <DeckSchedulerSettingsEditor
        schedulerDraft={schedulerDraft}
      />

      {hint ? (
        <Text type="secondary" className="block text-xs">
          {hint}
        </Text>
      ) : null}
    </div>
  )
}

export default NewDeckConfigurationFields
