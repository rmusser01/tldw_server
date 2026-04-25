import React from "react"
import { Select, Typography } from "antd"
import { useTranslation } from "react-i18next"

import type { DeckReviewPromptSide } from "@/services/flashcards"

const { Text } = Typography

export type DeckStudyDefaultsFieldsProps = {
  reviewPromptSide: DeckReviewPromptSide
  onReviewPromptSideChange: (value: DeckReviewPromptSide) => void
}

export const DeckStudyDefaultsFields: React.FC<DeckStudyDefaultsFieldsProps> = ({
  reviewPromptSide,
  onReviewPromptSideChange
}) => {
  const { t } = useTranslation(["option"])

  return (
    <label className="flex flex-col gap-1">
      <Text strong>
        {t("option:flashcards.reviewPromptSideLabel", {
          defaultValue: "Review prompt side"
        })}
      </Text>
      <Select<DeckReviewPromptSide>
        value={reviewPromptSide}
        onChange={onReviewPromptSideChange}
        aria-label={t("option:flashcards.reviewPromptSideLabel", {
          defaultValue: "Review prompt side"
        })}
        options={[
          {
            value: "front",
            label: t("option:flashcards.reviewPromptSideFront", {
              defaultValue: "Front first"
            })
          },
          {
            value: "back",
            label: t("option:flashcards.reviewPromptSideBack", {
              defaultValue: "Back first"
            })
          }
        ]}
        data-testid="deck-study-defaults-review-prompt-side"
      />
    </label>
  )
}

export default DeckStudyDefaultsFields
