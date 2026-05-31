import React from "react"
import { Button, Typography } from "antd"
import { useTranslation } from "react-i18next"

const { Text } = Typography

interface StudyPackPanelProps {
  onLaunch: () => void
}

export const StudyPackPanel: React.FC<StudyPackPanelProps> = ({ onLaunch }) => {
  const { t } = useTranslation(["option", "common"])

  return (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div className="max-w-3xl space-y-1">
        <Text strong className="block">
          {t("option:flashcards.studyPackLauncherHeadline", {
            defaultValue: "Turn media or notes into a review queue."
          })}
        </Text>
        <Text type="secondary">
          {t("option:flashcards.studyPackLauncherBody", {
            defaultValue:
              "Create a study pack from supported sources, then review the generated deck in Flashcards."
          })}
        </Text>
      </div>
      <Button
        type="primary"
        onClick={onLaunch}
        data-testid="study-pack-launcher-button"
      >
        {t("option:flashcards.studyPackLaunchButton", {
          defaultValue: "Create study pack"
        })}
      </Button>
    </div>
  )
}
