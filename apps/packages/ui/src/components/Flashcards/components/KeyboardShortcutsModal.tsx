import React from "react"
import { Modal, Typography } from "antd"
import { useTranslation } from "react-i18next"

const { Text, Title } = Typography

interface ShortcutItem {
  keys: string[]
  description: string
}

interface ShortcutGroup {
  title: string
  shortcuts: ShortcutItem[]
}

interface KeyboardShortcutsModalProps {
  open: boolean
  onClose: () => void
  activeTab: "review" | "cards" | "import" | "scheduler" | "templates"
}

export const KeyboardShortcutsModal: React.FC<KeyboardShortcutsModalProps> = ({
  open,
  onClose,
  activeTab
}) => {
  const { t } = useTranslation(["option"])

  const reviewShortcuts: ShortcutGroup = {
    title: t("option:flashcards.review", { defaultValue: "Review" }),
    shortcuts: [
      {
        keys: ["Space"],
        description: t("option:flashcards.shortcutSpace", { defaultValue: "Flip card" })
      },
      {
        keys: ["1", "2", "3", "4"],
        description: t("option:flashcards.shortcutRateKeys", { defaultValue: "Rate card" })
      },
      {
        keys: ["Ctrl+Z", "⌘Z"],
        description: t("option:flashcards.shortcutUndo", { defaultValue: "Undo last rating" })
      }
    ]
  }

  const cardsShortcuts: ShortcutGroup = {
    title: t("option:flashcards.cards", { defaultValue: "Cards" }),
    shortcuts: [
      {
        keys: ["j", "↓"],
        description: t("option:flashcards.shortcutDown", { defaultValue: "Move down" })
      },
      {
        keys: ["k", "↑"],
        description: t("option:flashcards.shortcutUp", { defaultValue: "Move up" })
      },
      {
        keys: ["Enter"],
        description: t("option:flashcards.shortcutEnter", { defaultValue: "Edit card" })
      },
      {
        keys: ["Space"],
        description: t("option:flashcards.shortcutSelectCard", { defaultValue: "Select card" })
      },
      {
        keys: ["Esc"],
        description: t("option:flashcards.shortcutEsc", { defaultValue: "Clear focus" })
      }
    ]
  }

  const globalShortcuts: ShortcutGroup = {
    title: t("option:flashcards.global", { defaultValue: "Global" }),
    shortcuts: [
      {
        keys: ["?"],
        description: t("option:flashcards.shortcutHelp", { defaultValue: "Show shortcuts" })
      }
    ]
  }

  const groups: ShortcutGroup[] = [
    ...(activeTab === "review" ? [reviewShortcuts] : []),
    ...(activeTab === "cards" ? [cardsShortcuts] : []),
    globalShortcuts
  ]

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      title={
        <Title level={5} className="!mb-0">
          {t("option:flashcards.keyboardShortcutsTitle", {
            defaultValue: "Keyboard Shortcuts"
          })}
        </Title>
      }
      width={400}
    >
      <div className="space-y-4 pt-2">
        {groups.map((group) => (
          <div key={group.title}>
            <Text strong className="text-sm text-text-muted block mb-2">
              {group.title}
            </Text>
            <div className="space-y-2">
              {group.shortcuts.map((shortcut, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between py-1"
                >
                  <Text className="text-sm">{shortcut.description}</Text>
                  <div className="flex gap-1">
                    {shortcut.keys.map((key, keyIdx) => (
                      <React.Fragment key={keyIdx}>
                        <kbd className="px-2 py-1 text-xs bg-surface2 border border-border rounded font-mono">
                          {key}
                        </kbd>
                        {keyIdx < shortcut.keys.length - 1 && (
                          <span className="text-text-muted text-xs">/</span>
                        )}
                      </React.Fragment>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
        {activeTab === "review" && (
          <div className="rounded border border-border bg-surface p-3">
            <Text strong className="text-sm block mb-2">
              {t("option:flashcards.ratingScaleTitle", {
                defaultValue: "Rating Scale (SM-2 values)"
              })}
            </Text>
            <Text className="text-xs block">
              {t("option:flashcards.ratingScaleAgain", {
                defaultValue: "Again = 0 (forgot it, repeat very soon)"
              })}
            </Text>
            <Text className="text-xs block">
              {t("option:flashcards.ratingScaleHard", {
                defaultValue: "Hard = 2 (remembered with strain, short gap)"
              })}
            </Text>
            <Text className="text-xs block">
              {t("option:flashcards.ratingScaleGood", {
                defaultValue: "Good = 3 (normal recall, default step)"
              })}
            </Text>
            <Text className="text-xs block">
              {t("option:flashcards.ratingScaleEasy", {
                defaultValue: "Easy = 5 (effortless recall, longest jump)"
              })}
            </Text>
          </div>
        )}
      </div>
    </Modal>
  )
}

export default KeyboardShortcutsModal
