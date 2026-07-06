import { Dropdown, Tooltip, Button, type MenuProps } from "antd"
import { Pen, MoreHorizontal, PlayCircle, Copy, FolderInput } from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"

import type { Flashcard } from "@/services/flashcards"

interface FlashcardActionsMenuProps {
  card: Flashcard
  disabled?: boolean
  onEdit: () => void
  onReview: () => void
  onDuplicate: () => void
  onMove: () => void
}

export const FlashcardActionsMenu: React.FC<FlashcardActionsMenuProps> = ({
  card,
  disabled = false,
  onEdit,
  onReview,
  onDuplicate,
  onMove
}) => {
  const { t } = useTranslation(["option", "common"])

  const menuItems: MenuProps["items"] = [
    {
      key: "review",
      label: t("option:flashcards.reviewCard", { defaultValue: "Review now" }),
      icon: <PlayCircle className="size-4" />,
      disabled,
      onClick: onReview
    },
    {
      key: "duplicate",
      label: t("option:flashcards.duplicate", { defaultValue: "Duplicate" }),
      icon: <Copy className="size-4" />,
      disabled,
      onClick: onDuplicate
    },
    {
      key: "move",
      label: t("option:flashcards.moveToDeck", { defaultValue: "Move to deck" }),
      icon: <FolderInput className="size-4" />,
      disabled,
      onClick: onMove
    }
  ]

  const handleEditPointerDown = React.useCallback(
    (event: React.PointerEvent<HTMLButtonElement>) => {
      if (event.button !== 0 || disabled) return
      event.preventDefault()
      event.stopPropagation()
      onEdit()
    },
    [disabled, onEdit]
  )

  const handleEditClick = React.useCallback(
    (event: React.MouseEvent<HTMLButtonElement>) => {
      event.stopPropagation()
      onEdit()
    },
    [onEdit]
  )

  return (
    <div className="flex items-center gap-1">
      <Tooltip title={t("common:edit", { defaultValue: "Edit" })}>
        <Button
          type="text"
          size="small"
          icon={<Pen className="size-4" />}
          onPointerDown={handleEditPointerDown}
          onClick={handleEditClick}
          disabled={disabled}
          aria-label={t("common:edit", { defaultValue: "Edit" })}
          data-testid={`flashcard-edit-${card.uuid}`}
        />
      </Tooltip>
      <Dropdown
        menu={{ items: menuItems }}
        trigger={["click"]}
        placement="bottomRight"
        aria-label={t("option:header.moreActions", { defaultValue: "More actions" })}
      >
        <Button
          type="text"
          size="small"
          icon={<MoreHorizontal className="size-4" />}
          onClick={(e) => e.stopPropagation()}
          disabled={disabled}
          aria-label={t("option:header.moreActions", { defaultValue: "More actions" })}
          data-testid={`flashcard-more-${card.uuid}`}
        />
      </Dropdown>
    </div>
  )
}
