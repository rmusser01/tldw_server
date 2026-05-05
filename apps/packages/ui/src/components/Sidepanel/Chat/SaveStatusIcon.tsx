import { Tooltip } from "antd"
import { Save } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useEffect, useRef } from "react"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { Badge, type BadgeVariant } from "@/components/ui/primitives"

type SaveMode = "ephemeral" | "local" | "server"

interface SaveStatusIconProps {
  temporaryChat: boolean
  serverChatId?: string | null
  onClick?: () => void
}

/**
 * Save status indicator icon with color-coded state.
 *
 * Colors:
 * - Blue: Ephemeral (not saved)
 * - Gray: Saved locally
 * - Green: Saved to server
 */
export const SaveStatusIcon = ({
  temporaryChat,
  serverChatId,
  onClick
}: SaveStatusIconProps) => {
  const { t } = useTranslation(["sidepanel", "playground"])
  const notification = useAntdNotification()
  const previousServerChatIdRef = useRef<string | null | undefined>(serverChatId)

  // Monitor serverChatId for unexpected failures
  useEffect(() => {
    const previous = previousServerChatIdRef.current
    const current = serverChatId

    // If we had a server chat ID and it became null (save failed)
    if (previous && previous !== "" && !current && !temporaryChat) {
      notification.warning({
        message: t(
          "sidepanel:saveStatus.saveFailed",
          "Failed to save chat to server"
        ),
        description: t(
          "sidepanel:saveStatus.saveFailedDescription",
          "Chat is now saving locally only. Check your connection and try again."
        ),
        placement: "bottomRight",
        duration: 4
      })
    }

    previousServerChatIdRef.current = current
  }, [serverChatId, temporaryChat, notification, t])

  const saveMode: SaveMode = (() => {
    if (temporaryChat) return "ephemeral"
    if (serverChatId) return "server"
    return "local"
  })()

  const badgeVariant: BadgeVariant = (() => {
    switch (saveMode) {
      case "ephemeral":
        return "primary"
      case "server":
        return "success"
      case "local":
      default:
        return "secondary"
    }
  })()

  const tooltip = (() => {
    switch (saveMode) {
      case "ephemeral":
        return t(
          "sidepanel:saveStatus.ephemeral",
          "Ephemeral: Not saved, cleared on close"
        )
      case "server":
        return t(
          "sidepanel:saveStatus.server",
          "Saved to server"
        )
      case "local":
      default:
        return t(
          "sidepanel:saveStatus.local",
          "Saved locally in this browser"
        )
    }
  })()

  return (
    <Tooltip title={tooltip}>
      <button
        type="button"
        onClick={onClick}
        data-testid="chat-save-status"
        className="rounded p-2 hover:bg-surface2 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        aria-label={tooltip}
        title={tooltip}
      >
        <Badge
          data-testid="chat-save-status-badge"
          variant={badgeVariant}
          size="sm"
          outline
          className="gap-0 leading-none"
        >
          <Save className="size-4 transition-colors" aria-hidden="true" />
        </Badge>
      </button>
    </Tooltip>
  )
}
