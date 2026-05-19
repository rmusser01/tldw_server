import React from "react"
import { useTranslation } from "react-i18next"
import { StopCircle } from "lucide-react"
import { Badge, type BadgeVariant } from "@/components/ui/primitives"

interface VoiceChatIndicatorProps {
  state: "idle" | "connecting" | "listening" | "thinking" | "speaking" | "error"
  statusLabel: string
  onStop: () => void
}

/**
 * Floating indicator that shows the current voice chat state.
 * Appears when voice chat is active and provides a quick way to stop.
 */
export const VoiceChatIndicator: React.FC<VoiceChatIndicatorProps> = ({
  state,
  statusLabel,
  onStop
}) => {
  const { t } = useTranslation(["playground"])

  if (state === "idle") return null

  const emoji: Record<string, string> = {
    connecting: "\uD83D\uDD0C",
    listening: "\uD83C\uDFA4",
    thinking: "\uD83D\uDCAD",
    speaking: "\uD83D\uDD0A",
    error: "\u26A0\uFE0F"
  }

  const stateEmoji = emoji[state] || ""

  const badgeVariant: BadgeVariant = (() => {
    switch (state) {
      case "listening":
        return "success"
      case "error":
        return "danger"
      case "connecting":
      case "thinking":
      case "speaking":
      default:
        return "primary"
    }
  })()

  return (
    <div className="fixed bottom-20 right-4 z-50 flex items-center gap-2 rounded-full bg-surface border border-border px-2 py-1.5 shadow-lg">
      <Badge
        data-testid="voice-chat-status-badge"
        variant={badgeVariant}
        size="md"
        className="gap-1.5"
      >
        <span className="text-sm" aria-hidden="true">{stateEmoji}</span>
        <span>{statusLabel}</span>
      </Badge>
      <button
        type="button"
        onClick={onStop}
        className="ml-2 rounded-full p-1 hover:bg-surface2 transition"
        aria-label={t("playground:voiceChat.stop", "Stop voice chat")}
        title={t("playground:voiceChat.stop", "Stop voice chat") as string}
      >
        <StopCircle className="h-4 w-4 text-danger" />
      </button>
    </div>
  )
}
