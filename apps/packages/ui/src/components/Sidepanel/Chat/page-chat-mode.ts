import React from "react"
import { useTranslation } from "react-i18next"
import { useAntdNotification } from "@/hooks/useAntdNotification"

/**
 * The side panel's "rag" chat mode answers with the current browser page as
 * context. It changes every later send, with or without page embeddings, so the
 * composer must show it whenever it is on (XS-08).
 */
export const isChatWithPageActive = (chatMode: string | null | undefined) =>
  chatMode === "rag"

type PageChatMode = "normal" | "rag" | "vision"

/**
 * Toggle "Chat with current page" and say so. The keyboard shortcut and the
 * command palette used to flip this mode silently.
 */
export const usePageChatModeToggle = <Mode extends PageChatMode>(
  chatMode: Mode,
  setChatMode: (mode: Mode) => void
) => {
  const { t } = useTranslation(["sidepanel"])
  const notification = useAntdNotification()

  return React.useCallback(() => {
    const turningOn = !isChatWithPageActive(chatMode)
    setChatMode((turningOn ? "rag" : "normal") as Mode)
    notification.info({
      key: "sidepanel-chat-with-page",
      message: turningOn
        ? t("sidepanel:chatWithPage.on", "Chat with current page is on")
        : t("sidepanel:chatWithPage.off", "Chat with current page is off"),
      description: turningOn
        ? t(
            "sidepanel:chatWithPage.onDescription",
            "Your next messages are answered using the page open in this tab."
          )
        : undefined
    })
  }, [chatMode, notification, setChatMode, t])
}
