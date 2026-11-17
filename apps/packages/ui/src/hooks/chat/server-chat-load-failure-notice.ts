import type { TFunction } from "i18next"

type NotificationApi = {
  error: (payload: { message: string; description?: string; key?: string }) => void
}

/**
 * Tell the user a saved chat failed to load (CS-06).
 *
 * With nothing on screen, the transcript shows a "Couldn't load this
 * conversation" panel with Retry, so a toast would only repeat it. When
 * messages are already shown, a toast is the only signal; it uses plain words
 * and leaves the raw request error to the panel's technical details.
 */
export const notifyServerChatLoadFailure = ({
  notification,
  t,
  messagesShown
}: {
  notification: NotificationApi
  t: TFunction
  messagesShown: number
}) => {
  if (messagesShown === 0) return false
  notification.error({
    key: "server-chat-load-failure",
    message: String(
      t("playground:serverChatLoadFailure.title", {
        defaultValue: "Couldn't load this conversation."
      })
    ),
    description: String(
      t("playground:serverChatLoadFailure.toastBody", {
        defaultValue:
          "Your messages are safe on the server. Reopen the chat to try again."
      })
    )
  })
  return true
}
