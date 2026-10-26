import React from "react"
import { useTranslation } from "react-i18next"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"

/**
 * Ask the server chat loader to load the selected chat again. A new selection
 * intent for the same id is a deliberate reload, so the loader retries even
 * though it already tried (and failed) to load this chat.
 */
export const retryServerChatLoad = (serverChatId: string) => {
  const state = useStoreMessageOption.getState()
  if (state.serverChatId !== serverChatId) return false
  state.setServerChatLoadError(null)
  state.setServerChatLoadState("loading")
  usePlaygroundSessionStore.getState().requestServerChatSelection(serverChatId)
  return true
}

type ServerChatLoadFailurePanelProps = {
  serverChatId: string
  /** The raw loader error (often a request line); shown only as a detail. */
  technicalDetail: string | null | undefined
}

/**
 * Shown in place of the transcript when a saved chat fails to load (CS-06).
 * The raw request error is kept behind "Technical details" so it can't
 * overflow the panel, and the user gets Retry and a way to pick another chat.
 */
export const ServerChatLoadFailurePanel: React.FC<ServerChatLoadFailurePanelProps> = ({
  serverChatId,
  technicalDetail
}) => {
  const { t } = useTranslation(["playground"])
  const detail = technicalDetail?.trim() || null

  return (
    <div className="mt-32 w-full px-6">
      <div
        role="alert"
        data-testid="server-chat-load-failure"
        className="mx-auto max-w-xl rounded-xl border border-destructive/30 bg-destructive/5 px-5 py-4 text-center text-sm text-text"
      >
        <p className="font-medium">
          {t(
            "playground:serverChatLoadFailure.title",
            "Couldn't load this conversation."
          )}
        </p>
        <p className="mt-1 text-text-muted">
          {t(
            "playground:serverChatLoadFailure.body",
            "Your messages are safe on the server."
          )}
        </p>
        <div className="mt-3 flex flex-wrap items-center justify-center gap-2">
          <button
            type="button"
            className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-white hover:bg-primaryStrong focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
            onClick={() => {
              retryServerChatLoad(serverChatId)
            }}
          >
            {t("playground:serverChatLoadFailure.retry", "Retry")}
          </button>
          <button
            type="button"
            className="rounded-md border border-border px-3 py-1.5 text-sm text-text hover:bg-surface2 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
            onClick={() => {
              window.dispatchEvent(new Event("tldw:open-chat-sidebar"))
            }}
          >
            {t("playground:serverChatLoadFailure.openAnother", "Open another chat")}
          </button>
        </div>
        {detail ? (
          <details className="mt-3 text-left text-xs text-text-muted">
            <summary className="cursor-pointer select-none">
              {t("playground:serverChatLoadFailure.details", "Technical details")}
            </summary>
            <p className="mt-1 break-all font-mono">{detail}</p>
          </details>
        ) : null}
      </div>
    </div>
  )
}
