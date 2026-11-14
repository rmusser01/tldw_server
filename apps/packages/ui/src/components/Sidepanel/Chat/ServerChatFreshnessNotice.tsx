import React from "react"
import { RefreshCw, X } from "lucide-react"
import { useTranslation } from "react-i18next"

export type FreshnessNoticeKind =
  /** Newer turns exist, but the tab kept messages that exist only here. */
  | "stale"
  /** The tab was refreshed to the server's latest message. */
  | "refreshed"
  /** The tab was refreshed just before a send, which now follows the newer turns. */
  | "refreshedBeforeSend"
  /** The tab couldn't be refreshed, so the send was held. */
  | "sendBlocked"

export type FreshnessNotice = {
  tabId: string
  kind: FreshnessNoticeKind
  newCount: number
}

type Props = {
  notice: FreshnessNotice
  onRefresh: () => void
  onDismiss: () => void
}

/**
 * Tells the user that this tab's server chat changed in another window or tab
 * (XP-08): either the tab now shows the newer messages, or it kept messages that
 * exist only here and offers Refresh, or a send was held because the tab could
 * not be refreshed.
 */
export const ServerChatFreshnessNotice = ({ notice, onRefresh, onDismiss }: Props) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const values = { count: notice.newCount }
  const message = (() => {
    switch (notice.kind) {
      case "stale":
        return t("sidepanel:freshness.stale", {
          defaultValue:
            "Updated in another window or tab. Newer messages ({{count}} new) aren't shown here yet.",
          ...values
        })
      case "refreshed":
        return t("sidepanel:freshness.refreshed", {
          defaultValue:
            "Updated in another window or tab. Showing the latest messages ({{count}} new).",
          ...values
        })
      case "refreshedBeforeSend":
        return t("sidepanel:freshness.refreshedBeforeSend", {
          defaultValue:
            "Updated in another window or tab. The latest messages ({{count}} new) were loaded first, and your message continues from them.",
          ...values
        })
      case "sendBlocked":
        return t("sidepanel:freshness.sendBlocked", {
          defaultValue:
            "Updated in another window or tab, and the newer messages couldn't be loaded. Your message wasn't sent; it's back in the composer. Refresh, then send it again.",
          ...values
        })
    }
  })()
  const offersRefresh = notice.kind === "stale" || notice.kind === "sendBlocked"
  return (
    <div
      role={notice.kind === "sendBlocked" ? "alert" : "status"}
      aria-label={t("sidepanel:freshness.label", "Chat update")}
      data-testid="sidepanel-chat-freshness-notice"
      className="mx-auto mb-2 flex w-full max-w-5xl items-start gap-2 rounded-md border border-border bg-surface2 px-3 py-2 text-xs text-text">
      <p className="min-w-0 flex-1">{message}</p>
      {offersRefresh && (
        <button
          type="button"
          onClick={onRefresh}
          className="inline-flex flex-shrink-0 items-center gap-1 rounded-md border border-border px-2 py-0.5 text-xs text-text hover:bg-surface">
          <RefreshCw className="size-3" aria-hidden="true" />
          {t("sidepanel:freshness.refresh", "Refresh")}
        </button>
      )}
      <button
        type="button"
        onClick={onDismiss}
        aria-label={t("common:dismiss", "Dismiss")}
        title={t("common:dismiss", "Dismiss")}
        className="flex-shrink-0 rounded p-0.5 text-text-subtle hover:bg-surface hover:text-text">
        <X className="size-3" aria-hidden="true" />
      </button>
    </div>
  )
}
