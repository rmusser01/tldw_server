import React from "react"
import { useTranslation } from "react-i18next"

type ModeTranslator = (key: string, defaultValue: string) => string

/** History eligibility is a mode, never a receipt for the latest write. */
export const getPersistenceModeLabel = (
  t: ModeTranslator,
  temporaryChat: boolean
) =>
  temporaryChat
    ? t(
        "playground:composer.persistence.temporaryModeHelp",
        "New messages are not added to saved chat history. Draft text may remain on this device."
      )
    : t(
        "playground:composer.persistence.historyModeHelp",
        "New messages can be stored in chat history."
      )

export const getPersistenceModePillLabel = (
  t: ModeTranslator,
  temporaryChat: boolean
) =>
  temporaryChat
    ? t("playground:composer.persistence.temporaryModeTitle", "Temporary")
    : t("playground:composer.persistence.historyModeTitle", "History on")

/** Keep the history toggle's semantic subject stable and include visible state. */
export const getPersistenceModeControlLabel = (
  t: ModeTranslator,
  temporaryChat: boolean
) =>
  `${t("playground:composer.persistence.historyControl", "Chat history")}: ${getPersistenceModePillLabel(t, temporaryChat)}`

export type UsePersistenceModeParams = {
  temporaryChat: boolean
}

export function usePersistenceMode({
  temporaryChat
}: UsePersistenceModeParams) {
  const { t } = useTranslation(["playground", "common"])
  const persistenceModeLabel = getPersistenceModeLabel(t, temporaryChat)
  const persistencePillLabel = getPersistenceModePillLabel(t, temporaryChat)
  const persistenceControlLabel = getPersistenceModeControlLabel(
    t,
    temporaryChat
  )

  const persistenceTooltip = React.useMemo(
    () => (
      <div className="flex flex-col gap-0.5 text-xs">
        <span className="font-medium">{persistencePillLabel}</span>
        <span className="text-text-subtle">{persistenceModeLabel}</span>
      </div>
    ),
    [persistenceModeLabel, persistencePillLabel]
  )

  const focusConnectionCard = React.useCallback(() => {
    try {
      const card = document.getElementById("server-connection-card")
      if (card) {
        card.scrollIntoView({ block: "nearest", behavior: "smooth" })
        ;(card as HTMLElement).focus()
        return
      }
    } catch {
      // ignore DOM errors and fall through to hash navigation
    }
    try {
      const base = window.location.href.replace(/#.*$/, "") || "/options.html"
      const target = `${base}#/settings/tldw`
      window.location.href = target
    } catch {
      // ignore navigation failures
    }
  }, [])

  return {
    persistenceModeLabel,
    persistencePillLabel,
    persistenceControlLabel,
    persistenceTooltip,
    focusConnectionCard,
    /** Re-exported for callers that need the raw function (e.g. toggle handler) */
    getPersistenceModeLabel
  }
}
