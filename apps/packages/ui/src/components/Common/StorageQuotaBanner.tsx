import { useState } from "react"
import { useTranslation } from "react-i18next"
import { Alert } from "@/components/ui"
import { useStorageQuota } from "@/hooks/useStorageQuota"

const DISMISS_KEY = "tldw:storage-quota-banner-dismissed"

export function StorageQuotaBanner() {
  const { t } = useTranslation(["common"])
  const { level, ratio, usedBytes, budgetBytes } = useStorageQuota()

  // Session-scoped dismiss (sessionStorage, not localStorage -- re-shows next session)
  // Must be called before any early return to satisfy Rules of Hooks.
  const [dismissed, setDismissed] = useState(() => {
    try {
      return sessionStorage.getItem(DISMISS_KEY) === "true"
    } catch {
      return false
    }
  })

  // Don't render if storage is ok
  if (level === "ok") return null

  if (dismissed && level !== "exceeded") return null // exceeded is never dismissible

  const handleDismiss = () => {
    setDismissed(true)
    try {
      sessionStorage.setItem(DISMISS_KEY, "true")
    } catch {
      /* ignore */
    }
  }

  const usedMB = (usedBytes / (1024 * 1024)).toFixed(1)
  const budgetMB = (budgetBytes / (1024 * 1024)).toFixed(0)
  const pct = Math.round(ratio * 100)

  if (level === "exceeded") {
    return (
      <Alert
        variant="error"
        data-testid="storage-quota-banner-exceeded"
        title={t("common:storageQuota.exceededTitle", {
          defaultValue: "Storage nearly full"
        })}
      >
        {t("common:storageQuota.exceededDescription", {
          defaultValue:
            "Workspace storage is {{pct}}% full ({{usedMB}}/{{budgetMB}} MB). New data may not save. Archive or delete old workspaces to free space.",
          pct,
          usedMB,
          budgetMB
        })}
      </Alert>
    )
  }

  // warning level
  return (
    <Alert
      variant="warning"
      dismissible
      onDismiss={handleDismiss}
      dismissLabel={t("common:close", { defaultValue: "Close" })}
      data-testid="storage-quota-banner-warning"
      title={t("common:storageQuota.warningTitle", {
        defaultValue: "Storage getting full"
      })}
    >
      {t("common:storageQuota.warningDescription", {
        defaultValue:
          "Workspace storage is {{pct}}% full ({{usedMB}}/{{budgetMB}} MB). Consider archiving old workspaces.",
        pct,
        usedMB,
        budgetMB
      })}
    </Alert>
  )
}
