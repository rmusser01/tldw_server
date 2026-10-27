import React from "react"
import { useTranslation } from "react-i18next"
import { Link } from "react-router-dom"
import { MessageSquare } from "lucide-react"

import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { CompanionHomeShell } from "@/components/Option/CompanionHome"
import { SidepanelHeaderSimple } from "~/components/Sidepanel/Chat/SidepanelHeaderSimple"
import {
  isCompanionConsentRequiredError,
  recordExplicitCompanionCapture
} from "@/services/companion"
import {
  buildExplicitCompanionCapture,
  clearPendingCompanionCapture,
  COMPANION_PENDING_CAPTURE_EVENT,
  readPendingCompanionCapture,
  type PendingCompanionCapture
} from "@/services/companion-capture"

const SidepanelCompanion = () => {
  const { t } = useTranslation(["option", "sidepanel"])
  const [banner, setBanner] = React.useState<string | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const lastHandledCaptureIdRef = React.useRef<string | null>(null)

  const handleCapture = React.useCallback(async (capture: PendingCompanionCapture) => {
    if (!capture.id || lastHandledCaptureIdRef.current === capture.id) {
      return
    }
    lastHandledCaptureIdRef.current = capture.id
    setBanner(null)
    setError(null)

    try {
      await recordExplicitCompanionCapture(buildExplicitCompanionCapture(capture))
      clearPendingCompanionCapture(capture.id)
      setBanner("Saved selection to companion.")
    } catch (caught) {
      const requestError = caught as Error & { status?: number }
      if (isCompanionConsentRequiredError(requestError)) {
        setError("Enable personalization before saving to companion.")
        lastHandledCaptureIdRef.current = null
        return
      }
      if (requestError?.status === 409) {
        clearPendingCompanionCapture(capture.id)
        setBanner("Selection already saved to companion.")
        return
      }
      setError("Failed to save selection to companion.")
      lastHandledCaptureIdRef.current = null
    }
  }, [])

  const retryPendingCapture = React.useCallback(() => {
    const pending = readPendingCompanionCapture()
    if (pending) {
      void handleCapture(pending)
    }
  }, [handleCapture])

  React.useEffect(() => {
    retryPendingCapture()

    const listener = (event: Event) => {
      const capture = (event as CustomEvent<PendingCompanionCapture>).detail
      if (!capture) return
      void handleCapture(capture)
    }

    window.addEventListener(COMPANION_PENDING_CAPTURE_EVENT, listener)
    return () => {
      window.removeEventListener(COMPANION_PENDING_CAPTURE_EVENT, listener)
    }
  }, [handleCapture, retryPendingCapture])

  return (
    <RouteErrorBoundary routeId="sidepanel-companion" routeLabel="Companion">
      <div
        className="min-h-screen bg-surface text-text"
        data-testid="sidepanel-companion-root"
      >
        <SidepanelHeaderSimple activeTitle={t("option:header.companion", "Companion")} />
        <div className="pt-11">
          {/* Chat stays one tap away above the dashboard (XS-17). */}
          <div className="px-3 pt-3">
            <Link
              to="/?view=chat"
              className="inline-flex min-h-[44px] items-center gap-2 rounded-lg border border-border bg-bg px-3 text-sm font-medium text-text transition-colors hover:bg-surface-hover"
            >
              <MessageSquare className="h-4 w-4" aria-hidden="true" />
              {t("sidepanel:companion.openChat", "Open chat")}
            </Link>
          </div>
          {banner ? (
            <div className="px-3 pt-3">
              <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-800">
                {banner}
              </div>
            </div>
          ) : null}
          {error ? (
            <div className="px-3 pt-3">
              <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-800">
                {error}
              </div>
            </div>
          ) : null}
          <CompanionHomeShell
            onPersonalizationEnabled={retryPendingCapture}
            surface="sidepanel"
          />
        </div>
      </div>
    </RouteErrorBoundary>
  )
}

export default SidepanelCompanion
