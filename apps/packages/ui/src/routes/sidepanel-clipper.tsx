import React from "react"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useServerOnline } from "@/hooks/useServerOnline"
import { useTranslation } from "react-i18next"
import { SidepanelHeaderSimple } from "~/components/Sidepanel/Chat/SidepanelHeaderSimple"
import {
  CLIPPER_PENDING_DRAFT_EVENT,
  clearPendingClipDraft,
  readPendingClipDraft,
  type PendingClipDraft
} from "@/services/web-clipper/pending-draft"
import WebClipperPanel from "@/components/Sidepanel/Clipper/WebClipperPanel"

const SidepanelClipper = () => {
  const { t } = useTranslation()
  const isOnline = useServerOnline()
  const { capabilities, loading: capsLoading } = useServerCapabilities()
  const [draft, setDraft] = React.useState<PendingClipDraft | null>(() =>
    readPendingClipDraft()
  )

  React.useEffect(() => {
    const handlePendingDraft = (event: Event) => {
      const nextDraft =
        (event as CustomEvent<PendingClipDraft | null>).detail ||
        readPendingClipDraft()
      setDraft(nextDraft)
    }

    window.addEventListener(CLIPPER_PENDING_DRAFT_EVENT, handlePendingDraft)
    return () => {
      window.removeEventListener(
        CLIPPER_PENDING_DRAFT_EVENT,
        handlePendingDraft
      )
    }
  }, [])

  return (
    <RouteErrorBoundary
      routeId="sidepanel-clipper"
      routeLabel={t("sidepanel:clipper.routeLabel", "Clipper")}
    >
      <div
        className="min-h-screen bg-surface text-text"
        data-testid="sidepanel-clipper-root"
      >
        <SidepanelHeaderSimple
          activeTitle={t("sidepanel:clipper.routeLabel", "Clipper")}
        />
        <div className="space-y-4 px-3 pb-4 pt-14">
          {!isOnline ? (
            <FeatureEmptyState
              title={t(
                "sidepanel:clipper.offlineTitle",
                "Connect to use the web clipper"
              )}
              description={t(
                "sidepanel:clipper.offlineDescription",
                "The review sheet needs a reachable tldw server before it can save clips."
              )}
            />
          ) : capsLoading ? (
            <div className="rounded-lg border border-border bg-surface2 p-4 text-sm text-text-muted">
              {t(
                "sidepanel:clipper.loadingCapabilities",
                "Checking clipper support..."
              )}
            </div>
          ) : !capabilities?.hasWebClipper ? (
            <FeatureEmptyState
              title={t(
                "sidepanel:clipper.unavailableTitle",
                "Web clipper unavailable"
              )}
              description={t(
                "sidepanel:clipper.unavailableDescription",
                "This server does not advertise web clipper support."
              )}
            />
          ) : !draft ? (
            <FeatureEmptyState
              title={t(
                "sidepanel:clipper.noDraftTitle",
                "No clip is ready to review yet."
              )}
              description={t(
                "sidepanel:clipper.noDraftDescription",
                "Capture a page from the browser context menu, then return here to file it."
              )}
            />
          ) : (
            <WebClipperPanel
              draft={draft}
              onCancel={() => {
                clearPendingClipDraft(draft.clipId)
                setDraft(null)
              }}
            />
          )}
        </div>
      </div>
    </RouteErrorBoundary>
  )
}

export default SidepanelClipper
