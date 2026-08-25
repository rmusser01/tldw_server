import React from "react"
import { Link } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { useMobile } from "@/hooks/useMediaQuery"
import { SharedWorkspaceChatPane } from "./SharedWorkspaceChatPane"
import { SharedWorkspaceHeader } from "./SharedWorkspaceHeader"
import { SharedWorkspacePreview } from "./SharedWorkspacePreview"
import { SharedWorkspaceSourcesPane } from "./SharedWorkspaceSourcesPane"
import { useSharedResearchWorkspace } from "./useSharedResearchWorkspace"

type SharedResearchWorkspaceProps = {
  shareId?: number
  invalidRoute?: boolean
}

type UnavailableProps = {
  detail?: string
  focus?: boolean
}

const SharedWorkspaceUnavailable: React.FC<UnavailableProps> = ({
  detail,
  focus = false
}) => {
  const { t } = useTranslation("playground")
  const headingRef = React.useRef<HTMLHeadingElement>(null)

  React.useEffect(() => {
    if (focus) headingRef.current?.focus()
  }, [focus])

  return (
    <main className="flex h-full min-h-0 w-full flex-1 items-center justify-center bg-bg p-6 text-center text-text">
      <div className="max-w-md space-y-3">
        <h1 ref={headingRef} tabIndex={-1} className="text-xl font-semibold">
          {t(
            "sharedWorkspace.unavailable",
            "This shared workspace isn't available."
          )}
        </h1>
        {detail ? <p className="text-sm text-text-muted">{detail}</p> : null}
        <Link
          to="/shared-with-me"
          className="inline-flex min-h-11 items-center rounded-md px-3 text-sm font-medium text-primary outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2"
        >
          {t(
            "sharedWorkspace.returnToShared",
            "Return to Shared with me"
          )}
        </Link>
      </div>
    </main>
  )
}

const LoadedSharedWorkspace: React.FC<{ shareId: number }> = ({ shareId }) => {
  const { t } = useTranslation("playground")
  const controller = useSharedResearchWorkspace(shareId)
  const { previewSource, state } = controller
  const isMobile = useMobile()
  const headingRef = React.useRef<HTMLHeadingElement>(null)
  const previewTriggerRef = React.useRef<HTMLElement | null>(null)
  const [activePane, setActivePane] = React.useState<"sources" | "chat">(
    "sources"
  )
  const [previewOpen, setPreviewOpen] = React.useState(false)
  const canInspectSources =
    state.allowedActions?.inspect_sources.allowed === true

  React.useEffect(() => {
    if (state.status === "loaded") headingRef.current?.focus()
  }, [state.status])

  const openPreview = React.useCallback(
    (sourceId: string, chunkIndex: number | undefined, trigger: HTMLElement) => {
      if (!canInspectSources) return
      previewTriggerRef.current = trigger
      setPreviewOpen(true)
      void previewSource(sourceId, chunkIndex)
    },
    [canInspectSources, previewSource]
  )

  const closePreview = React.useCallback(() => {
    setPreviewOpen(false)
    globalThis.setTimeout(() => previewTriggerRef.current?.focus(), 0)
  }, [])

  const activateMobilePane = (
    pane: "sources" | "chat",
    moveFocus = false
  ) => {
    setActivePane(pane)
    if (moveFocus) {
      document.getElementById(`shared-workspace-${pane}-tab`)?.focus()
    }
  }

  if (state.status === "loading") {
    return (
      <main
        className="flex h-full min-h-0 w-full flex-1 items-center justify-center bg-bg p-6 text-text"
        aria-live="polite"
      >
        <h1 className="text-lg font-semibold">
          {t("sharedWorkspace.loading", "Loading shared workspace")}
        </h1>
      </main>
    )
  }

  if (state.status !== "loaded" || !state.bootstrap) {
    return (
      <SharedWorkspaceUnavailable
        detail={
          state.status === "unavailable"
            ? t(
                "sharedWorkspace.unavailableDetail",
                "Shared workspace access is temporarily unavailable."
              )
            : undefined
        }
        focus
      />
    )
  }

  const sourcesPane = (
    <SharedWorkspaceSourcesPane
      controller={controller}
      onPreview={openPreview}
    />
  )
  const chatPane = (
    <SharedWorkspaceChatPane
      controller={controller}
      onPreviewCitation={openPreview}
    />
  )

  return (
    <main
      data-testid="shared-workspace-shell"
      className="flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-bg text-text"
    >
      <SharedWorkspaceHeader
        allowedActions={state.allowedActions}
        bootstrap={state.bootstrap}
        headingRef={headingRef}
      />

      {isMobile ? (
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
          <div
            role="tablist"
            aria-label={t(
              "sharedWorkspace.panesLabel",
              "Shared workspace panes"
            )}
            className="grid h-11 shrink-0 grid-cols-2 border-b border-border bg-surface px-2"
          >
            {(["sources", "chat"] as const).map((pane) => (
              <button
                key={pane}
                type="button"
                role="tab"
                aria-selected={activePane === pane}
                aria-controls={"shared-workspace-" + pane + "-panel"}
                id={"shared-workspace-" + pane + "-tab"}
                tabIndex={activePane === pane ? 0 : -1}
                onClick={() => activateMobilePane(pane)}
                onKeyDown={(event) => {
                  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
                    return
                  }
                  event.preventDefault()
                  activateMobilePane(pane === "sources" ? "chat" : "sources", true)
                }}
                className="h-11 border-b-2 border-transparent px-3 text-sm font-medium capitalize outline-none aria-selected:border-primary aria-selected:text-primary focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-focus"
              >
                {pane === "sources"
                  ? t("sharedWorkspace.sources", "Sources")
                  : t("sharedWorkspace.chat", "Chat")}
              </button>
            ))}
          </div>
          <section
            id="shared-workspace-sources-panel"
            role="tabpanel"
            aria-labelledby="shared-workspace-sources-tab"
            hidden={activePane !== "sources"}
            className="min-h-0 min-w-0 flex-1 overflow-hidden"
          >
            {sourcesPane}
          </section>
          <section
            id="shared-workspace-chat-panel"
            role="tabpanel"
            aria-labelledby="shared-workspace-chat-tab"
            hidden={activePane !== "chat"}
            className="min-h-0 min-w-0 flex-1 overflow-hidden"
          >
            {chatPane}
          </section>
        </div>
      ) : (
        <div
          data-testid="shared-workspace-desktop-grid"
          className="grid min-h-0 min-w-0 flex-1 grid-cols-[minmax(18rem,0.72fr)_minmax(0,1.28fr)] overflow-hidden"
        >
          {sourcesPane}
          {chatPane}
        </div>
      )}

      <SharedWorkspacePreview
        error={state.errors.preview}
        isMobile={isMobile}
        loading={state.previewLoading}
        onClose={closePreview}
        open={previewOpen}
        preview={state.preview}
      />
    </main>
  )
}

export const SharedResearchWorkspace: React.FC<
  SharedResearchWorkspaceProps
> = ({ shareId, invalidRoute = false }) => {
  if (invalidRoute || shareId === undefined) {
    return <SharedWorkspaceUnavailable focus />
  }

  return <LoadedSharedWorkspace shareId={shareId} />
}

export default SharedResearchWorkspace
