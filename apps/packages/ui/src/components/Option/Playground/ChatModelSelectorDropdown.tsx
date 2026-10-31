import { ProviderIcons } from "@/components/Common/ProviderIcon"
import { Dropdown, Input, Select, Tooltip } from "antd"
import { ArrowRight, ChevronDown, HelpCircle } from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"
import { Link } from "react-router-dom"
import type { ModelSortMode } from "@/hooks/playground"
import type { ComposerModelReadiness } from "./hooks/useComposerModelReadiness"

// The search box, not the scope control's radio inputs, takes focus on open.
const MODEL_SEARCH_INPUT = 'input:not([type="radio"]):not([type="checkbox"])'

type ChatModelSelectorDropdownProps = {
  apiModelLabel: string
  connectionStatusLabel: string
  /** The server cannot take a send (offline, unconfigured); always a warning. */
  connectionStatusWarning?: boolean
  /** The first connection check is still running; named, but not a warning (CC-03). */
  connectionStatusPending?: boolean
  /**
   * The server answers but is degraded. A warning unless the model's own
   * readiness is shown, which is the more specific signal (CC-01).
   */
  connectionDegraded?: boolean
  catalogControls?: React.ReactNode
  modelUsabilityLabel?: string
  modelUsabilityTitle?: string
  modelUsabilityWarning?: boolean
  /**
   * The selected model's own readiness. When given, it replaces the server
   * connection label for a connected session, so the chip never calls a model
   * "Healthy" just because the tldw server answers (CC-01).
   */
  modelReadiness?: ComposerModelReadiness | null
  modelDropdownMenuItems: any[]
  modelDropdownOpen: boolean
  modelSearchQuery?: string
  modelSelectorWarning?: boolean
  modelSortMode?: ModelSortMode
  placement?: "topLeft" | "bottomLeft"
  resolvedProviderKey: string
  selectedModel?: string | null | undefined
  setModelDropdownOpen: (open: boolean, restoreFocus?: boolean) => void
  setModelSearchQuery: (query: string) => void
  setModelSortMode?: (mode: ModelSortMode) => void
}

export const ChatModelSelectorDropdown = React.memo(
  function ChatModelSelectorDropdown({
    apiModelLabel,
    catalogControls,
    connectionStatusLabel,
    connectionStatusWarning = false,
    connectionStatusPending = false,
    connectionDegraded = false,
    modelUsabilityLabel,
    modelUsabilityTitle,
    modelUsabilityWarning = false,
    modelReadiness = null,
    modelDropdownMenuItems,
    modelDropdownOpen,
    modelSearchQuery = "",
    modelSelectorWarning = false,
    modelSortMode = "favorites",
    placement = "topLeft",
    resolvedProviderKey,
    selectedModel,
    setModelDropdownOpen,
    setModelSearchQuery,
    setModelSortMode = () => undefined
  }: ChatModelSelectorDropdownProps) {
    const { t } = useTranslation(["playground", "common"])
    const selectorRef = React.useRef<HTMLButtonElement>(null)
    const popupRef = React.useRef<HTMLDivElement>(null)
    const returningFocusRef = React.useRef(false)
    const returnFocusToSelector = () => {
      returningFocusRef.current = true
      selectorRef.current?.focus()
      returningFocusRef.current = false
    }
    React.useEffect(() => {
      if (!modelDropdownOpen) return
      const openingFocus = document.activeElement
      let current = true
      const frame = window.requestAnimationFrame(() => {
        if (!current || !popupRef.current?.isConnected) return
        const active = document.activeElement
        if (active !== openingFocus && active !== document.body) return
        popupRef.current.querySelector<HTMLInputElement>(MODEL_SEARCH_INPUT)?.focus()
      })
      return () => {
        current = false
        window.cancelAnimationFrame(frame)
      }
    }, [modelDropdownOpen])
    // When no model is selected, the provider connection may well be "Healthy",
    // but showing that green badge next to "Select a model" is contradictory —
    // you cannot chat yet. Surface a truthful "No model selected" state instead.
    const noModelSelected = !selectedModel?.trim()
    // Offline or still connecting: the server connection is the only status.
    const connectionBlocked = connectionStatusWarning || connectionStatusPending
    const providerSelectionNeeded = !noModelSelected && modelSelectorWarning && !modelUsabilityWarning && !connectionBlocked
    // The model's own readiness applies only when nothing more specific
    // (offline server, missing provider, unusable model) is being shown.
    // Without evidence about the model (no reply yet, no provider report),
    // the chip names only the server connection.
    const readinessShown = Boolean(
      modelReadiness &&
        modelReadiness.state !== "not_checked" &&
        !connectionBlocked &&
        !providerSelectionNeeded &&
        !modelUsabilityLabel &&
        !noModelSelected
    )
    const readinessWarning = readinessShown && modelReadiness?.state === "unreachable"
    const degradedShown = connectionDegraded && !connectionBlocked && !readinessShown
    const connectionShown = connectionBlocked || degradedShown
    const statusLabel = connectionShown ? connectionStatusLabel :
      (providerSelectionNeeded
        ? t("playground:composer.chooseProvider", "Choose provider")
        : modelUsabilityLabel) ??
      (noModelSelected
        ? t("playground:composer.providerStatusNoModel", "No model selected")
        : readinessShown && modelReadiness
          ? modelReadiness.label
          : connectionStatusLabel)
    const statusTitle = connectionShown ? connectionStatusLabel : providerSelectionNeeded ? apiModelLabel :
      modelUsabilityTitle ?? (readinessShown && modelReadiness ? modelReadiness.title : apiModelLabel)
    const statusWarning =
      noModelSelected ||
      modelUsabilityWarning ||
      connectionStatusWarning ||
      degradedShown ||
      modelSelectorWarning ||
      readinessWarning
    const switchTarget = readinessShown ? modelReadiness?.switchTarget ?? null : null
    const [switchAnnouncement, setSwitchAnnouncement] = React.useState("")
    // The aria-label replaces the chip's content (and the Tooltip overrides
    // aria-describedby), so the name carries the status: the model, the
    // status word when the explanation does not already say it, then the
    // explanation.
    const statusReported =
      readinessShown || (statusWarning && statusTitle === apiModelLabel)
    const chipAccessibleName = [
      apiModelLabel,
      statusReported ? statusLabel : null,
      statusTitle
    ]
      .filter((part, index, parts): part is string => Boolean(part) && parts.indexOf(part) === index)
      .join(". ")
    // The switch button unmounts once the new model is selected, so focus
    // moves to the chip and the change is announced (CC-01).
    const handleSwitchToTarget = React.useCallback(() => {
      if (!switchTarget || !modelReadiness) return
      modelReadiness.switchToTarget()
      // Clear first so a repeat switch to the same model is announced again.
      setSwitchAnnouncement("")
      window.setTimeout(() => setSwitchAnnouncement(
        String(
          t("playground:composer.modelReadiness.switchedTo", {
            defaultValue: "Switched to {{model}}",
            model: switchTarget.modelLabel
          } as never)
        )
      ), 0)
      selectorRef.current?.focus()
    }, [modelReadiness, switchTarget, t])

    return (
      <>
      <Dropdown
        open={modelDropdownOpen}
        destroyOnHidden
        styles={{ root: { maxWidth: "calc(100vw - 16px)", minWidth: 0 } }}
        onOpenChange={(open) => {
          if (!open && popupRef.current?.contains(document.activeElement)) {
            returnFocusToSelector()
          }
          setModelDropdownOpen(open)
          if (!open) {
            setModelSearchQuery("")
          }
        }}
        menu={{
          items: modelDropdownMenuItems,
          className: "no-scrollbar [&_.ant-dropdown-menu-item]:whitespace-normal [&_.ant-dropdown-menu-item-group-title]:!text-text",
          style: { minWidth: 0 },
          activeKey: selectedModel ?? undefined
        }}
        popupRender={(menu) => (
          <div
            ref={popupRef}
            tabIndex={-1}
            onFocus={(event) => {
              if (event.target === event.currentTarget) {
                event.currentTarget.querySelector<HTMLInputElement>(MODEL_SEARCH_INPUT)?.focus()
              }
            }}
            onBlur={(event) => {
              if (!returningFocusRef.current && !event.currentTarget.contains(event.relatedTarget)) {
                setModelDropdownOpen(false, false)
                setModelSearchQuery("")
              }
            }}
            className="flex max-h-[calc(100dvh-16px)] min-w-0 flex-col overflow-y-auto rounded-lg border border-border bg-surface shadow-lg"
            onKeyDown={(event) => {
              // Keep Tab available to the search, sort, menu and help controls.
              if (event.key === "Tab") event.stopPropagation()
              // Dropdown wraps a Tooltip; its ref cannot focus the native button.
              if (event.key === "Escape") returnFocusToSelector()
            }}
          >
            {catalogControls ?? (
              <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-border p-2">
                <Input
                  size="small"
                  aria-label={t("playground:composer.modelSearchLabel", "Search models")}
                  placeholder={t(
                    "playground:composer.modelSearchPlaceholder",
                    "Search models"
                  )}
                  value={modelSearchQuery}
                  allowClear
                  className="min-w-0 flex-1 basis-36"
                  onChange={(event) => setModelSearchQuery(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key !== "Escape") event.stopPropagation()
                  }}
                />
                <Select
                  size="small"
                  aria-label={t("playground:composer.modelSortLabel", "Sort models")}
                  value={modelSortMode}
                  onChange={(value) => setModelSortMode(value as ModelSortMode)}
                  options={[
                    {
                      value: "favorites",
                      label: t(
                        "playground:composer.sort.favorites",
                        "Favorites"
                      )
                    },
                    {
                      value: "az",
                      label: t("playground:composer.sort.az", "A-Z")
                    },
                    {
                      value: "provider",
                      label: t("playground:composer.sort.provider", "Provider")
                    },
                    {
                      value: "localFirst",
                      label: t(
                        "playground:composer.sort.localFirst",
                        "Local-first"
                      )
                    }
                  ]}
                  className="min-w-0 flex-1 basis-32"
                  onKeyDown={(event) => {
                    if (event.key !== "Escape") event.stopPropagation()
                  }}
                />
              </div>
            )}
            <div className="no-scrollbar min-h-0 flex-auto max-h-[400px] overflow-y-auto">
              {menu}
            </div>
            <div className="shrink-0 border-t border-border p-2">
              <Link
                to="/docs/models"
                className="flex items-center gap-1.5 text-xs !text-text underline underline-offset-2"
                onClick={() => setModelDropdownOpen(false)}
              >
                <HelpCircle className="h-3.5 w-3.5" />
                <span>
                  {t(
                    "playground:composer.helpMeChoose",
                    "Help me choose a model"
                  )}
                </span>
                <ArrowRight className="h-3 w-3" />
              </Link>
            </div>
          </div>
        )}
        trigger={["click"]}
        placement={placement}
      >
        <Tooltip
          title={
            modelDropdownOpen
              ? null
              : statusTitle ||
                (statusWarning
                  ? t("playground:composer.selectModelTooltip", "Click to select a model")
                  : apiModelLabel)
          }
          placement="top"
        >
          <button
            type="button"
            ref={selectorRef}
            onMouseDown={(event) => {
              if (modelDropdownOpen) event.preventDefault()
            }}
            title={modelDropdownOpen ? undefined : statusTitle}
            aria-label={chipAccessibleName}
            aria-haspopup="menu"
            aria-expanded={modelDropdownOpen}
            data-testid="model-selector"
            className={`inline-flex min-h-[44px] min-w-0 max-w-full cursor-pointer items-center gap-2 rounded-lg border border-transparent px-2 text-xs transition-colors ${
              statusWarning
                ? "bg-warn/10 text-text hover:bg-warn/20"
                : "bg-transparent text-text hover:bg-surface2"
            }`}
          >
            <ProviderIcons
              provider={resolvedProviderKey}
              className={`h-3 w-3 ${
                statusWarning ? "text-text" : "text-text-subtle"
              }`}
            />
            <span className="min-w-0 max-w-[180px] truncate">{apiModelLabel}</span>
            <span
              data-testid="model-selector-status"
              data-model-readiness={readinessShown ? modelReadiness?.state : undefined}
              className="sr-only"
              title={t("common:status", "Status") as string}
            >
              {statusLabel}
            </span>
            <ChevronDown className="h-3 w-3 shrink-0" aria-hidden="true" />
          </button>
        </Tooltip>
      </Dropdown>
      {switchTarget && modelReadiness ? (
        <button
          type="button"
          data-testid="model-readiness-switch"
          onClick={handleSwitchToTarget}
          className="inline-flex min-h-[44px] min-w-0 max-w-full items-center rounded-lg border border-border px-2 text-xs text-text hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        >
          <span className="truncate">{switchTarget.label}</span>
        </button>
      ) : null}
      <span
        aria-live="polite"
        aria-atomic="true"
        data-testid="model-readiness-announcement"
        className="sr-only"
      >
        {switchAnnouncement}
      </span>
      </>
    )
  }
)
