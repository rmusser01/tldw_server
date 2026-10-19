import { ProviderIcons } from "@/components/Common/ProviderIcon"
import { Dropdown, Input, Select, Tooltip } from "antd"
import { ArrowRight, HelpCircle } from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"
import { Link } from "react-router-dom"
import type { ModelSortMode } from "@/hooks/playground"

type ChatModelSelectorDropdownProps = {
  apiModelLabel: string
  connectionStatusLabel: string
  connectionStatusWarning?: boolean
  catalogControls?: React.ReactNode
  modelUsabilityLabel?: string
  modelUsabilityTitle?: string
  modelUsabilityWarning?: boolean
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
    modelUsabilityLabel,
    modelUsabilityTitle,
    modelUsabilityWarning = false,
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
        popupRef.current.querySelector<HTMLInputElement>("input")?.focus()
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
    const providerSelectionNeeded = !noModelSelected && modelSelectorWarning && !modelUsabilityWarning && !connectionStatusWarning
    const statusLabel = connectionStatusWarning ? connectionStatusLabel :
      (providerSelectionNeeded
        ? t("playground:composer.chooseProvider", "Choose provider")
        : modelUsabilityLabel) ??
      (noModelSelected
        ? t("playground:composer.providerStatusNoModel", "No model selected")
        : connectionStatusLabel)
    const statusTitle = connectionStatusWarning ? connectionStatusLabel : providerSelectionNeeded ? apiModelLabel : modelUsabilityTitle ?? apiModelLabel
    const statusWarning =
      noModelSelected ||
      modelUsabilityWarning ||
      connectionStatusWarning ||
      modelSelectorWarning

    return (
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
                event.currentTarget.querySelector<HTMLInputElement>("input")?.focus()
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
            aria-label={statusTitle === apiModelLabel ? apiModelLabel : `${apiModelLabel}. ${statusTitle}`}
            aria-haspopup="menu"
            aria-expanded={modelDropdownOpen}
            data-testid="model-selector"
            className={`inline-flex min-h-[44px] min-w-0 max-w-full cursor-pointer items-center gap-1 rounded-full border px-2 text-[10px] transition-colors ${
              statusWarning
                ? "border-warn/50 bg-warn/10 text-text hover:bg-warn/20"
                : "border-border bg-surface hover:bg-surface-hover"
            }`}
          >
            <ProviderIcons
              provider={resolvedProviderKey}
              className={`h-3 w-3 ${
                statusWarning ? "text-warn" : "text-text-subtle"
              }`}
            />
            <span className="max-w-[120px] truncate">{apiModelLabel}</span>
            <span
              className={`min-w-0 truncate rounded-full px-1.5 py-0.5 text-[9px] ${
                statusWarning
                  ? "bg-warn/10 text-text"
                  : "bg-success/10 text-text"
              }`}
              title={
                t(
                  "playground:composer.statusTooltip",
                  "Status"
                ) as string
              }
            >
              {statusLabel}
            </span>
          </button>
        </Tooltip>
      </Dropdown>
    )
  }
)
