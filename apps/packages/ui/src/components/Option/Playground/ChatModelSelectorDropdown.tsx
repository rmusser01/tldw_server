import { ProviderIcons } from "@/components/Common/ProviderIcon"
import { Dropdown, Tooltip } from "antd"
import type { MenuProps } from "antd"
import { ArrowRight, HelpCircle } from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"
import { Link } from "react-router-dom"

type ChatModelSelectorDropdownProps = {
  activeModelKey?: string | null
  apiModelLabel: string
  catalogControls: React.ReactNode
  connectionStatusLabel: string
  connectionStatusWarning?: boolean
  modelDropdownMenuItems: MenuProps["items"]
  modelDropdownOpen: boolean
  modelSelectorWarning?: boolean
  modelUsabilityLabel?: string | null
  modelUsabilityTitle?: string | null
  modelUsabilityWarning?: boolean
  onBeforeOpen?: () => void
  placement?: "topLeft" | "bottomLeft"
  resolvedProviderKey: string
  setModelDropdownOpen: (open: boolean) => void
  setModelSearchQuery: (query: string) => void
}

export const ChatModelSelectorDropdown = React.memo(
  function ChatModelSelectorDropdown({
    activeModelKey,
    apiModelLabel,
    catalogControls,
    connectionStatusLabel,
    connectionStatusWarning = false,
    modelDropdownMenuItems,
    modelDropdownOpen,
    modelSelectorWarning = false,
    modelUsabilityLabel = null,
    modelUsabilityTitle = null,
    modelUsabilityWarning = false,
    onBeforeOpen,
    placement = "topLeft",
    resolvedProviderKey,
    setModelDropdownOpen,
    setModelSearchQuery
  }: ChatModelSelectorDropdownProps) {
    const { t } = useTranslation(["playground", "common"])
    const trimmedModelUsabilityLabel = modelUsabilityLabel?.trim() || null
    const hasModelUsabilityOverride = Boolean(trimmedModelUsabilityLabel)
    const selectorLabel = hasModelUsabilityOverride
      ? `${apiModelLabel} - ${trimmedModelUsabilityLabel}`
      : apiModelLabel
    const selectorTitle = hasModelUsabilityOverride
      ? modelUsabilityTitle?.trim() || selectorLabel
      : apiModelLabel
    const selectorWarning = modelSelectorWarning || modelUsabilityWarning

    return (
      <Dropdown
        open={modelDropdownOpen}
        onOpenChange={(open) => {
          if (open) {
            onBeforeOpen?.()
          }
          setModelDropdownOpen(open)
          if (!open) {
            setModelSearchQuery("")
          }
        }}
        menu={{
          items: modelDropdownMenuItems,
          className: "no-scrollbar",
          activeKey: activeModelKey ?? undefined
        }}
        popupRender={(menu) => (
          <div className="rounded-lg border border-border bg-surface shadow-lg">
            {catalogControls}
            <div className="no-scrollbar max-h-[400px] overflow-y-auto">
              {menu}
            </div>
            <div className="border-t border-border p-2">
              <Link
                to="/docs/models"
                className="flex items-center gap-1.5 text-xs text-primary transition-colors hover:text-primary/80"
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
            hasModelUsabilityOverride
              ? selectorTitle
              : modelSelectorWarning
              ? t(
                  "playground:composer.selectModelTooltip",
                  "Click to select a model"
                )
              : apiModelLabel
          }
          placement="top"
        >
          <button
            type="button"
            title={selectorTitle}
            aria-label={selectorTitle}
            aria-haspopup="listbox"
            aria-expanded={modelDropdownOpen}
            data-testid="model-selector"
            className={`inline-flex min-h-[44px] min-w-0 cursor-pointer items-center gap-1 rounded-full border px-2 text-[10px] transition-colors ${
              selectorWarning
                ? "border-warn/50 bg-warn/10 text-warn hover:bg-warn/20"
                : "border-border bg-surface hover:bg-surface-hover"
            }`}
          >
            <ProviderIcons
              provider={resolvedProviderKey}
              className={`h-3 w-3 ${
                selectorWarning ? "text-warn" : "text-text-subtle"
              }`}
            />
            <span className="max-w-[120px] truncate">{selectorLabel}</span>
            {!hasModelUsabilityOverride && (
              <span
                className={`rounded-full px-1.5 py-0.5 text-[9px] ${
                  connectionStatusWarning
                    ? "bg-warn/10 text-warn"
                    : "bg-success/10 text-success"
                }`}
                title={
                  t(
                    "playground:composer.providerStatusTooltip",
                    "Provider status"
                  ) as string
                }
              >
                {connectionStatusLabel}
              </span>
            )}
          </button>
        </Tooltip>
      </Dropdown>
    )
  }
)
