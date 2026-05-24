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
  modelDropdownMenuItems: any[]
  modelDropdownOpen: boolean
  modelSearchQuery: string
  modelSelectorWarning?: boolean
  modelSortMode: ModelSortMode
  placement?: "topLeft" | "bottomLeft"
  resolvedProviderKey: string
  selectedModel: string | null | undefined
  setModelDropdownOpen: (open: boolean) => void
  setModelSearchQuery: (query: string) => void
  setModelSortMode: (mode: ModelSortMode) => void
}

export const ChatModelSelectorDropdown = React.memo(
  function ChatModelSelectorDropdown({
    apiModelLabel,
    connectionStatusLabel,
    connectionStatusWarning = false,
    modelDropdownMenuItems,
    modelDropdownOpen,
    modelSearchQuery,
    modelSelectorWarning = false,
    modelSortMode,
    placement = "topLeft",
    resolvedProviderKey,
    selectedModel,
    setModelDropdownOpen,
    setModelSearchQuery,
    setModelSortMode
  }: ChatModelSelectorDropdownProps) {
    const { t } = useTranslation(["playground", "common"])

    return (
      <Dropdown
        open={modelDropdownOpen}
        onOpenChange={(open) => {
          setModelDropdownOpen(open)
          if (!open) {
            setModelSearchQuery("")
          }
        }}
        menu={{
          items: modelDropdownMenuItems,
          className: "no-scrollbar",
          activeKey: selectedModel ?? undefined
        }}
        popupRender={(menu) => (
          <div className="rounded-lg border border-border bg-surface shadow-lg">
            <div className="flex items-center gap-2 border-b border-border p-2">
              <Input
                size="small"
                placeholder={t(
                  "playground:composer.modelSearchPlaceholder",
                  "Search models"
                )}
                value={modelSearchQuery}
                allowClear
                className="flex-1"
                onChange={(event) => setModelSearchQuery(event.target.value)}
                onKeyDown={(event) => event.stopPropagation()}
              />
              <Select
                size="small"
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
                className="min-w-[120px]"
                onKeyDown={(event) => event.stopPropagation()}
              />
            </div>
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
            modelSelectorWarning
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
            title={apiModelLabel}
            aria-label={apiModelLabel}
            aria-haspopup="listbox"
            aria-expanded={modelDropdownOpen}
            data-testid="model-selector"
            className={`inline-flex min-h-[44px] min-w-0 cursor-pointer items-center gap-1 rounded-full border px-2 text-[10px] transition-colors ${
              modelSelectorWarning
                ? "border-warn/50 bg-warn/10 text-warn hover:bg-warn/20"
                : "border-border bg-surface hover:bg-surface-hover"
            }`}
          >
            <ProviderIcons
              provider={resolvedProviderKey}
              className={`h-3 w-3 ${
                modelSelectorWarning ? "text-warn" : "text-text-subtle"
              }`}
            />
            <span className="max-w-[120px] truncate">{apiModelLabel}</span>
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
          </button>
        </Tooltip>
      </Dropdown>
    )
  }
)
