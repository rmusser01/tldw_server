import React from "react"
import { InputNumber, Modal, Select, Switch, Input } from "antd"
import { ModalFooter } from "@/components/ui/layout"
import { McpToolSelector } from "@/components/Common/McpToolSelector"
import type { ChatToolFilterCounts, ResolvedMcpTool } from "@/utils/chat-tools"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type McpCatalog = {
  id: number
  name: string
}

export type McpCatalogGroups = {
  team: McpCatalog[]
  org: McpCatalog[]
  global: McpCatalog[]
}

export interface PlaygroundMcpSettingsModalProps {
  open: boolean
  onClose: () => void
  hasMcp: boolean
  mcpStatusLabel: string

  // Catalog
  catalogsLoading: boolean
  catalogGroups: McpCatalogGroups
  catalogDraft: string
  onCatalogDraftChange: (value: string) => void
  onCatalogCommit: () => void
  onCatalogSelect: (value: number | undefined) => void
  toolCatalogId: number | null
  onToolCatalogIdChange: (value: number | null) => void
  toolCatalogStrict: boolean
  onToolCatalogStrictChange: (checked: boolean) => void

  // Module
  moduleOptions: string[]
  moduleOptionsLoading: boolean
  toolModules: string[]
  onModuleSelect: (value?: string[]) => void

  // Personal tool filter
  discoveredTools: ResolvedMcpTool[]
  toolCounts: ChatToolFilterCounts
  toolsLoading: boolean
  mcpHealthState: string
  onToolEnabledChange: (toolName: string, enabled: boolean) => void
  onResetToolFilter: () => void

  // Small model hint
  isSmallModel: boolean

  t: (key: string, defaultValue?: string, options?: any) => any
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const PlaygroundMcpSettingsModal: React.FC<PlaygroundMcpSettingsModalProps> =
  React.memo(function PlaygroundMcpSettingsModal(props) {
    const {
      open,
      onClose,
      hasMcp,
      mcpStatusLabel,
      catalogsLoading,
      catalogGroups,
      catalogDraft,
      onCatalogDraftChange,
      onCatalogCommit,
      onCatalogSelect,
      toolCatalogId,
      onToolCatalogIdChange,
      toolCatalogStrict,
      onToolCatalogStrictChange,
      moduleOptions,
      moduleOptionsLoading,
      toolModules,
      onModuleSelect,
      discoveredTools,
      toolCounts,
      toolsLoading,
      mcpHealthState,
      onToolEnabledChange,
      onResetToolFilter,
      isSmallModel,
      t
    } = props

    return (
      <Modal
        open={open}
        onCancel={onClose}
        footer={
          <ModalFooter
            hideCancel
            data-testid="mcp-settings-modal-footer"
            actions={[
              {
                label: t("common:close", "Close"),
                onClick: onClose,
                variant: "primary"
              }
            ]}
          />
        }
        width={560}
        title={t(
          "playground:composer.mcpSettingsTitle",
          "MCP tool settings"
        )}
      >
        <div className="flex flex-col gap-4">
          <div className="text-xs text-text-muted">{mcpStatusLabel}</div>
          {!hasMcp ? (
            <div className="text-sm text-text-muted">
              {t(
                "playground:composer.mcpToolsUnavailable",
                "MCP tools unavailable"
              )}
            </div>
          ) : (
            <div className="flex flex-col gap-3">
              <div className="flex flex-col gap-1">
                <label className="text-xs text-text-muted">
                  {t("playground:composer.mcpToolSelectorLabel", "Tools")}
                </label>
                <McpToolSelector
                  discoveredTools={discoveredTools}
                  toolCounts={toolCounts}
                  toolsLoading={toolsLoading}
                  hasMcp={hasMcp}
                  healthState={mcpHealthState}
                  onToolEnabledChange={onToolEnabledChange}
                  onReset={onResetToolFilter}
                  t={t}
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-text-muted">
                  {t("playground:composer.mcpCatalogLabel", "Catalog")}
                </label>
                <Select
                  size="small"
                  allowClear
                  showSearch
                  loading={catalogsLoading}
                  value={toolCatalogId ?? undefined}
                  placeholder={t(
                    "playground:composer.mcpCatalogSelectPlaceholder",
                    "Select a catalog"
                  )}
                  onChange={(value) =>
                    onCatalogSelect(value as number | undefined)
                  }
                  optionFilterProp="label"
                  className="w-full"
                >
                  {catalogGroups.team.length > 0 && (
                    <Select.OptGroup
                      label={t(
                        "playground:composer.mcpCatalogTeam",
                        "Team catalogs"
                      )}
                    >
                      {catalogGroups.team.map((catalog) => (
                        <Select.Option
                          key={`team-${catalog.id}`}
                          value={catalog.id}
                          label={catalog.name}
                        >
                          <div className="flex flex-col">
                            <span className="text-sm">{catalog.name}</span>
                            <span className="text-[11px] text-text-muted">
                              ID {catalog.id}
                            </span>
                          </div>
                        </Select.Option>
                      ))}
                    </Select.OptGroup>
                  )}
                  {catalogGroups.org.length > 0 && (
                    <Select.OptGroup
                      label={t(
                        "playground:composer.mcpCatalogOrg",
                        "Org catalogs"
                      )}
                    >
                      {catalogGroups.org.map((catalog) => (
                        <Select.Option
                          key={`org-${catalog.id}`}
                          value={catalog.id}
                          label={catalog.name}
                        >
                          <div className="flex flex-col">
                            <span className="text-sm">{catalog.name}</span>
                            <span className="text-[11px] text-text-muted">
                              ID {catalog.id}
                            </span>
                          </div>
                        </Select.Option>
                      ))}
                    </Select.OptGroup>
                  )}
                  {catalogGroups.global.length > 0 && (
                    <Select.OptGroup
                      label={t(
                        "playground:composer.mcpCatalogGlobal",
                        "Global catalogs"
                      )}
                    >
                      {catalogGroups.global.map((catalog) => (
                        <Select.Option
                          key={`global-${catalog.id}`}
                          value={catalog.id}
                          label={catalog.name}
                        >
                          <div className="flex flex-col">
                            <span className="text-sm">{catalog.name}</span>
                            <span className="text-[11px] text-text-muted">
                              ID {catalog.id}
                            </span>
                          </div>
                        </Select.Option>
                      ))}
                    </Select.OptGroup>
                  )}
                </Select>
                <Input
                  size="small"
                  placeholder={t(
                    "playground:composer.mcpCatalogPlaceholder",
                    "catalog name"
                  )}
                  value={catalogDraft}
                  onChange={(e) => onCatalogDraftChange(e.target.value)}
                  onBlur={onCatalogCommit}
                  onPressEnter={onCatalogCommit}
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-text-muted">
                  {t(
                    "playground:composer.mcpCatalogIdLabel",
                    "Catalog ID"
                  )}
                </label>
                <InputNumber
                  size="small"
                  min={0}
                  value={toolCatalogId ?? undefined}
                  onChange={(value) =>
                    onToolCatalogIdChange(
                      typeof value === "number" && Number.isFinite(value)
                        ? value
                        : null
                    )
                  }
                  placeholder={t(
                    "playground:composer.mcpCatalogIdPlaceholder",
                    "optional"
                  )}
                  className="w-full"
                />
              </div>
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs text-text-muted">
                  {t(
                    "playground:composer.mcpCatalogStrictLabel",
                    "Strict catalog filter"
                  )}
                </span>
                <Switch
                  size="small"
                  checked={toolCatalogStrict}
                  onChange={(checked) => onToolCatalogStrictChange(checked)}
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-text-muted">
                  {t("playground:composer.mcpModuleLabel", "Module")}
                </label>
                <Select
                  size="small"
                  allowClear
                  showSearch
                  mode="multiple"
                  loading={moduleOptionsLoading}
                  disabled={
                    moduleOptionsLoading || moduleOptions.length === 0
                  }
                  value={
                    toolModules.length > 0 ? toolModules : undefined
                  }
                  placeholder={t(
                    "playground:composer.mcpModuleSelectPlaceholder",
                    "Select modules"
                  )}
                  onChange={(value) =>
                    onModuleSelect(value as string[] | undefined)
                  }
                  optionFilterProp="label"
                  className="w-full"
                >
                  {moduleOptions.map((moduleId) => (
                    <Select.Option
                      key={moduleId}
                      value={moduleId}
                      label={moduleId}
                    >
                      <span className="text-sm">{moduleId}</span>
                    </Select.Option>
                  ))}
                </Select>
              </div>
              {isSmallModel && (
                <div className="rounded-md border border-border bg-surface2/60 px-2 py-1 text-[11px] text-text-muted">
                  {t(
                    "playground:composer.mcpSmallModelHint",
                    "Small/fast model: use catalog/module filters or the discovery tools (mcp.catalogs.list → mcp.modules.list → mcp.tools.list) to keep tool context light."
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </Modal>
    )
  })
