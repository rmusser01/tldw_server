import React from "react"
import { Popover, Radio, Tooltip } from "antd"
import { ChevronDown } from "lucide-react"
import { Button as TldwButton } from "@/components/Common/Button"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PlaygroundMcpControlProps {
  hasMcp: boolean
  mcpHealthState: string
  mcpToolsLoading: boolean
  mcpToolsCount: number
  toolChoice: string
  onToolChoiceChange: (value: string) => void
  toolRunStatusLabel: string

  mcpAriaLabel: string
  mcpSummaryLabel: string
  mcpChoiceLabel: string
  mcpDisabledReason: string
  mcpPopoverOpen: boolean
  onMcpPopoverChange: (open: boolean) => void
  onOpenMcpSettings: () => void

  t: (key: string, defaultValue?: string, options?: any) => any
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const PlaygroundMcpControl: React.FC<PlaygroundMcpControlProps> =
  React.memo(function PlaygroundMcpControl(props) {
    const {
      hasMcp,
      mcpHealthState,
      mcpToolsLoading,
      mcpToolsCount,
      toolChoice,
      onToolChoiceChange,
      toolRunStatusLabel,
      mcpAriaLabel,
      mcpSummaryLabel,
      mcpChoiceLabel,
      mcpDisabledReason,
      mcpPopoverOpen,
      onMcpPopoverChange,
      onOpenMcpSettings,
      t
    } = props

    const disabled = !hasMcp || mcpHealthState === "unhealthy"

    const controlContent = (
      <div className="flex w-64 flex-col gap-2 p-2">
        <div className="text-[10px] font-semibold uppercase tracking-wider text-text-muted">
          {t("playground:composer.mcpToolsLabel", "MCP tools")}
        </div>
        <div className="text-xs text-text-muted">{mcpSummaryLabel}</div>
        <div className="flex flex-col gap-1">
          <div className="text-xs font-semibold text-text-muted">
            {t("playground:composer.toolChoiceLabel", "Tool choice")}
          </div>
          <Radio.Group
            size="small"
            value={toolChoice}
            onChange={(e) =>
              onToolChoiceChange(e.target.value)
            }
            className="flex flex-wrap gap-1"
            aria-label={t(
              "playground:composer.toolChoiceLabel",
              "Tool choice"
            )}
            disabled={
              disabled || mcpToolsLoading || mcpToolsCount === 0
            }
          >
            <Radio.Button value="auto">
              {t("playground:composer.toolChoiceAuto", "Auto")}
            </Radio.Button>
            <Radio.Button value="required">
              {t("playground:composer.toolChoiceRequired", "Required")}
            </Radio.Button>
            <Radio.Button value="none">
              {t("playground:composer.toolChoiceNone", "None")}
            </Radio.Button>
          </Radio.Group>
          <div className="text-[11px] text-text-muted">
            {t("playground:composer.toolRunStatus", "Tool run")}:{" "}
            {toolRunStatusLabel}
          </div>
          <button
            type="button"
            onClick={() => {
              onMcpPopoverChange(false)
              onOpenMcpSettings()
            }}
            className="mt-1 inline-flex w-fit items-center gap-1 text-xs font-medium text-primary hover:text-primaryStrong"
          >
            {t("playground:composer.mcpConfigure", "Configure tools")}
          </button>
        </div>
      </div>
    )

    const button = (
      <TldwButton
        variant="outline"
        size="md"
        shape="pill"
        ariaLabel={mcpAriaLabel}
        title={mcpAriaLabel}
        disabled={disabled}
        data-testid="mcp-tools-toggle"
        className="gap-1.5 min-h-[44px]"
      >
        <span className="inline-flex items-center gap-1.5">
          <span className="text-[11px] font-semibold">MCP</span>
          <span className="text-[11px] text-text-muted">
            {mcpChoiceLabel}
          </span>
          {!mcpToolsLoading && hasMcp && mcpToolsCount > 0 && (
            <span className="rounded-full bg-surface2 px-1.5 py-0.5 text-[10px] text-text-muted">
              {mcpToolsCount}
            </span>
          )}
          <ChevronDown
            className="h-3.5 w-3.5 text-text-subtle"
            aria-hidden="true"
          />
        </span>
      </TldwButton>
    )

    if (disabled) {
      return (
        <Tooltip title={mcpDisabledReason}>
          <span>{button}</span>
        </Tooltip>
      )
    }

    return (
      <Popover
        trigger="click"
        placement="topRight"
        content={controlContent}
        open={mcpPopoverOpen}
        onOpenChange={onMcpPopoverChange}
      >
        <span className="inline-flex">{button}</span>
      </Popover>
    )
  })
