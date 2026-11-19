/**
 * The Pro composer's model parameters as one "Runtime" status chip (XS-15).
 *
 * The chip names what is set for this chat (temperature, Top P, the output
 * limit, JSON mode) and opens the full parameter panel in a popover. It
 * replaces the two always-visible accordion rows above the composer, so the
 * conversation keeps its share of a narrow side panel. "Runtime" is the full
 * page's name for the same controls (decision [1]).
 */
import { Popover } from "antd"
import { Settings2 } from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"
import { useStoreChatModelSettings } from "@/store/model"
import { useUiModeStore } from "@/store/ui-mode"
import { ModelParamsPanel } from "./ModelParamsPanel"

type SidepanelRuntimeChipProps = {
  onOpenFullSettings?: () => void
  selectedModel?: string | null
}

export const SidepanelRuntimeChip: React.FC<SidepanelRuntimeChipProps> = ({
  onOpenFullSettings,
  selectedModel
}) => {
  const { t } = useTranslation(["sidepanel", "playground"])
  const mode = useUiModeStore((state) => state.mode)
  const temperature = useStoreChatModelSettings((state) => state.temperature)
  const topP = useStoreChatModelSettings((state) => state.topP)
  const numPredict = useStoreChatModelSettings((state) => state.numPredict)
  const jsonMode = useStoreChatModelSettings((state) => state.jsonMode)
  const [open, setOpen] = React.useState(false)

  if (mode !== "pro") return null

  const summary =
    [
      typeof temperature === "number" ? `temp ${temperature}` : null,
      typeof topP === "number" ? `top-p ${topP}` : null,
      typeof numPredict === "number" ? `max ${numPredict}` : null,
      jsonMode ? "JSON" : null
    ]
      .filter(Boolean)
      .join(" · ") ||
    t("sidepanel:runtimeChip.defaults", "Model defaults")

  return (
    <div className="flex justify-end px-2 pt-1">
      <Popover
        trigger="click"
        open={open}
        onOpenChange={setOpen}
        placement="topRight"
        content={
          <div className="max-h-[60vh] w-[min(22rem,calc(100vw-2rem))] overflow-y-auto overscroll-contain">
            <ModelParamsPanel
              onOpenFullSettings={onOpenFullSettings}
              selectedModel={selectedModel}
            />
          </div>
        }
      >
        <button
          type="button"
          data-testid="sidepanel-runtime-chip"
          aria-haspopup="dialog"
          aria-expanded={open}
          title={t(
            "sidepanel:runtimeChip.title",
            "Model parameters for this chat"
          )}
          className="inline-flex max-w-full min-h-7 items-center gap-1.5 rounded-full border border-border px-2.5 py-0.5 text-xs text-text-muted hover:bg-surface2 hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        >
          <Settings2 className="size-3 shrink-0" aria-hidden="true" />
          <span className="shrink-0 font-medium text-text">
            {t("playground:cockpit.runtime", "Runtime")}
          </span>
          <span className="min-w-0 truncate">{summary}</span>
        </button>
      </Popover>
    </div>
  )
}
