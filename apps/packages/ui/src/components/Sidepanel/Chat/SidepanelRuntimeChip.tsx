/**
 * The Pro composer's model parameters as one "Runtime" status chip (XS-15).
 *
 * The chip names what is set for this chat (the first two values set here,
 * then "+N" for the rest, or "Defaults") and opens the full parameter panel
 * in a popover dialog. It replaces the two always-visible accordion rows
 * above the composer, so the conversation keeps its share of a narrow side
 * panel. "Runtime" is the full page's name for the same controls
 * (decision [1]).
 *
 * Keyboard: opening moves focus to the dialog's first control; Escape
 * closes it and returns focus to the chip.
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

const isSetNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value)
const isSetText = (value: unknown) => typeof value === "string" && value.trim().length > 0

/** The values set for this chat's model (not inherited defaults), in display order. */
const setHereParts = (own: Record<string, unknown> | undefined): string[] => {
  if (!own) return []
  const parts: string[] = []
  if (isSetNumber(own.temperature)) parts.push(`temp ${own.temperature}`)
  if (isSetNumber(own.topP)) parts.push(`top-p ${own.topP}`)
  if (isSetNumber(own.numPredict)) parts.push(`max ${own.numPredict}`)
  if (isSetNumber(own.topK)) parts.push(`top-k ${own.topK}`)
  if (isSetNumber(own.minP)) parts.push(`min-p ${own.minP}`)
  if (isSetNumber(own.frequencyPenalty)) parts.push(`freq ${own.frequencyPenalty}`)
  if (isSetNumber(own.presencePenalty)) parts.push(`pres ${own.presencePenalty}`)
  if (isSetNumber(own.seed)) parts.push(`seed ${own.seed}`)
  if (own.jsonMode === true) parts.push("JSON")
  if (isSetText(own.apiProvider)) parts.push(String(own.apiProvider))
  if (isSetText(own.extraHeaders)) parts.push("headers")
  if (isSetText(own.extraBody)) parts.push("body")
  return parts
}

export const SidepanelRuntimeChip: React.FC<SidepanelRuntimeChipProps> = ({
  onOpenFullSettings,
  selectedModel
}) => {
  const { t } = useTranslation(["sidepanel", "playground"])
  const mode = useUiModeStore((state) => state.mode)
  const ownSettings = useStoreChatModelSettings((state) =>
    state.activeSettingsScope
      ? state.scopedSettingsByModelKey[state.activeSettingsScope]
      : state.globalSettings
  ) as Record<string, unknown> | undefined
  const [open, setOpen] = React.useState(false)
  const chipRef = React.useRef<HTMLButtonElement>(null)
  const dialogRef = React.useRef<HTMLDivElement>(null)

  const close = React.useCallback(() => {
    setOpen(false)
    requestAnimationFrame(() => chipRef.current?.focus())
  }, [])

  // On open, focus the dialog's first control once the popover has mounted it.
  React.useEffect(() => {
    if (!open) return
    let frame = 0
    let tries = 0
    const focusFirst = () => {
      const first = dialogRef.current?.querySelector<HTMLElement>(
        "button, input, select, textarea, [tabindex]:not([tabindex='-1'])"
      )
      if (first) first.focus()
      else if (tries++ < 10) frame = requestAnimationFrame(focusFirst)
    }
    frame = requestAnimationFrame(focusFirst)
    return () => cancelAnimationFrame(frame)
  }, [open])

  if (mode !== "pro") return null

  const parts = setHereParts(ownSettings)
  const summary =
    parts.length === 0
      ? t("sidepanel:runtimeChip.defaults", "Defaults")
      : [...parts.slice(0, 2), ...(parts.length > 2 ? [`+${parts.length - 2}`] : [])].join(" · ")
  const label = t("playground:cockpit.runtime", "Runtime")

  return (
    <div className="flex justify-end px-2 pt-1">
      <Popover
        trigger="click"
        open={open}
        onOpenChange={(next) => (next ? setOpen(true) : close())}
        placement="topRight"
        content={
          <div
            ref={dialogRef}
            role="dialog"
            aria-label={label}
            onKeyDown={(event) => {
              if (event.key !== "Escape") return
              event.preventDefault()
              event.stopPropagation()
              close()
            }}
            className="max-h-[60vh] w-[min(22rem,calc(100vw-2rem))] overflow-y-auto overscroll-contain"
          >
            <ModelParamsPanel
              onOpenFullSettings={onOpenFullSettings}
              selectedModel={selectedModel}
            />
          </div>
        }
      >
        <button
          ref={chipRef}
          type="button"
          data-testid="sidepanel-runtime-chip"
          aria-haspopup="dialog"
          aria-expanded={open}
          title={t("sidepanel:runtimeChip.title", "Model parameters for this chat")}
          className="inline-flex max-w-full min-h-7 items-center gap-1.5 rounded-full border border-border px-2.5 py-0.5 text-xs text-text-muted hover:bg-surface2 hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        >
          <Settings2 className="size-3 shrink-0" aria-hidden="true" />
          <span className="shrink-0 font-medium text-text">{label}</span>
          <span className="min-w-0 truncate">{summary}</span>
        </button>
      </Popover>
    </div>
  )
}
