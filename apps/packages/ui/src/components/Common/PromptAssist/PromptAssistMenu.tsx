import { Button } from "@/components/Common/Button"
import type { PromptImproveModelSelection } from "@/services/prompt-improvement"
import { Sparkles } from "lucide-react"
import {
  type RefObject,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState
} from "react"
import { useTranslation } from "react-i18next"

export type PromptAssistCapability = "supported" | "unsupported" | "unknown"

export type PromptAssistMenuProps = {
  draft: string
  capability: PromptAssistCapability
  modelSelection: PromptImproveModelSelection | null
  modelDisplayName?: string
  onImproveNow: () => void
  onReviewChanges: () => void
  onBuildFromRecipe?: () => void
  onSelectModel?: () => void
  triggerRef?: RefObject<HTMLButtonElement | null>
  disabled?: boolean
  compact?: boolean
  placement?: "top" | "bottom"
}

export function PromptAssistMenu({
  draft,
  capability,
  modelSelection,
  modelDisplayName,
  onImproveNow,
  onReviewChanges,
  onBuildFromRecipe,
  onSelectModel,
  triggerRef: providedTriggerRef,
  disabled = false,
  compact = false,
  placement = "bottom"
}: PromptAssistMenuProps) {
  const { t } = useTranslation(["common"])
  const [open, setOpen] = useState(false)
  const internalTriggerRef = useRef<HTMLButtonElement>(null)
  const triggerRef = providedTriggerRef ?? internalTriggerRef
  const disclosureRef = useRef<HTMLDivElement>(null)
  const menuHorizontalOffsetRef = useRef(0)
  const [menuHorizontalOffset, setMenuHorizontalOffset] = useState(0)
  const [menuMaxHeight, setMenuMaxHeight] = useState<number>()
  const popupId = useId()
  const hasDraft = Boolean(draft.trim())
  const hasModel = Boolean(modelSelection?.selected_model.trim())
  const actionsEnabled =
    !disabled && capability === "supported" && hasDraft && hasModel

  useEffect(() => {
    if (!open) return
    const handlePointerDown = (event: MouseEvent) => {
      if (
        disclosureRef.current?.contains(event.target as Node) ||
        triggerRef.current?.contains(event.target as Node)
      ) {
        return
      }
      setOpen(false)
    }
    document.addEventListener("mousedown", handlePointerDown)
    return () => {
      document.removeEventListener("mousedown", handlePointerDown)
    }
  }, [open, triggerRef])

  useLayoutEffect(() => {
    if (!open) return

    const updateMenuPosition = () => {
      const menu = disclosureRef.current
      const trigger = triggerRef.current
      if (!menu || !trigger) return

      const menuRect = menu.getBoundingClientRect()
      const baseLeft = menuRect.left - menuHorizontalOffsetRef.current
      const baseRight = menuRect.right - menuHorizontalOffsetRef.current
      const viewportInset = 8
      const nextHorizontalOffset =
        baseLeft < viewportInset
          ? viewportInset - baseLeft
          : baseRight > window.innerWidth - viewportInset
            ? window.innerWidth - viewportInset - baseRight
            : 0
      menuHorizontalOffsetRef.current = nextHorizontalOffset
      setMenuHorizontalOffset(nextHorizontalOffset)

      const triggerRect = trigger.getBoundingClientRect()
      setMenuMaxHeight(
        Math.max(
          0,
          placement === "top"
            ? triggerRect.top - viewportInset * 2
            : window.innerHeight - triggerRect.bottom - viewportInset * 2
        )
      )
    }

    updateMenuPosition()
    window.addEventListener("resize", updateMenuPosition)
    window.addEventListener("scroll", updateMenuPosition, true)
    return () => {
      window.removeEventListener("resize", updateMenuPosition)
      window.removeEventListener("scroll", updateMenuPosition, true)
    }
  }, [open, placement, triggerRef])

  const activeModel = !hasModel
    ? null
    : modelSelection?.selected_model.trim().toLowerCase() === "auto"
      ? t("common:promptAssist.autoModel", "Auto")
      : `${modelDisplayName || modelSelection?.selected_model.trim()}${
          modelSelection?.provider_hint
            ? ` (${modelSelection.provider_hint})`
            : ""
        }`

  const recovery = !hasDraft
    ? t(
        "common:promptAssist.emptyDraft",
        "Write a draft to enable prompt improvement."
      )
    : capability === "unsupported"
      ? t(
          "common:promptAssist.unsupported",
          "Prompt improvement requires a newer server version."
        )
      : capability === "unknown"
        ? t(
            "common:promptAssist.unknownCapability",
            "Reconnect to check prompt improvement availability."
          )
        : !hasModel
          ? t(
              "common:promptAssist.selectModelRecovery",
              "Select a chat model to improve this draft."
            )
          : null

  const runAction = (action: () => void) => {
    if (!actionsEnabled) return
    setOpen(false)
    action()
  }

  const runLocalAction = (action: () => void) => {
    if (disabled) return
    setOpen(false)
    action()
  }

  return (
    <div
      className="relative inline-flex"
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
          setOpen(false)
        }
      }}
      onKeyDown={(event) => {
        if (!open || event.key !== "Escape") return
        event.preventDefault()
        event.stopPropagation()
        setOpen(false)
        triggerRef.current?.focus()
      }}>
      <button
        ref={triggerRef}
        type="button"
        aria-expanded={open}
        aria-controls={open ? popupId : undefined}
        aria-label={t("common:promptAssist.trigger", "Improve prompt")}
        title={t("common:promptAssist.trigger", "Improve prompt")}
        className={
          compact
            ? "inline-flex h-11 w-11 min-h-[44px] min-w-[44px] items-center justify-center rounded-md border border-border bg-background p-0 text-sm font-medium hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
            : "inline-flex min-h-11 items-center gap-2 rounded-md border border-border bg-background px-3 text-sm font-medium hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
        }
        disabled={disabled}
        onClick={() => setOpen((value) => !value)}>
        <Sparkles aria-hidden="true" className="h-4 w-4" />
        {compact ? null : t("common:promptAssist.label", "Improve my prompt")}
      </button>
      {open ? (
        <div
          ref={disclosureRef}
          id={popupId}
          role="group"
          aria-label={t(
            "common:promptAssist.menuLabel",
            "Prompt improvement actions"
          )}
          style={{
            maxHeight: menuMaxHeight,
            transform: `translateX(${menuHorizontalOffset}px)`
          }}
          className={`absolute right-0 z-50 max-h-[calc(100vh-1rem)] w-72 max-w-[calc(100vw-1rem)] overflow-y-auto rounded-lg border border-border bg-popover p-2 text-popover-foreground shadow-lg ${
            placement === "top" ? "bottom-full mb-2" : "top-full mt-2"
          }`}>
          <p className="px-2 pb-2 text-xs text-muted-foreground">
            {recovery ??
              t("common:promptAssist.activeModel", "Active model: {{model}}", {
                model: activeModel ?? ""
              })}
          </p>
          <Button
            variant="ghost"
            size="lg"
            disabled={!actionsEnabled}
            className="w-full flex-col items-start text-left"
            onClick={() => runAction(onImproveNow)}>
            <span className="font-medium">
              {t("common:promptAssist.improveNow", "Improve now")}
            </span>
            <span className="text-xs text-muted-foreground">
              {t(
                "common:promptAssist.improveNowHelp",
                "Replace this draft when checks pass."
              )}
            </span>
          </Button>
          <Button
            variant="ghost"
            size="lg"
            disabled={!actionsEnabled}
            className="w-full flex-col items-start text-left"
            onClick={() => runAction(onReviewChanges)}>
            <span className="font-medium">
              {t("common:promptAssist.reviewChanges", "Review changes")}
            </span>
            <span className="text-xs text-muted-foreground">
              {t(
                "common:promptAssist.reviewChangesHelp",
                "Edit and compare before applying."
              )}
            </span>
          </Button>
          {onBuildFromRecipe ? (
            <Button
              variant="ghost"
              size="lg"
              disabled={disabled}
              className="w-full flex-col items-start text-left"
              onClick={() => runLocalAction(onBuildFromRecipe)}>
              <span className="font-medium">
                {t("common:promptAssist.buildFromRecipe", "Build from recipe")}
              </span>
              <span className="text-xs text-muted-foreground">
                {t(
                  "common:promptAssist.buildFromRecipeHelp",
                  "Compile a structured recipe locally into this draft."
                )}
              </span>
            </Button>
          ) : null}
          {!hasModel &&
          capability === "supported" &&
          hasDraft &&
          onSelectModel ? (
            <Button
              variant="outline"
              size="lg"
              className="mt-2 w-full"
              onClick={() => {
                setOpen(false)
                onSelectModel()
              }}>
              {t("common:promptAssist.selectModel", "Select model")}
            </Button>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
