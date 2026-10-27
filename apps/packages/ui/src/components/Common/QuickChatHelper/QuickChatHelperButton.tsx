import React, {
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useState,
  useRef,
  useLayoutEffect
} from "react"
import { Tooltip, Switch, type TooltipProps } from "antd"
import { MessageCircle } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useStorage } from "@plasmohq/storage/hook"
import { useQuickChatStore } from "@/store/quick-chat"
import { classNames } from "@/libs/class-name"

// Lazy load the modal for bundle optimization
const QuickChatHelperModal = lazy(() =>
  import("./QuickChatHelperModal").then((m) => ({ default: m.QuickChatHelperModal }))
)

type QuickChatHelperButtonProps = {
  variant?: "floating" | "inline"
  showToggle?: boolean
  className?: string
  tooltipPlacement?: TooltipProps["placement"]
  appearance?: "primary" | "ghost"
  ariaLabel?: string
}

export const QuickChatHelperButton: React.FC<QuickChatHelperButtonProps> = ({
  variant = "floating",
  showToggle,
  className,
  tooltipPlacement,
  appearance = "primary",
  ariaLabel
}) => {
  const { t } = useTranslation("option")
  const [hideQuickChatHelper, setHideQuickChatHelper] = useStorage(
    "hideQuickChatHelper",
    false
  )
  const { isOpen, setIsOpen } = useQuickChatStore()
  const [isCompactViewport, setIsCompactViewport] = useState(false)
  const openerRef = useRef<HTMLElement | null>(null)
  useLayoutEffect(() => {
    if (isOpen) {
      const active = document.activeElement
      openerRef.current =
        active instanceof HTMLElement && active !== document.body ? active : null
    }
  }, [isOpen])
  const getTutorialFocusTarget = useCallback(() => openerRef.current, [])

  const handleOpen = useCallback(() => {
    setIsOpen(true)
  }, [setIsOpen])

  const handleClose = useCallback(() => {
    setIsOpen(false)
  }, [setIsOpen])

  const tooltip =
    ariaLabel ?? t("option:quickChatHelper.tooltip", "Quick Chat Helper")
  const toggleLabel = hideQuickChatHelper
    ? t(
        "settings:generalSettings.settings.hideQuickChatHelper.showLabel",
        "Show Quick Chat Helper button"
      )
    : t(
        "settings:generalSettings.settings.hideQuickChatHelper.label",
        "Hide Quick Chat Helper button"
      )

  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return
    const mediaQuery = window.matchMedia("(max-width: 520px)")
    const update = () => setIsCompactViewport(mediaQuery.matches)
    update()
    mediaQuery.addEventListener("change", update)
    return () => {
      mediaQuery.removeEventListener("change", update)
    }
  }, [])

  const isSidepanel =
    typeof window !== "undefined" &&
    window.location?.pathname?.includes("sidepanel")
  const isDocked = isSidepanel && isCompactViewport

  const shouldShowToggle = showToggle ?? variant === "floating"
  const showButton = !hideQuickChatHelper
  const showControls = showButton || shouldShowToggle
  const placement =
    tooltipPlacement ?? (variant === "floating" ? "left" : "top")
  const buttonClassName =
    variant === "floating"
      ? classNames(
          "flex items-center justify-center",
          "w-12 h-12 rounded-full",
          "bg-primary hover:bg-primaryStrong",
          "text-primary-foreground hover:text-primary-strong-foreground shadow-lg",
          "transition-all duration-200",
          "hover:scale-105 active:scale-95",
          "focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2",
          "focus-visible:ring-offset-surface"
        )
      : appearance === "ghost"
        ? classNames(
            "flex items-center justify-center",
            "p-2 rounded-lg",
            "text-text-muted hover:text-text",
            "hover:bg-surface",
            "focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          )
        : classNames(
            "flex items-center justify-center",
            "w-9 h-9 rounded-full",
            "bg-primary hover:bg-primaryStrong",
            "text-primary-foreground hover:text-primary-strong-foreground shadow-sm",
            "transition-colors duration-150",
            "focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2",
            "focus-visible:ring-offset-surface"
          )
  const iconClassName =
    variant === "floating"
      ? "h-6 w-6"
      : appearance === "ghost"
        ? "size-4"
        : "size-5"

  const controls = showControls ? (
    <div
      className={classNames(
        variant === "floating"
          ? "fixed bottom-6 right-6 z-[60] flex flex-col items-center gap-3"
          : "flex items-center gap-2",
        className
      )}
    >
      {showButton && (
        <Tooltip title={tooltip} placement={placement}>
          <button
            onClick={handleOpen}
            className={buttonClassName}
            data-testid="quick-chat-helper-open-button"
            aria-label={tooltip}
            aria-haspopup="dialog"
            aria-expanded={isOpen}
            type="button">
            <MessageCircle className={iconClassName} />
          </button>
        </Tooltip>
      )}

      {shouldShowToggle && (
        <Tooltip title={toggleLabel} placement={placement}>
          <Switch
            checked={!hideQuickChatHelper}
            onChange={(checked) => setHideQuickChatHelper(!checked)}
            aria-label={toggleLabel}
            size={variant === "floating" ? "default" : "small"}
          />
        </Tooltip>
      )}
    </div>
  ) : null

  return (
    <>
      {variant === "floating" ? (!isDocked && controls) : controls}

      {/* Modal - lazy loaded */}
      {isOpen && (
        <Suspense fallback={null}>
          <QuickChatHelperModal open={isOpen} onClose={handleClose} getTutorialFocusTarget={getTutorialFocusTarget} />
        </Suspense>
      )}
    </>
  )
}

export default QuickChatHelperButton
