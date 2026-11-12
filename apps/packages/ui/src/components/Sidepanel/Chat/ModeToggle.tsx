import React from "react"
import { useTranslation } from "react-i18next"
import { classNames } from "@/libs/class-name"
import { useUiModeStore } from "@/store/ui-mode"

export const ModeToggle = () => {
  const { t } = useTranslation(["common"])
  const mode = useUiModeStore((state) => state.mode)
  const setMode = useUiModeStore((state) => state.setMode)

  const buttons = [
    { id: "casual", label: t("common:chatSidebar.modeCasual", "Casual") },
    { id: "pro", label: t("common:chatSidebar.modePro", "Pro") }
  ] as const

  return (
    <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
      <span className="text-caption font-medium text-text-muted">
        {t("common:chatSidebar.modeLabel", "Mode")}
      </span>
      <div
        role="group"
        aria-label={t("common:chatSidebar.modeLabel", "Mode")}
        className="flex items-center rounded-full border border-border bg-surface2 p-1"
      >
        {buttons.map((button) => {
          const isActive = mode === button.id
          return (
            <button
              key={button.id}
              type="button"
              onClick={() => setMode(button.id)}
              aria-pressed={isActive}
              title={button.label}
              className={classNames(
                "rounded-full px-3 py-1 text-caption font-medium transition",
                isActive
                  ? "bg-surface text-text shadow-sm"
                  : "text-text-muted hover:text-text"
              )}
            >
              {button.label}
            </button>
          )
        })}
      </div>
    </div>
  )
}
