import { Button as TldwButton } from "@/components/Common/Button"
import { Popover } from "antd"
import { Settings2 } from "lucide-react"
import React from "react"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PlaygroundModeLauncherProps {
  open: boolean
  onOpenChange: (open: boolean) => void

  compareModeActive: boolean
  compareFeatureEnabled: boolean
  onToggleCompare: () => void

  selectedCharacterName: string | null
  onOpenActorSettings: () => void

  onModeAnnouncement: (msg: string) => void
  t: (key: string, defaultValue?: string, options?: any) => any
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const PlaygroundModeLauncher: React.FC<PlaygroundModeLauncherProps> =
  React.memo(function PlaygroundModeLauncher(props) {
    const {
      open,
      onOpenChange,
      compareModeActive,
      compareFeatureEnabled,
      onToggleCompare,
      selectedCharacterName,
      onOpenActorSettings,
      onModeAnnouncement,
      t
    } = props

    const triggerRef = React.useRef<HTMLButtonElement>(null)
    const handleEscape = (event: React.KeyboardEvent<HTMLElement>) => {
      if (
        event.key !== "Escape" ||
        event.defaultPrevented ||
        event.nativeEvent.isComposing
      )
        return
      event.preventDefault()
      event.stopPropagation()
      onOpenChange(false)
      triggerRef.current?.focus()
    }

    const content = (
      <div onKeyDown={handleEscape} className="flex w-72 flex-col gap-1 p-1">
        <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-text-muted">
          {t("playground:composer.modes", "Modes")}
        </div>
        {/* Modes holds conversation modes only (CC-08, decision [1]). Search &
            Context and Web search are Tools-tray toggles; Voice has its own
            composer button. Compare stays hidden until it is verified (D4). */}
        {compareFeatureEnabled ? (
          <button
            type="button"
            onClick={() => {
              const next = !compareModeActive
              onToggleCompare()
              onModeAnnouncement(
                next
                  ? t(
                      "playground:composer.modeCompareEnabled",
                      "Compare mode enabled."
                    )
                  : t(
                      "playground:composer.modeCompareDisabled",
                      "Compare mode disabled."
                    )
              )
              onOpenChange(false)
            }}
            className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-sm text-text transition hover:bg-surface2">
            <span>
              {t("playground:composer.modeCompare", "Compare responses")}
            </span>
            <span className="text-xs text-text-muted">
              {compareModeActive ? t("common:on", "On") : t("common:off", "Off")}
            </span>
          </button>
        ) : null}
        <button
          type="button"
          onClick={() => {
            onOpenActorSettings()
            onModeAnnouncement(
              t(
                "playground:composer.modeCharacterNotice",
                "Character / scene settings opened."
              )
            )
            onOpenChange(false)
          }}
          className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-sm text-text transition hover:bg-surface2">
          <span>
            {t("playground:composer.modeCharacter", "Character / Scene")}
          </span>
          <span className="truncate text-xs text-text-muted">
            {selectedCharacterName
              ? t(
                  "playground:composer.modeCharacterActive",
                  "Active: {{name}}",
                  { name: selectedCharacterName }
                )
              : t("common:off", "Off")}
          </span>
        </button>
      </div>
    )

    return (
      <Popover
        trigger="click"
        placement="topLeft"
        content={content}
        open={open}
        onOpenChange={onOpenChange}>
        <TldwButton
          ref={triggerRef}
          onKeyDown={open ? handleEscape : undefined}
          variant="outline"
          size="sm"
          shape="pill"
          ariaLabel={t("playground:composer.modes", "Modes") as string}
          title={t("playground:composer.modes", "Modes") as string}
          className="min-h-[44px]">
          <span className="inline-flex items-center gap-1.5">
            <Settings2 className="h-4 w-4" aria-hidden="true" />
            <span>{t("playground:composer.modes", "Modes")}</span>
          </span>
        </TldwButton>
      </Popover>
    )
  })
