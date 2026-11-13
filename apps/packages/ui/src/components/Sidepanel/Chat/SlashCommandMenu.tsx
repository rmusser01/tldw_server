import React from "react"
import { useTranslation } from "react-i18next"
import { cn } from "@/libs/utils"
import { comboboxOptionId } from "@/hooks/useComposerCombobox"
import { SuggestionCountStatus } from "@/components/Common/SuggestionCountStatus"
import type { SlashCommandItem } from "@/types/slash-command"

export type { SlashCommandItem }

type SlashCommandMenuProps = {
  open: boolean
  commands: SlashCommandItem[]
  activeIndex: number
  onActiveIndexChange: (index: number) => void
  onSelect: (command: SlashCommandItem) => void
  className?: string
  emptyLabel?: string
  /** The listbox id the composer's combobox controls (`useComposerCombobox`). */
  listboxId?: string
}

export const SlashCommandMenu: React.FC<SlashCommandMenuProps> = ({
  open,
  commands,
  activeIndex,
  onActiveIndexChange,
  onSelect,
  className,
  emptyLabel = "No commands found",
  listboxId = "slash-commands"
}) => {
  const { t } = useTranslation(["sidepanel"])
  const activeOptionRef = React.useRef<HTMLDivElement>(null)
  React.useEffect(() => {
    if (open)
      activeOptionRef.current?.scrollIntoView?.({
        block: "nearest",
        inline: "nearest"
      })
  }, [open, activeIndex, commands])
  const countText =
    commands.length === 0
      ? emptyLabel
      : String(
          t("sidepanel:composer.commandCount", {
            count: commands.length,
            defaultValue: "{{count}} commands",
            defaultValue_one: "{{count}} command"
          })
        )

  return (
    <>
      <SuggestionCountStatus open={open} text={countText} />
      {open && (
        <div className={cn("z-20", className)}>
          <div className="max-h-48 overflow-auto rounded-md border border-border bg-elevated shadow-lg">
            {commands.length === 0 ? (
              <div className="px-3 py-2 text-xs text-text-muted">{emptyLabel}</div>
            ) : (
              <div
                role="listbox"
                id={listboxId}
                aria-label={String(t("sidepanel:composer.commandsLabel", "Commands"))}
              >
                {commands.map((command, index) => (
                  <div
                    key={command.id}
                    ref={index === activeIndex ? activeOptionRef : undefined}
                    id={comboboxOptionId(listboxId, index)}
                    role="option"
                    aria-selected={index === activeIndex}
                    // Focus stays in the composer; the option is chosen by click.
                    onMouseDown={(event) => event.preventDefault()}
                    onMouseEnter={() => onActiveIndexChange(index)}
                    onClick={() => onSelect(command)}
                    title={
                      command.description
                        ? `${command.label} - ${command.description}`
                        : command.label
                    }
                    className={cn(
                      "flex w-full cursor-pointer items-start gap-2 px-3 py-2 text-left text-sm transition-colors",
                      index === activeIndex ? "bg-surface2 text-text" : "text-text"
                    )}
                  >
                    <span className="text-text-muted">/{command.command}</span>
                    <span className="flex flex-col">
                      <span className="text-text">{command.label}</span>
                      {command.description && (
                        <span className="text-xs text-text-subtle">
                          {command.description}
                        </span>
                      )}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  )
}

export default SlashCommandMenu
