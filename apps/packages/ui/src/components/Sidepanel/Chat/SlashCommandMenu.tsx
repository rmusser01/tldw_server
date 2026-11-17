import React from "react"
import { cn } from "@/libs/utils"

export type SlashCommandItem = {
  id: string
  command: string
  label: string
  description?: string
  keywords?: string[]
  action?: () => void
  insertText?: string
}

type SlashCommandMenuProps = {
  open: boolean
  commands: SlashCommandItem[]
  activeIndex: number
  onActiveIndexChange: (index: number) => void
  onSelect: (command: SlashCommandItem) => void
  className?: string
  emptyLabel?: string
}

export const SlashCommandMenu: React.FC<SlashCommandMenuProps> = ({
  open,
  commands,
  activeIndex,
  onActiveIndexChange,
  onSelect,
  className,
  emptyLabel = "No commands found"
}) => {
  const activeOptionRef = React.useRef<HTMLButtonElement>(null)
  React.useEffect(() => {
    if (open)
      activeOptionRef.current?.scrollIntoView?.({
        block: "nearest",
        inline: "nearest"
      })
  }, [open, activeIndex, commands])

  if (!open) {
    return null
  }

  return (
    <div className={cn("z-20", className)}>
      <div className="max-h-48 overflow-auto rounded-md border border-border bg-elevated shadow-lg">
        {commands.length === 0 ? (
          <div className="px-3 py-2 text-xs text-text-muted">{emptyLabel}</div>
        ) : (
          commands.map((command, index) => (
            <button
              key={command.id}
              ref={index === activeIndex ? activeOptionRef : undefined}
              type="button"
              role="option"
              aria-selected={index === activeIndex}
              onMouseEnter={() => onActiveIndexChange(index)}
              onClick={() => onSelect(command)}
              title={
                command.description
                  ? `${command.label} - ${command.description}`
                  : command.label
              }
              className={cn(
                "flex w-full items-start gap-2 px-3 py-2 text-left text-sm transition-colors",
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
            </button>
          ))
        )}
      </div>
    </div>
  )
}

export default SlashCommandMenu
