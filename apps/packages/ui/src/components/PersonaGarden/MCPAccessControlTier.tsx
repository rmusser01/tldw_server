import React from "react"

type ConfirmationMode = "always" | "destructive_only" | "never"

type MCPAccessControlTierProps = {
  serverId: string
  mode: ConfirmationMode
  onChange: (mode: ConfirmationMode) => void
}

const CONFIRMATION_OPTIONS: Array<{
  value: ConfirmationMode
  label: string
  description: string
}> = [
  {
    value: "always",
    label: "Always ask before actions",
    description: "Require approval before every tool-backed action."
  },
  {
    value: "destructive_only",
    label: "Ask for destructive actions",
    description: "Only stop for actions that may change or delete data."
  },
  {
    value: "never",
    label: "Never ask",
    description: "Run matched commands immediately without an approval pause."
  }
]

export const MCPAccessControlTier: React.FC<MCPAccessControlTierProps> = ({
  serverId,
  mode,
  onChange
}) => {
  return (
    <div
      data-testid={`mcp-access-control-${serverId}`}
      className="space-y-2"
    >
      {CONFIRMATION_OPTIONS.map((option) => {
        const selected = mode === option.value
        return (
          <button
            key={option.value}
            type="button"
            aria-label={option.label}
            aria-pressed={selected}
            data-selected={selected ? "true" : "false"}
            className={
              "flex w-full items-start justify-between rounded-lg border px-3 py-3 text-left transition-colors " +
              (selected
                ? "border-blue-500/60 bg-blue-500/10"
                : "border-border bg-surface2")
            }
            onClick={() => onChange(option.value)}
          >
            <div>
              <div className="text-sm font-medium text-text">
                {option.label}
              </div>
              <div className="mt-1 text-xs text-text-muted">
                {option.description}
              </div>
            </div>
          </button>
        )
      })}
      <div className="text-xs text-text-subtle">
        <span>Fine-tune per-tool permissions</span>
      </div>
    </div>
  )
}
