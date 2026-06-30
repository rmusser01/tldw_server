import React from "react"
import { Tooltip } from "antd"
import { CATEGORY_SUGGESTIONS } from "../moderation-utils"

interface CategoryPickerProps {
  value: string[]
  onChange: (categories: string[]) => void
  disabled?: boolean
}

const severityColor: Record<string, string> = {
  critical: "text-red-600 dark:text-red-400",
  high: "text-orange-600 dark:text-orange-400",
  medium: "text-yellow-600 dark:text-yellow-400",
  low: "text-gray-600 dark:text-gray-400"
}

export const CategoryPicker: React.FC<CategoryPickerProps> = ({ value, onChange, disabled }) => {
  const [customInput, setCustomInput] = React.useState("")
  const customInputId = React.useId()
  const selected = new Set(value)

  const toggle = (cat: string) => {
    if (disabled) return
    const next = new Set(selected)
    if (next.has(cat)) next.delete(cat)
    else next.add(cat)
    onChange([...next])
  }

  const addCustom = () => {
    const trimmed = customInput.trim().toLowerCase()
    if (!trimmed || selected.has(trimmed)) return
    onChange([...value, trimmed])
    setCustomInput("")
  }

  return (
    <div>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {CATEGORY_SUGGESTIONS.map((cat) => {
          const isSelected = selected.has(cat.value)
          return (
            <Tooltip key={cat.value} title={cat.description}>
              <button
                type="button"
                disabled={disabled}
                onClick={() => toggle(cat.value)}
                aria-pressed={isSelected}
                className={`
                  text-left px-3 py-2 rounded-lg border text-sm transition-all
                  ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer hover:border-blue-400"}
                  ${isSelected
                    ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20 dark:border-blue-400"
                    : "border-border bg-surface/50"
                  }
                `}
              >
                <div className="flex items-center gap-2">
                  <span className={`w-3.5 h-3.5 rounded border flex items-center justify-center text-[10px] ${isSelected ? "bg-blue-500 border-blue-500 text-white" : "border-gray-300 dark:border-gray-600"}`}>
                    {isSelected ? "✓" : ""}
                  </span>
                  <span className="font-medium">{cat.label}</span>
                </div>
                <div className={`text-xs mt-0.5 ml-5.5 ${severityColor[cat.severity ?? "low"]}`}>
                  {cat.severity}
                </div>
              </button>
            </Tooltip>
          )
        })}
      </div>
      <div className="flex flex-col gap-2 mt-3 sm:flex-row">
        <label htmlFor={customInputId} className="sr-only">
          Custom category
        </label>
        <input
          id={customInputId}
          type="text"
          placeholder="Add custom category..."
          value={customInput}
          onChange={(e) => setCustomInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && addCustom()}
          disabled={disabled}
          className="flex-1 px-3 py-1.5 text-sm border border-border rounded-md bg-bg text-text placeholder:text-text-muted focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-50"
        />
        <button
          type="button"
          onClick={addCustom}
          disabled={disabled || !customInput.trim()}
          className="px-3 py-1.5 text-sm border border-border rounded-md hover:bg-surface disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Add
        </button>
      </div>
      <p className="text-xs text-text-muted mt-1.5">
        Leave all unchecked to monitor all categories.
      </p>
    </div>
  )
}
