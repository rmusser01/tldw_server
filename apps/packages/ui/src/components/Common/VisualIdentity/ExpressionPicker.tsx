import React from "react"
import { Button, Tooltip } from "antd"
import { Smile } from "lucide-react"
import {
  getVisualIdentityExpressionDisplayLabel,
  VISUAL_IDENTITY_EXPRESSION_OPTIONS
} from "@/utils/visual-identity-expressions"

export type ExpressionPickerEntry = {
  key: string
  label?: string
  hasAsset?: boolean
}

export type ExpressionPickerProps = {
  value?: string | null
  expressions?: ExpressionPickerEntry[]
  disabled?: boolean
  onChange: (expressionKey: string) => void
}

const defaultExpressions = VISUAL_IDENTITY_EXPRESSION_OPTIONS.map((option) => ({
  ...option,
  hasAsset: true
}))

export const ExpressionPicker = ({
  value,
  expressions = defaultExpressions,
  disabled = false,
  onChange
}: ExpressionPickerProps) => {
  return (
    <div
      className="flex flex-wrap items-center gap-1"
      role="group"
      aria-label="Expression picker"
    >
      {expressions.map((entry) => {
        const label =
          entry.label || getVisualIdentityExpressionDisplayLabel(entry.key) || entry.key
        const hasAsset = entry.hasAsset !== false
        const selected = value === entry.key
        const button = (
          <Button
            key={entry.key}
            size="small"
            type={selected ? "primary" : "default"}
            icon={<Smile className="h-3.5 w-3.5" aria-hidden />}
            disabled={disabled || !hasAsset}
            aria-pressed={selected}
            onClick={() => {
              if (!hasAsset || disabled) return
              onChange(entry.key)
            }}
          >
            {label}
          </Button>
        )

        if (hasAsset) return button
        return (
          <Tooltip key={entry.key} title="No expression asset">
            <span>{button}</span>
          </Tooltip>
        )
      })}
    </div>
  )
}

export default ExpressionPicker
