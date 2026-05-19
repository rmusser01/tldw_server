import React, { useMemo } from "react"
import type { ThemeColorTokens, ThemeRgbTokenKey } from "@/themes/types"
import { rgbTripleToHex } from "@/themes/conversion"

interface ThemePreviewProps {
  tokens: ThemeColorTokens
  label?: string
}

/**
 * Live preview rendered with inline CSS variable overrides scoped to a container.
 * Shows sample UI elements so the user can see how colors interact.
 */
export function ThemePreview({ tokens, label }: ThemePreviewProps) {
  const vars = useMemo(() => {
    const style: Record<string, string> = {}
    const map: Record<ThemeRgbTokenKey, string> = {
      bg: "--color-bg",
      surface: "--color-surface",
      surface2: "--color-surface-2",
      elevated: "--color-elevated",
      primary: "--color-primary",
      primaryStrong: "--color-primary-strong",
      accent: "--color-accent",
      success: "--color-success",
      warn: "--color-warn",
      danger: "--color-danger",
      muted: "--color-muted",
      border: "--color-border",
      borderStrong: "--color-border-strong",
      text: "--color-text",
      textMuted: "--color-text-muted",
      textSubtle: "--color-text-subtle",
      focus: "--color-focus",
    }
    for (const [key, cssVar] of Object.entries(map)) {
      style[cssVar] = tokens[key as ThemeRgbTokenKey]
    }
    return style as React.CSSProperties
  }, [tokens])

  const hex = (key: ThemeRgbTokenKey) => rgbTripleToHex(tokens[key])

  return (
    <div style={vars} className="rounded-lg overflow-hidden border" >
      {label && (
        <div
          className="px-3 py-1.5 text-xs font-medium"
          style={{ background: `rgb(${tokens.surface2})`, color: `rgb(${tokens.textMuted})` }}
        >
          {label}
        </div>
      )}
      <div
        className="p-3 space-y-2.5"
        style={{ background: `rgb(${tokens.bg})`, color: `rgb(${tokens.text})` }}
      >
        {/* Text hierarchy */}
        <div className="space-y-0.5">
          <div className="text-sm font-medium" style={{ color: `rgb(${tokens.text})` }}>
            Heading text
          </div>
          <div className="text-xs" style={{ color: `rgb(${tokens.textMuted})` }}>
            Muted body text for descriptions
          </div>
          <div className="text-[10px]" style={{ color: `rgb(${tokens.textSubtle})` }}>
            Subtle caption text
          </div>
        </div>

        {/* Card with button */}
        <div
          className="rounded-md p-2.5 text-xs"
          style={{
            background: `rgb(${tokens.surface})`,
            border: `1px solid rgb(${tokens.border})`,
            color: `rgb(${tokens.text})`,
          }}
        >
          <div className="flex items-center gap-2 mb-2">
            <button
              type="button"
              className="rounded px-2.5 py-1 text-[11px] font-medium text-white"
              style={{ background: hex("primary") }}
            >
              Primary
            </button>
            <button
              type="button"
              className="rounded px-2.5 py-1 text-[11px] font-medium"
              style={{
                background: `rgb(${tokens.surface2})`,
                border: `1px solid rgb(${tokens.border})`,
                color: `rgb(${tokens.text})`,
              }}
            >
              Secondary
            </button>
          </div>

          {/* Input */}
          <div
            className="rounded px-2 py-1 text-[11px]"
            style={{
              background: `rgb(${tokens.surface2})`,
              border: `1px solid rgb(${tokens.border})`,
              color: `rgb(${tokens.textMuted})`,
            }}
          >
            Placeholder text...
          </div>
        </div>

        {/* Status badges */}
        <div className="flex gap-1.5 flex-wrap">
          {(["success", "warn", "danger", "accent"] as const).map((key) => (
            <span
              key={key}
              className="rounded-full px-2 py-0.5 text-[10px] font-medium text-white"
              style={{ background: hex(key) }}
            >
              {key}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}
