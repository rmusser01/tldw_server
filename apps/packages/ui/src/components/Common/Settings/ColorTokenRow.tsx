import React, { useCallback } from "react"
import { ColorPicker } from "antd"
import type { Color } from "antd/es/color-picker"
import type { ThemeRgbTokenKey } from "@/themes/types"
import { rgbTripleToHex, hexToRgbTriple } from "@/themes/conversion"

const TOKEN_LABELS: Record<ThemeRgbTokenKey, string> = {
  bg: "Background",
  surface: "Surface",
  surface2: "Surface Alt",
  elevated: "Elevated",
  primary: "Primary",
  primaryStrong: "Primary Strong",
  accent: "Accent",
  success: "Success",
  warn: "Warning",
  danger: "Danger",
  muted: "Muted",
  border: "Border",
  borderStrong: "Border Strong",
  text: "Text",
  textMuted: "Text Muted",
  textSubtle: "Text Subtle",
  focus: "Focus Ring",
}

interface ColorTokenRowProps {
  tokenKey: ThemeRgbTokenKey
  value: string // RGB triple
  onChange: (key: ThemeRgbTokenKey, value: string) => void
}

export function ColorTokenRow({ tokenKey, value, onChange }: ColorTokenRowProps) {
  const hexValue = rgbTripleToHex(value)

  const handleChange = useCallback(
    (_color: Color, hex: string) => {
      onChange(tokenKey, hexToRgbTriple(hex))
    },
    [tokenKey, onChange]
  )

  return (
    <div className="flex items-center gap-2">
      <ColorPicker
        value={hexValue}
        onChange={handleChange}
        size="small"
        showText={false}
      />
      <span className="text-xs text-text-muted flex-1 min-w-0 truncate">
        {TOKEN_LABELS[tokenKey]}
      </span>
      <span className="text-[10px] font-mono text-text-subtle">{hexValue}</span>
    </div>
  )
}

export { TOKEN_LABELS }
