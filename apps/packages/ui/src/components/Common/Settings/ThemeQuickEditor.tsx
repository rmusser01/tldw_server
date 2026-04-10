import React, { useState, useMemo, useEffect, useRef, useCallback } from "react"
import { ColorPicker, Slider, Select, Segmented, Modal, Button, Input } from "antd"
import type { Color } from "antd/es/color-picker"
import { Palette, Type, Maximize2, Layers } from "lucide-react"
import type { ThemeDefinition, ThemeColorTokens } from "@/themes/types"
import { getBuiltinPresets, getDefaultTheme } from "@/themes/presets"
import { applyThemeTokens, clearThemeTokens } from "@/themes/apply-theme"
import { rgbTripleToHex } from "@/themes/antd-theme"
import { hexToRgbTriple } from "@/themes/conversion"
import { parseRgbTriple, contrastRatio } from "@/themes/contrast"
import { deriveSurfacePalette, deriveRadii, deriveShadows } from "@/themes/derivation"
import { generateThemeId } from "@/themes/validation"
import {
  defaultTypography,
  defaultShape,
  defaultLayout,
  defaultComponents,
} from "@/themes/defaults"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ThemeQuickEditorProps {
  open: boolean
  onClose: () => void
  onSave: (theme: ThemeDefinition) => void
  /** Current dark/light mode */
  isDark: boolean
  /** Existing theme to edit (optional -- for re-entering quick mode) */
  editingTheme?: ThemeDefinition
  /** The currently active theme — restored on cancel when creating a new theme */
  activeTheme?: ThemeDefinition
  /** Called when the user clicks "Advanced..." to hand off the current preview state */
  onOpenAdvanced?: (previewTheme: ThemeDefinition) => void
}

/** The 10 lever values that drive quick mode derivation. */
interface LeverState {
  presetId: string
  primaryHex: string
  accentHex: string
  bgTintHex: string
  textContrast: number
  fontFamily: string
  roundness: number
  density: "compact" | "default" | "comfortable"
  sidebarWidth: number
  shadowIntensity: number
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const FONT_OPTIONS = [
  { value: "Inter", label: "Inter" },
  { value: "Space Grotesk", label: "Space Grotesk" },
  { value: "Arimo", label: "Arimo" },
  { value: "system-ui", label: "System UI" },
  { value: "Georgia", label: "Georgia" },
  { value: "Courier New", label: "Courier New" },
]

const DENSITY_OPTIONS = [
  { value: "compact", label: "Compact" },
  { value: "default", label: "Default" },
  { value: "comfortable", label: "Comfortable" },
]

// Minimum contrast ratio for WCAG AA normal text is 4.5:1.
const WCAG_AA_RATIO = 4.5
const WCAG_AAA_RATIO = 7

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build font-family CSS value with appropriate fallbacks. */
function buildFontFamily(primary: string): string {
  if (primary === "system-ui") return "system-ui, sans-serif"
  if (primary === "Georgia") return "Georgia, serif"
  if (primary === "Courier New") return '"Courier New", monospace'
  return `"${primary}", system-ui, sans-serif`
}

/**
 * Darken a hex color by a given percentage (0-100).
 * Simple RGB channel multiplication approach.
 */
function darkenHex(hex: string, percent: number): string {
  const triple = hexToRgbTriple(hex)
  const [r, g, b] = parseRgbTriple(triple)
  const factor = 1 - percent / 100
  const dr = Math.round(Math.max(0, r * factor))
  const dg = Math.round(Math.max(0, g * factor))
  const db = Math.round(Math.max(0, b * factor))
  return rgbTripleToHex(`${dr} ${dg} ${db}`)
}

/**
 * Lerp between two RGB triples by factor t (0 = a, 1 = b).
 */
function lerpTriple(a: string, b: string, t: number): string {
  const [ar, ag, ab] = parseRgbTriple(a)
  const [br, bg, bb] = parseRgbTriple(b)
  const clamp = (v: number) => Math.round(Math.max(0, Math.min(255, v)))
  return `${clamp(ar + (br - ar) * t)} ${clamp(ag + (bg - ag) * t)} ${clamp(ab + (bb - ab) * t)}`
}

/**
 * Derive text colors from a contrast slider value (0-100) and background triple.
 *
 * In light mode, full contrast = near-black text. Lower contrast = lighter text.
 * In dark mode, full contrast = near-white text. Lower contrast = darker text.
 *
 * Clamps minimum to ensure WCAG AA compliance on bg.
 */
function deriveTextColors(
  contrastValue: number,
  bgTriple: string,
  isDark: boolean,
): { text: string; textMuted: string; textSubtle: string } {
  const fullContrast = isDark ? "255 255 255" : "0 0 0"
  const lowContrast = isDark ? "80 80 80" : "200 200 200"

  // t=1 at contrastValue=100 means full contrast
  const t = contrastValue / 100

  const text = lerpTriple(lowContrast, fullContrast, t)
  const textMuted = lerpTriple(lowContrast, fullContrast, t * 0.75)
  const textSubtle = lerpTriple(lowContrast, fullContrast, t * 0.6)

  // Enforce WCAG AA minimum: if text doesn't meet 4.5:1 on bg, push it toward full contrast
  const ratio = contrastRatio(text, bgTriple)
  if (ratio < WCAG_AA_RATIO) {
    // Find a usable text value by pushing further toward full contrast
    let adjustedT = t
    let adjusted = text
    while (adjustedT < 1 && contrastRatio(adjusted, bgTriple) < WCAG_AA_RATIO) {
      adjustedT = Math.min(1, adjustedT + 0.05)
      adjusted = lerpTriple(lowContrast, fullContrast, adjustedT)
    }
    return {
      text: adjusted,
      textMuted: lerpTriple(lowContrast, fullContrast, adjustedT * 0.75),
      textSubtle: lerpTriple(lowContrast, fullContrast, adjustedT * 0.6),
    }
  }

  return { text, textMuted, textSubtle }
}

/**
 * Extract lever values from an existing ThemeDefinition.
 * Best-effort reverse engineering of derived values.
 */
function extractLeversFromTheme(theme: ThemeDefinition, isDark: boolean): LeverState {
  const tokens = isDark ? theme.palette.dark : theme.palette.light

  // Try to match a builtin preset
  const presets = getBuiltinPresets()
  const matchedPreset = presets.find((p) => p.id === theme.basePresetId) ?? presets[0]

  return {
    presetId: matchedPreset.id,
    primaryHex: rgbTripleToHex(tokens.primary),
    accentHex: rgbTripleToHex(tokens.accent),
    bgTintHex: rgbTripleToHex(tokens.bg),
    textContrast: 70,
    fontFamily: theme.typography.fontFamily.split(",")[0].replace(/"/g, "").trim() || "Inter",
    roundness: estimateRoundness(theme.shape),
    density: theme.layout.density,
    sidebarWidth: theme.layout.sidebarWidth,
    shadowIntensity: 50,
  }
}

/** Estimate a 0-100 roundness from shape radii. */
function estimateRoundness(shape: { radiusMd: number }): number {
  // deriveRadii maps: 0->0, 50->6, 100->12 for radiusMd
  if (shape.radiusMd <= 0) return 0
  if (shape.radiusMd <= 6) return Math.round((shape.radiusMd / 6) * 50)
  return Math.round(50 + ((shape.radiusMd - 6) / 6) * 50)
}

/**
 * Build a complete ThemeDefinition from the 10 lever values.
 */
function buildThemeFromLevers(levers: LeverState, isDark: boolean): ThemeDefinition {
  const preset = getBuiltinPresets().find((p) => p.id === levers.presetId) ?? getDefaultTheme()

  // Start from preset's palette for the tokens we don't derive
  const baseTokens = isDark ? { ...preset.palette.dark } : { ...preset.palette.light }

  // --- Colors: Primary, PrimaryStrong, Focus ---
  const primaryTriple = hexToRgbTriple(levers.primaryHex)
  const primaryStrongTriple = hexToRgbTriple(darkenHex(levers.primaryHex, 15))
  const focusTriple = primaryTriple

  // --- Colors: Accent ---
  const accentTriple = hexToRgbTriple(levers.accentHex)

  // --- Colors: Background surfaces ---
  const surfaces = deriveSurfacePalette(levers.bgTintHex, isDark)

  // --- Colors: Text ---
  const textColors = deriveTextColors(levers.textContrast, surfaces.bg, isDark)

  // --- Derive border from bg (slightly more contrast) ---
  const borderTriple = isDark
    ? lerpTriple(surfaces.bg, "255 255 255", 0.12)
    : lerpTriple(surfaces.bg, "0 0 0", 0.10)
  const borderStrongTriple = isDark
    ? lerpTriple(surfaces.bg, "255 255 255", 0.18)
    : lerpTriple(surfaces.bg, "0 0 0", 0.16)

  // --- Derive muted from text (lower opacity effect) ---
  const mutedTriple = lerpTriple(surfaces.bg, textColors.text, 0.45)

  // --- Shadows ---
  const shadows = deriveShadows(levers.shadowIntensity, isDark)

  // --- Shape: radii ---
  const radiiResult = deriveRadii(levers.roundness)

  // Assemble color tokens
  const derivedTokens: ThemeColorTokens = {
    bg: surfaces.bg,
    surface: surfaces.surface,
    surface2: surfaces.surface2,
    elevated: surfaces.elevated,
    primary: primaryTriple,
    primaryStrong: primaryStrongTriple,
    accent: accentTriple,
    success: baseTokens.success,
    warn: baseTokens.warn,
    danger: baseTokens.danger,
    muted: mutedTriple,
    border: borderTriple,
    borderStrong: borderStrongTriple,
    text: textColors.text,
    textMuted: textColors.textMuted,
    textSubtle: textColors.textSubtle,
    focus: focusTriple,
    shadowSm: shadows.shadowSm,
    shadowMd: shadows.shadowMd,
  }

  // Build the counter-mode tokens: just use the preset's other side
  // so we have a valid palette for both modes
  const otherTokens = isDark ? { ...preset.palette.light } : { ...preset.palette.dark }

  return {
    id: "",
    name: "Quick Theme",
    version: 1,
    builtin: false,
    basePresetId: levers.presetId,
    palette: isDark
      ? { light: otherTokens, dark: derivedTokens }
      : { light: derivedTokens, dark: otherTokens },
    typography: {
      ...defaultTypography(),
      fontFamily: buildFontFamily(levers.fontFamily),
    },
    shape: {
      radiusSm: radiiResult.radiusSm,
      radiusMd: radiiResult.radiusMd,
      radiusLg: radiiResult.radiusLg,
      radiusXl: radiiResult.radiusXl,
      surfaceBlur: 0,
    },
    layout: {
      ...defaultLayout(),
      sidebarWidth: levers.sidebarWidth,
      density: levers.density,
    },
    components: {
      ...defaultComponents(),
      buttonStyle: radiiResult.buttonStyle,
    },
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ThemeQuickEditor({
  open,
  onClose,
  onSave,
  isDark,
  editingTheme,
  activeTheme,
  onOpenAdvanced,
}: ThemeQuickEditorProps) {
  // Store the original theme to revert on cancel
  const originalThemeRef = useRef<ThemeDefinition | undefined>(undefined)

  // Initialize levers from editingTheme or default preset
  const initialLevers = useMemo((): LeverState => {
    if (editingTheme) {
      return extractLeversFromTheme(editingTheme, isDark)
    }
    const def = getDefaultTheme()
    const tokens = isDark ? def.palette.dark : def.palette.light
    return {
      presetId: "default",
      primaryHex: rgbTripleToHex(tokens.primary),
      accentHex: rgbTripleToHex(tokens.accent),
      bgTintHex: rgbTripleToHex(tokens.bg),
      textContrast: 70,
      fontFamily: "Inter",
      roundness: 50,
      density: "default",
      sidebarWidth: 260,
      shadowIntensity: 50,
    }
    // Only compute on open
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const [presetId, setPresetId] = useState(initialLevers.presetId)
  const [themeName, setThemeName] = useState(editingTheme?.name ?? "Quick Theme")
  const [primaryHex, setPrimaryHex] = useState(initialLevers.primaryHex)
  const [accentHex, setAccentHex] = useState(initialLevers.accentHex)
  const [bgTintHex, setBgTintHex] = useState(initialLevers.bgTintHex)
  const [textContrast, setTextContrast] = useState(initialLevers.textContrast)
  const [fontFamily, setFontFamily] = useState(initialLevers.fontFamily)
  const [roundness, setRoundness] = useState(initialLevers.roundness)
  const [density, setDensity] = useState(initialLevers.density)
  const [sidebarWidth, setSidebarWidth] = useState(initialLevers.sidebarWidth)
  const [shadowIntensity, setShadowIntensity] = useState(initialLevers.shadowIntensity)

  // Reset all state when modal opens
  useEffect(() => {
    if (open) {
      originalThemeRef.current = activeTheme
      setPresetId(initialLevers.presetId)
      setThemeName(editingTheme?.name ?? "Quick Theme")
      setPrimaryHex(initialLevers.primaryHex)
      setAccentHex(initialLevers.accentHex)
      setBgTintHex(initialLevers.bgTintHex)
      setTextContrast(initialLevers.textContrast)
      setFontFamily(initialLevers.fontFamily)
      setRoundness(initialLevers.roundness)
      setDensity(initialLevers.density)
      setSidebarWidth(initialLevers.sidebarWidth)
      setShadowIntensity(initialLevers.shadowIntensity)
    }
  }, [open, initialLevers, activeTheme])

  // Derive the full theme from all current lever values
  const derivedTheme = useMemo(
    () =>
      buildThemeFromLevers(
        {
          presetId,
          primaryHex,
          accentHex,
          bgTintHex,
          textContrast,
          fontFamily,
          roundness,
          density,
          sidebarWidth,
          shadowIntensity,
        },
        isDark,
      ),
    [
      presetId,
      primaryHex,
      accentHex,
      bgTintHex,
      textContrast,
      fontFamily,
      roundness,
      density,
      sidebarWidth,
      shadowIntensity,
      isDark,
    ],
  )

  const previewTheme = useMemo(
    (): ThemeDefinition => ({
      ...derivedTheme,
      id: editingTheme?.id ?? "",
      name: themeName.trim() || "Quick Theme",
    }),
    [derivedTheme, editingTheme?.id, themeName]
  )

  // Live-preview: apply tokens on every derivedTheme change
  useEffect(() => {
    if (!open) return
    const tokens = isDark ? previewTheme.palette.dark : previewTheme.palette.light
    applyThemeTokens(tokens, previewTheme)
  }, [isDark, open, previewTheme])

  // Check text contrast for warnings
  const contrastWarning = useMemo(() => {
    const tokens = isDark ? derivedTheme.palette.dark : derivedTheme.palette.light
    const ratio = contrastRatio(tokens.text, tokens.bg)
    if (ratio < WCAG_AA_RATIO) return "Below WCAG AA minimum (4.5:1)"
    if (ratio < WCAG_AAA_RATIO) return "Meets AA but below AAA (7:1)"
    return null
  }, [derivedTheme, isDark])

  // --- Preset change handler ---
  const handlePresetChange = useCallback(
    (newPresetId: string) => {
      const preset = getBuiltinPresets().find((p) => p.id === newPresetId)
      if (!preset) return
      setPresetId(newPresetId)
      const tokens = isDark ? preset.palette.dark : preset.palette.light
      setPrimaryHex(rgbTripleToHex(tokens.primary))
      setAccentHex(rgbTripleToHex(tokens.accent))
      setBgTintHex(rgbTripleToHex(tokens.bg))
      setTextContrast(70)
      setFontFamily(
        preset.typography.fontFamily.split(",")[0].replace(/"/g, "").trim() || "Inter",
      )
      setRoundness(estimateRoundness(preset.shape))
      setDensity(preset.layout.density)
      setSidebarWidth(preset.layout.sidebarWidth)
      setShadowIntensity(50)
    },
    [isDark],
  )

  // --- Color picker handlers ---
  const handlePrimaryChange = useCallback((_color: Color, hex: string) => {
    setPrimaryHex(hex)
  }, [])

  const handleAccentChange = useCallback((_color: Color, hex: string) => {
    setAccentHex(hex)
  }, [])

  const handleBgTintChange = useCallback((_color: Color, hex: string) => {
    setBgTintHex(hex)
  }, [])

  // --- Apply (save) ---
  const handleApply = useCallback(() => {
    const fallbackId = generateThemeId("Quick Theme")
    const resolvedName =
      themeName.trim()
      || editingTheme?.name
      || fallbackId.replace(/^quick-theme-/, "Quick Theme ")
    const theme: ThemeDefinition = {
      ...previewTheme,
      id: editingTheme?.id ?? generateThemeId(resolvedName),
      name: resolvedName,
    }
    onSave(theme)
    onClose()
  }, [editingTheme?.id, editingTheme?.name, onClose, onSave, previewTheme, themeName])

  // --- Cancel (revert) ---
  const handleCancel = useCallback(() => {
    const original = originalThemeRef.current
    const restoreTarget = original ?? activeTheme
    if (restoreTarget) {
      const tokens = isDark ? restoreTarget.palette.dark : restoreTarget.palette.light
      applyThemeTokens(tokens, restoreTarget)
    } else {
      clearThemeTokens()
    }
    onClose()
  }, [isDark, activeTheme, onClose])

  // --- Advanced handoff ---
  const handleAdvanced = useCallback(() => {
    if (onOpenAdvanced) {
      // Hand off the current preview state to the advanced editor
      onOpenAdvanced({
        ...previewTheme,
        id: editingTheme?.id ?? "",
      })
      onClose()
    } else {
      // No advanced editor callback — revert preview and close
      const original = originalThemeRef.current
      const restoreTarget = original ?? activeTheme
      if (restoreTarget) {
        const tokens = isDark ? restoreTarget.palette.dark : restoreTarget.palette.light
        applyThemeTokens(tokens, restoreTarget)
      } else {
        clearThemeTokens()
      }
      onClose()
    }
  }, [activeTheme, editingTheme?.id, isDark, onClose, onOpenAdvanced, previewTheme])

  const presetOptions = useMemo(
    () => getBuiltinPresets().map((p) => ({ value: p.id, label: p.name })),
    [],
  )

  return (
    <Modal
      open={open}
      onCancel={handleCancel}
      title="Quick Customize"
      width={600}
      destroyOnClose
      footer={
        <div className="flex items-center justify-between">
          <Button onClick={handleAdvanced}>Advanced...</Button>
          <div className="flex gap-2">
            <Button onClick={handleCancel}>Cancel</Button>
            <Button type="primary" onClick={handleApply}>
              Apply
            </Button>
          </div>
        </div>
      }
    >
      <div className="space-y-5 max-h-[65vh] overflow-y-auto pr-1">
        <div>
          <label className="text-xs text-text-muted block mb-1">Theme name</label>
          <Input
            value={themeName}
            onChange={(event) => setThemeName(event.target.value)}
            placeholder="Quick Theme"
          />
        </div>
        {/* ---- Section: Colors ---- */}
        <div>
          <div className="flex items-center gap-1.5 mb-3">
            <Palette className="h-4 w-4 text-text-muted" />
            <span className="text-sm font-medium text-text">Colors</span>
          </div>
          <div className="space-y-3">
            {/* 1. Base preset */}
            <div>
              <label className="text-xs text-text-muted block mb-1">Base preset</label>
              <Select
                className="w-full"
                value={presetId}
                onChange={handlePresetChange}
                options={presetOptions}
              />
            </div>

            {/* 2. Primary color */}
            <div className="flex items-center gap-3">
              <div className="flex-1">
                <label className="text-xs text-text-muted block mb-1">Primary color</label>
                <ColorPicker
                  value={primaryHex}
                  onChange={handlePrimaryChange}
                  showText
                  format="hex"
                />
              </div>
              {/* 3. Accent color */}
              <div className="flex-1">
                <label className="text-xs text-text-muted block mb-1">Accent color</label>
                <ColorPicker
                  value={accentHex}
                  onChange={handleAccentChange}
                  showText
                  format="hex"
                />
              </div>
            </div>

            {/* 4. Background tint */}
            <div>
              <label className="text-xs text-text-muted block mb-1">Background tint</label>
              <ColorPicker
                value={bgTintHex}
                onChange={handleBgTintChange}
                showText
                format="hex"
              />
            </div>

            {/* 5. Text contrast */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs text-text-muted">Text contrast</label>
                <span className="text-xs text-text-subtle">{textContrast}%</span>
              </div>
              <Slider
                min={0}
                max={100}
                value={textContrast}
                onChange={setTextContrast}
              />
              {contrastWarning && (
                <p className="text-xs text-warn mt-0.5">{contrastWarning}</p>
              )}
            </div>
          </div>
        </div>

        {/* ---- Section: Typography & Shape ---- */}
        <div>
          <div className="flex items-center gap-1.5 mb-3">
            <Type className="h-4 w-4 text-text-muted" />
            <span className="text-sm font-medium text-text">Typography & Shape</span>
          </div>
          <div className="space-y-3">
            {/* 6. Font family */}
            <div>
              <label className="text-xs text-text-muted block mb-1">Font family</label>
              <Select
                className="w-full"
                value={fontFamily}
                onChange={setFontFamily}
                options={FONT_OPTIONS}
              />
            </div>

            {/* 7. Roundness */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs text-text-muted">Roundness</label>
                <span className="text-xs text-text-subtle">{roundness}%</span>
              </div>
              <Slider
                min={0}
                max={100}
                value={roundness}
                onChange={setRoundness}
              />
            </div>
          </div>
        </div>

        {/* ---- Section: Layout & Feel ---- */}
        <div>
          <div className="flex items-center gap-1.5 mb-3">
            <Layers className="h-4 w-4 text-text-muted" />
            <span className="text-sm font-medium text-text">Layout & Feel</span>
          </div>
          <div className="space-y-3">
            {/* 8. Density */}
            <div>
              <label className="text-xs text-text-muted block mb-1">Density</label>
              <Segmented
                value={density}
                onChange={(val) => setDensity(val as "compact" | "default" | "comfortable")}
                options={DENSITY_OPTIONS}
                block
              />
            </div>

            {/* 9. Sidebar width */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs text-text-muted">Sidebar width</label>
                <span className="text-xs text-text-subtle">{sidebarWidth}px</span>
              </div>
              <Slider
                min={200}
                max={400}
                step={10}
                value={sidebarWidth}
                onChange={setSidebarWidth}
              />
            </div>

            {/* 10. Shadow intensity */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs text-text-muted">Shadow intensity</label>
                <span className="text-xs text-text-subtle">{shadowIntensity}%</span>
              </div>
              <Slider
                min={0}
                max={100}
                value={shadowIntensity}
                onChange={setShadowIntensity}
              />
            </div>
          </div>
        </div>
      </div>
    </Modal>
  )
}
