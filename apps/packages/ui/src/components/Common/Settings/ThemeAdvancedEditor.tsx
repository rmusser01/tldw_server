import React, { useState, useCallback, useEffect, useRef, useMemo } from "react"
import { Modal, Tabs, Input, InputNumber, Select, Slider, Segmented, Button, message } from "antd"
import { RotateCcw } from "lucide-react"
import type {
  ThemeDefinition,
  ThemeColorTokens,
  ThemeTypography,
  ThemeShape,
  ThemeLayout,
  ThemeComponents,
} from "@/themes/types"
import { getBuiltinPresets } from "@/themes/presets"
import { applyThemeTokens, clearThemeTokens } from "@/themes/apply-theme"
import { generateThemeId } from "@/themes/validation"
import { rgbTripleToHex } from "@/themes/conversion"
import {
  defaultTypography,
  defaultShape,
  defaultLayout,
  defaultComponents,
} from "@/themes/defaults"
import { ColorTokenRow } from "./ColorTokenRow"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** The 17 RGB color token keys rendered via ColorPicker rows. */
const TOKEN_KEYS: (keyof ThemeColorTokens)[] = [
  "bg", "surface", "surface2", "elevated",
  "primary", "primaryStrong", "accent",
  "success", "warn", "danger", "muted",
  "border", "borderStrong",
  "text", "textMuted", "textSubtle", "focus",
]

/** Shadow tokens are CSS box-shadow strings, edited as plain text. */
const SHADOW_KEYS: { key: "shadowSm" | "shadowMd"; label: string }[] = [
  { key: "shadowSm", label: "Shadow Small" },
  { key: "shadowMd", label: "Shadow Medium" },
]

const FONT_OPTIONS = [
  { value: "Inter", label: "Inter" },
  { value: "Space Grotesk", label: "Space Grotesk" },
  { value: "Arimo", label: "Arimo" },
  { value: "system-ui", label: "System UI" },
  { value: "Georgia", label: "Georgia" },
  { value: "Courier New", label: "Courier New" },
]

const MONO_FONT_OPTIONS = [
  { value: "Courier New", label: "Courier New" },
  { value: "monospace", label: "monospace" },
]

const DENSITY_OPTIONS = [
  { value: "compact", label: "Compact" },
  { value: "default", label: "Default" },
  { value: "comfortable", label: "Comfortable" },
]

const ANIMATION_OPTIONS = [
  { value: "none", label: "None" },
  { value: "subtle", label: "Subtle" },
  { value: "normal", label: "Normal" },
]

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

/** Build mono font-family CSS value. */
function buildMonoFontFamily(primary: string): string {
  if (primary === "monospace") return "monospace"
  return `"${primary}", monospace`
}

/** Extract the primary font name from a CSS font-family string. */
function extractPrimaryFont(fontFamily: string): string {
  return fontFamily.split(",")[0].replace(/"/g, "").trim() || "Inter"
}

/** Extract the mono font name from a CSS font-family-mono string. */
function extractMonoFont(fontFamilyMono: string): string {
  const primary = fontFamilyMono.split(",")[0].replace(/"/g, "").trim()
  return primary || "Courier New"
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface ThemeAdvancedEditorProps {
  open: boolean
  onClose: () => void
  onSave: (theme: ThemeDefinition) => void
  onDelete?: (id: string) => void
  isDark: boolean
  editingTheme?: ThemeDefinition
  activeTheme?: ThemeDefinition
}

export function ThemeAdvancedEditor({
  open,
  onClose,
  onSave,
  onDelete,
  isDark,
  editingTheme,
  activeTheme,
}: ThemeAdvancedEditorProps) {
  const editingThemeId = editingTheme?.id?.trim() ?? ""
  const isEditing = editingThemeId.length > 0

  const defaultPreset = getBuiltinPresets()[0]
  const initialPalette = editingTheme?.palette ?? defaultPreset.palette

  // ---- Top-level fields ----
  const [name, setName] = useState(editingTheme?.name ?? "My Theme")
  const [description, setDescription] = useState(editingTheme?.description ?? "")

  // ---- Colors ----
  const [lightTokens, setLightTokens] = useState<ThemeColorTokens>({ ...initialPalette.light })
  const [darkTokens, setDarkTokens] = useState<ThemeColorTokens>({ ...initialPalette.dark })

  // ---- Typography ----
  const [typography, setTypography] = useState<ThemeTypography>(
    editingTheme?.typography ? { ...editingTheme.typography } : defaultTypography(),
  )

  // ---- Shape ----
  const [shape, setShape] = useState<ThemeShape>(
    editingTheme?.shape ? { ...editingTheme.shape } : defaultShape(),
  )

  // ---- Layout ----
  const [layout, setLayout] = useState<ThemeLayout>(
    editingTheme?.layout ? { ...editingTheme.layout } : defaultLayout(),
  )

  // ---- Components ----
  const [components, setComponents] = useState<ThemeComponents>(
    editingTheme?.components ? { ...editingTheme.components } : defaultComponents(),
  )

  // Store the original theme for revert-on-cancel
  const originalThemeRef = useRef<ThemeDefinition | undefined>(undefined)

  // Reset all state when modal opens
  useEffect(() => {
    if (open) {
      originalThemeRef.current = activeTheme
      const palette = editingTheme?.palette ?? defaultPreset.palette
      setName(editingTheme?.name ?? "My Theme")
      setDescription(editingTheme?.description ?? "")
      setLightTokens({ ...palette.light })
      setDarkTokens({ ...palette.dark })
      setTypography(editingTheme?.typography ? { ...editingTheme.typography } : defaultTypography())
      setShape(editingTheme?.shape ? { ...editingTheme.shape } : defaultShape())
      setLayout(editingTheme?.layout ? { ...editingTheme.layout } : defaultLayout())
      setComponents(editingTheme?.components ? { ...editingTheme.components } : defaultComponents())
    }
  }, [open, editingTheme, activeTheme, defaultPreset.palette])

  // ---- Build current theme for live preview ----
  const currentTheme = useMemo((): ThemeDefinition => ({
    id: editingThemeId,
    name: name.trim() || "Untitled",
    version: 1,
    builtin: false,
    palette: { light: { ...lightTokens }, dark: { ...darkTokens } },
    typography,
    shape,
    layout,
    components,
    basePresetId: editingTheme?.basePresetId,
  }), [
    components,
    darkTokens,
    editingTheme?.basePresetId,
    editingThemeId,
    layout,
    lightTokens,
    name,
    shape,
    typography,
  ])

  // Live-preview: apply tokens on every change
  useEffect(() => {
    if (!open) return
    const tokens = isDark ? currentTheme.palette.dark : currentTheme.palette.light
    applyThemeTokens(tokens, currentTheme)
  }, [currentTheme, isDark, open])

  // ---- Color handlers ----
  const handleLightChange = useCallback(
    (key: keyof ThemeColorTokens, value: string) => {
      setLightTokens((prev) => ({ ...prev, [key]: value }))
    },
    [],
  )

  const handleDarkChange = useCallback(
    (key: keyof ThemeColorTokens, value: string) => {
      setDarkTokens((prev) => ({ ...prev, [key]: value }))
    },
    [],
  )

  // ---- Typography handlers ----
  const handleFontFamilyChange = useCallback((value: string) => {
    setTypography((prev) => ({ ...prev, fontFamily: buildFontFamily(value) }))
  }, [])

  const handleMonoFontChange = useCallback((value: string) => {
    setTypography((prev) => ({ ...prev, fontFamilyMono: buildMonoFontFamily(value) }))
  }, [])

  const handleFontSizeChange = useCallback(
    (key: keyof Pick<ThemeTypography, "fontSizeBody" | "fontSizeMessage" | "fontSizeCaption" | "fontSizeLabel">, value: number | null) => {
      if (value == null) return
      setTypography((prev) => ({ ...prev, [key]: value }))
    },
    [],
  )

  // ---- Shape handlers ----
  const handleRadiusChange = useCallback(
    (key: keyof Pick<ThemeShape, "radiusSm" | "radiusMd" | "radiusLg" | "radiusXl">, value: number) => {
      setShape((prev) => ({ ...prev, [key]: value }))
    },
    [],
  )

  const handleSurfaceBlurChange = useCallback((value: number) => {
    setShape((prev) => ({ ...prev, surfaceBlur: value }))
  }, [])

  // ---- Layout handlers ----
  const handleLayoutChange = useCallback(
    (key: keyof Omit<ThemeLayout, "density">, value: number) => {
      setLayout((prev) => ({ ...prev, [key]: value }))
    },
    [],
  )

  const handleDensityChange = useCallback((value: string | number) => {
    setLayout((prev) => ({ ...prev, density: value as ThemeLayout["density"] }))
  }, [])

  // ---- Components handlers ----
  const handleButtonStyleChange = useCallback((value: ThemeComponents["buttonStyle"]) => {
    setComponents((prev) => ({ ...prev, buttonStyle: value }))
  }, [])

  const handleInputStyleChange = useCallback((value: ThemeComponents["inputStyle"]) => {
    setComponents((prev) => ({ ...prev, inputStyle: value }))
  }, [])

  const handleCardStyleChange = useCallback((value: ThemeComponents["cardStyle"]) => {
    setComponents((prev) => ({ ...prev, cardStyle: value }))
  }, [])

  const handleAnimationChange = useCallback((value: string | number) => {
    setComponents((prev) => ({ ...prev, animationSpeed: value as ThemeComponents["animationSpeed"] }))
  }, [])

  // ---- Reset handlers (per tab) ----
  const handleResetColors = useCallback(() => {
    const preset = defaultPreset
    setLightTokens({ ...preset.palette.light })
    setDarkTokens({ ...preset.palette.dark })
  }, [defaultPreset])

  const handleResetTypography = useCallback(() => {
    setTypography(defaultTypography())
  }, [])

  const handleResetShape = useCallback(() => {
    setShape(defaultShape())
  }, [])

  const handleResetLayout = useCallback(() => {
    setLayout(defaultLayout())
  }, [])

  const handleResetComponents = useCallback(() => {
    setComponents(defaultComponents())
  }, [])

  // ---- Save ----
  const handleSave = useCallback(() => {
    if (!name.trim()) {
      void message.warning("Please enter a theme name")
      return
    }
    const theme: ThemeDefinition = {
      id: editingThemeId || generateThemeId(name.trim()),
      name: name.trim(),
      description: description.trim() || undefined,
      version: 1,
      builtin: false,
      palette: { light: { ...lightTokens }, dark: { ...darkTokens } },
      typography,
      shape,
      layout,
      components,
      basePresetId: editingTheme?.basePresetId,
    }
    onSave(theme)
    onClose()
  }, [
    components,
    darkTokens,
    description,
    editingTheme?.basePresetId,
    editingThemeId,
    layout,
    lightTokens,
    name,
    onClose,
    onSave,
    shape,
    typography,
  ])

  // ---- Cancel (revert) ----
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

  // ---- Delete ----
  const handleDelete = useCallback(() => {
    if (editingTheme && editingThemeId && onDelete) {
      Modal.confirm({
        title: "Delete theme?",
        content: `"${editingTheme.name}" will be permanently removed.`,
        okText: "Delete",
        okType: "danger",
        onOk: () => {
          onDelete(editingThemeId)
          // Restore the original theme (undo live preview of the deleted theme)
          const original = originalThemeRef.current
          const restoreTarget = original ?? activeTheme
          if (restoreTarget) {
            const tokens = isDark ? restoreTarget.palette.dark : restoreTarget.palette.light
            applyThemeTokens(tokens, restoreTarget)
          } else {
            clearThemeTokens()
          }
          onClose()
        },
      })
    }
  }, [editingTheme, editingThemeId, onDelete, isDark, activeTheme, onClose])

  // ---- Reset button shared component ----
  const ResetButton = useCallback(
    ({ onClick }: { onClick: () => void }) => (
      <Button
        size="small"
        type="text"
        icon={<RotateCcw className="h-3 w-3" />}
        onClick={onClick}
        className="text-text-muted"
      >
        Reset to defaults
      </Button>
    ),
    [],
  )

  // ---- Tab content ----
  const tabItems = useMemo(() => [
    {
      key: "colors",
      label: "Colors",
      children: (
        <div className="space-y-4">
          <div className="flex justify-end">
            <ResetButton onClick={handleResetColors} />
          </div>
          <div className="grid grid-cols-2 gap-4">
            {/* Light palette */}
            <div>
              <h4 className="text-xs font-medium text-text-muted mb-2">Light Palette</h4>
              <div className="space-y-1.5">
                {TOKEN_KEYS.map((key) => (
                  <ColorTokenRow
                    key={key}
                    tokenKey={key}
                    value={lightTokens[key]}
                    onChange={handleLightChange}
                  />
                ))}
              </div>
              <h4 className="text-xs font-medium text-text-muted mb-2 mt-3">Shadows</h4>
              <div className="space-y-1.5">
                {SHADOW_KEYS.map(({ key, label }) => (
                  <div key={key} className="flex items-center gap-2">
                    <span className="text-xs text-text-muted flex-1 min-w-0 truncate">{label}</span>
                    <Input
                      size="small"
                      value={lightTokens[key]}
                      onChange={(e) => handleLightChange(key, e.target.value)}
                      className="w-48 font-mono text-[10px]"
                    />
                  </div>
                ))}
              </div>
            </div>
            {/* Dark palette */}
            <div>
              <h4 className="text-xs font-medium text-text-muted mb-2">Dark Palette</h4>
              <div className="space-y-1.5">
                {TOKEN_KEYS.map((key) => (
                  <ColorTokenRow
                    key={key}
                    tokenKey={key}
                    value={darkTokens[key]}
                    onChange={handleDarkChange}
                  />
                ))}
              </div>
              <h4 className="text-xs font-medium text-text-muted mb-2 mt-3">Shadows</h4>
              <div className="space-y-1.5">
                {SHADOW_KEYS.map(({ key, label }) => (
                  <div key={key} className="flex items-center gap-2">
                    <span className="text-xs text-text-muted flex-1 min-w-0 truncate">{label}</span>
                    <Input
                      size="small"
                      value={darkTokens[key]}
                      onChange={(e) => handleDarkChange(key, e.target.value)}
                      className="w-48 font-mono text-[10px]"
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      ),
    },
    {
      key: "typography",
      label: "Typography",
      children: (
        <div className="space-y-4">
          <div className="flex justify-end">
            <ResetButton onClick={handleResetTypography} />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-text-muted block mb-1">Font family</label>
              <Select
                className="w-full"
                value={extractPrimaryFont(typography.fontFamily)}
                onChange={handleFontFamilyChange}
                options={FONT_OPTIONS}
              />
            </div>
            <div>
              <label className="text-xs text-text-muted block mb-1">Mono font</label>
              <Select
                className="w-full"
                value={extractMonoFont(typography.fontFamilyMono)}
                onChange={handleMonoFontChange}
                options={MONO_FONT_OPTIONS}
              />
            </div>
          </div>
          <div>
            <h4 className="text-xs font-medium text-text-muted mb-2">Font Sizes</h4>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-text-muted block mb-1">Body</label>
                <InputNumber
                  size="small"
                  min={8}
                  max={32}
                  step={1}
                  value={typography.fontSizeBody}
                  onChange={(v) => handleFontSizeChange("fontSizeBody", v)}
                  className="w-full"
                  suffix="px"
                />
              </div>
              <div>
                <label className="text-xs text-text-muted block mb-1">Message</label>
                <InputNumber
                  size="small"
                  min={8}
                  max={32}
                  step={1}
                  value={typography.fontSizeMessage}
                  onChange={(v) => handleFontSizeChange("fontSizeMessage", v)}
                  className="w-full"
                  suffix="px"
                />
              </div>
              <div>
                <label className="text-xs text-text-muted block mb-1">Caption</label>
                <InputNumber
                  size="small"
                  min={8}
                  max={32}
                  step={1}
                  value={typography.fontSizeCaption}
                  onChange={(v) => handleFontSizeChange("fontSizeCaption", v)}
                  className="w-full"
                  suffix="px"
                />
              </div>
              <div>
                <label className="text-xs text-text-muted block mb-1">Label</label>
                <InputNumber
                  size="small"
                  min={8}
                  max={32}
                  step={1}
                  value={typography.fontSizeLabel}
                  onChange={(v) => handleFontSizeChange("fontSizeLabel", v)}
                  className="w-full"
                  suffix="px"
                />
              </div>
            </div>
          </div>
          {/* Typography preview */}
          <div>
            <h4 className="text-xs font-medium text-text-muted mb-2">Preview</h4>
            <div
              className="rounded-md border border-border p-3 space-y-2"
              style={{ fontFamily: typography.fontFamily }}
            >
              <p style={{ fontSize: typography.fontSizeBody }}>
                Body ({typography.fontSizeBody}px): The quick brown fox jumps over the lazy dog.
              </p>
              <p style={{ fontSize: typography.fontSizeMessage }}>
                Message ({typography.fontSizeMessage}px): Hello, how can I help you today?
              </p>
              <p style={{ fontSize: typography.fontSizeCaption }} className="text-text-muted">
                Caption ({typography.fontSizeCaption}px): Last edited 2 hours ago
              </p>
              <p style={{ fontSize: typography.fontSizeLabel }} className="text-text-subtle">
                Label ({typography.fontSizeLabel}px): OPTIONAL FIELD
              </p>
            </div>
          </div>
        </div>
      ),
    },
    {
      key: "shape",
      label: "Shape",
      children: (
        <div className="space-y-4">
          <div className="flex justify-end">
            <ResetButton onClick={handleResetShape} />
          </div>
          <div>
            <h4 className="text-xs font-medium text-text-muted mb-3">Border Radii</h4>
            <div className="space-y-3">
              {([
                { key: "radiusSm" as const, label: "Small" },
                { key: "radiusMd" as const, label: "Medium" },
                { key: "radiusLg" as const, label: "Large" },
                { key: "radiusXl" as const, label: "Extra Large" },
              ]).map(({ key, label }) => (
                <div key={key}>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-xs text-text-muted">{label}</label>
                    <span className="text-xs text-text-subtle">{shape[key]}px</span>
                  </div>
                  <Slider
                    min={0}
                    max={40}
                    value={shape[key]}
                    onChange={(v) => handleRadiusChange(key, v)}
                  />
                </div>
              ))}
            </div>
          </div>
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs text-text-muted">Surface Blur</label>
              <span className="text-xs text-text-subtle">{shape.surfaceBlur}px</span>
            </div>
            <Slider
              min={0}
              max={20}
              value={shape.surfaceBlur}
              onChange={handleSurfaceBlurChange}
            />
          </div>
          {/* Shape preview */}
          <div>
            <h4 className="text-xs font-medium text-text-muted mb-2">Preview</h4>
            <div className="flex items-end gap-3">
              {([
                { key: "radiusSm" as const, label: "Sm", size: 40 },
                { key: "radiusMd" as const, label: "Md", size: 52 },
                { key: "radiusLg" as const, label: "Lg", size: 64 },
                { key: "radiusXl" as const, label: "XL", size: 76 },
              ]).map(({ key, label, size }) => (
                <div key={key} className="flex flex-col items-center gap-1">
                  <div
                    style={{
                      width: size,
                      height: size,
                      borderRadius: shape[key],
                      backgroundColor: `rgb(${isDark ? darkTokens.primary : lightTokens.primary})`,
                    }}
                  />
                  <span className="text-[10px] text-text-subtle">
                    {label} ({shape[key]}px)
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      ),
    },
    {
      key: "layout",
      label: "Layout",
      children: (
        <div className="space-y-4">
          <div className="flex justify-end">
            <ResetButton onClick={handleResetLayout} />
          </div>
          <div className="space-y-3">
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs text-text-muted">Sidebar width</label>
                <span className="text-xs text-text-subtle">{layout.sidebarWidth}px</span>
              </div>
              <Slider
                min={150}
                max={600}
                step={10}
                value={layout.sidebarWidth}
                onChange={(v) => handleLayoutChange("sidebarWidth", v)}
              />
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs text-text-muted">Sidebar collapsed width</label>
                <span className="text-xs text-text-subtle">{layout.sidebarCollapsedWidth}px</span>
              </div>
              <Slider
                min={40}
                max={120}
                step={4}
                value={layout.sidebarCollapsedWidth}
                onChange={(v) => handleLayoutChange("sidebarCollapsedWidth", v)}
              />
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs text-text-muted">Header height</label>
                <span className="text-xs text-text-subtle">{layout.headerHeight}px</span>
              </div>
              <Slider
                min={40}
                max={80}
                step={4}
                value={layout.headerHeight}
                onChange={(v) => handleLayoutChange("headerHeight", v)}
              />
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs text-text-muted">Content max width</label>
                <span className="text-xs text-text-subtle">{layout.contentMaxWidth}px</span>
              </div>
              <Slider
                min={600}
                max={1400}
                step={20}
                value={layout.contentMaxWidth}
                onChange={(v) => handleLayoutChange("contentMaxWidth", v)}
              />
            </div>
            <div>
              <label className="text-xs text-text-muted block mb-1">Density</label>
              <Segmented
                value={layout.density}
                onChange={handleDensityChange}
                options={DENSITY_OPTIONS}
                block
              />
            </div>
          </div>
        </div>
      ),
    },
    {
      key: "components",
      label: "Components",
      children: (
        <div className="space-y-5">
          <div className="flex justify-end">
            <ResetButton onClick={handleResetComponents} />
          </div>

          {/* Button style */}
          <div>
            <h4 className="text-xs font-medium text-text-muted mb-2">Button Style</h4>
            <div className="flex gap-3">
              {([
                { value: "square" as const, label: "Square", radius: 2 },
                { value: "rounded" as const, label: "Rounded", radius: 6 },
                { value: "pill" as const, label: "Pill", radius: 999 },
              ]).map(({ value, label, radius }) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={components.buttonStyle === value}
                  onClick={() => handleButtonStyleChange(value)}
                  className={`
                    flex-1 py-2 px-3 text-sm font-medium transition-all cursor-pointer
                    ${components.buttonStyle === value
                      ? "ring-2 ring-primary bg-surface text-text"
                      : "bg-surface2 text-text-muted hover:bg-surface hover:text-text"
                    }
                  `}
                  style={{
                    borderRadius: radius,
                    border: `1px solid rgb(${isDark ? darkTokens.border : lightTokens.border})`,
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Input style */}
          <div>
            <h4 className="text-xs font-medium text-text-muted mb-2">Input Style</h4>
            <div className="flex gap-3">
              {([
                { value: "bordered" as const, label: "Bordered" },
                { value: "underlined" as const, label: "Underlined" },
                { value: "filled" as const, label: "Filled" },
              ]).map(({ value, label }) => {
                const isActive = components.inputStyle === value
                const borderColor = `rgb(${isDark ? darkTokens.border : lightTokens.border})`
                const surfaceColor = `rgb(${isDark ? darkTokens.surface2 : lightTokens.surface2})`
                let inputPreviewStyle: React.CSSProperties = {}
                if (value === "bordered") {
                  inputPreviewStyle = {
                    border: `1px solid ${borderColor}`,
                    borderRadius: shape.radiusSm,
                    padding: "4px 8px",
                    background: "transparent",
                  }
                } else if (value === "underlined") {
                  inputPreviewStyle = {
                    border: "none",
                    borderBottom: `2px solid ${borderColor}`,
                    borderRadius: 0,
                    padding: "4px 8px",
                    background: "transparent",
                  }
                } else {
                  inputPreviewStyle = {
                    border: "1px solid transparent",
                    borderRadius: shape.radiusSm,
                    padding: "4px 8px",
                    background: surfaceColor,
                  }
                }
                return (
                  <button
                    key={value}
                    type="button"
                    aria-pressed={isActive}
                    onClick={() => handleInputStyleChange(value)}
                    className={`
                      flex-1 p-3 rounded-md cursor-pointer transition-all text-left
                      ${isActive
                        ? "ring-2 ring-primary bg-surface"
                        : "bg-surface2 hover:bg-surface"
                      }
                    `}
                  >
                    <div className="text-[10px] text-text-subtle mb-1.5">{label}</div>
                    <div
                      style={inputPreviewStyle}
                      className="text-xs text-text-muted"
                    >
                      Placeholder...
                    </div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Card style */}
          <div>
            <h4 className="text-xs font-medium text-text-muted mb-2">Card Style</h4>
            <div className="flex gap-3">
              {([
                { value: "flat" as const, label: "Flat" },
                { value: "elevated" as const, label: "Elevated" },
                { value: "outlined" as const, label: "Outlined" },
              ]).map(({ value, label }) => {
                const isActive = components.cardStyle === value
                const activeTokens = isDark ? darkTokens : lightTokens
                let cardPreviewStyle: React.CSSProperties = {
                  borderRadius: shape.radiusMd,
                  padding: "8px 12px",
                }
                if (value === "flat") {
                  cardPreviewStyle = {
                    ...cardPreviewStyle,
                    background: `rgb(${activeTokens.surface})`,
                    border: "1px solid transparent",
                  }
                } else if (value === "elevated") {
                  cardPreviewStyle = {
                    ...cardPreviewStyle,
                    background: `rgb(${activeTokens.surface})`,
                    border: "1px solid transparent",
                    boxShadow: activeTokens.shadowSm,
                  }
                } else {
                  cardPreviewStyle = {
                    ...cardPreviewStyle,
                    background: "transparent",
                    border: `1px solid rgb(${activeTokens.border})`,
                  }
                }
                return (
                  <button
                    key={value}
                    type="button"
                    aria-pressed={isActive}
                    onClick={() => handleCardStyleChange(value)}
                    className={`
                      flex-1 p-3 rounded-md cursor-pointer transition-all text-left
                      ${isActive
                        ? "ring-2 ring-primary bg-surface"
                        : "bg-surface2 hover:bg-surface"
                      }
                    `}
                  >
                    <div className="text-[10px] text-text-subtle mb-1.5">{label}</div>
                    <div style={cardPreviewStyle}>
                      <div className="text-xs text-text">Card content</div>
                      <div className="text-[10px] text-text-muted mt-1">Description</div>
                    </div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Animation speed */}
          <div>
            <h4 className="text-xs font-medium text-text-muted mb-2">Animation Speed</h4>
            <Segmented
              value={components.animationSpeed}
              onChange={handleAnimationChange}
              options={ANIMATION_OPTIONS}
              block
            />
          </div>
        </div>
      ),
    },
  ], [
    lightTokens, darkTokens, typography, shape, layout, components, isDark,
    handleLightChange, handleDarkChange,
    handleFontFamilyChange, handleMonoFontChange, handleFontSizeChange,
    handleRadiusChange, handleSurfaceBlurChange,
    handleLayoutChange, handleDensityChange,
    handleButtonStyleChange, handleInputStyleChange, handleCardStyleChange, handleAnimationChange,
    handleResetColors, handleResetTypography, handleResetShape, handleResetLayout, handleResetComponents,
    ResetButton,
  ])

  return (
    <Modal
      open={open}
      onCancel={handleCancel}
      title={isEditing ? "Edit Theme (Advanced)" : "Create Theme (Advanced)"}
      width={800}
      destroyOnClose
      footer={
        <div className="flex items-center justify-between">
          <div className="flex gap-2">
            {isEditing && onDelete && (
              <Button danger onClick={handleDelete}>
                Delete
              </Button>
            )}
          </div>
          <div className="flex gap-2">
            <Button onClick={handleCancel}>Cancel</Button>
            <Button type="primary" onClick={handleSave}>
              Save
            </Button>
          </div>
        </div>
      }
    >
      {/* Always-visible header fields */}
      <div className="space-y-3 mb-4">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-text-muted block mb-1">Name</label>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My Theme"
              maxLength={40}
            />
          </div>
          <div>
            <label className="text-xs text-text-muted block mb-1">Description (optional)</label>
            <Input
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="A short description..."
              maxLength={100}
            />
          </div>
        </div>
      </div>

      {/* Tabbed editor */}
      <Tabs
        items={tabItems}
        className="theme-advanced-tabs"
        size="small"
        style={{ maxHeight: "55vh", overflow: "auto" }}
      />
    </Modal>
  )
}
