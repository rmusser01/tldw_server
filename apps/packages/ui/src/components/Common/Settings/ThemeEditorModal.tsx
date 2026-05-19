import React, { useState, useCallback, useRef } from "react"
import { Modal, Input, Select, Button, message } from "antd"
import { Download, Upload } from "lucide-react"
import type { ThemeColorTokens, ThemeRgbTokenKey, ThemeDefinition, ThemeTypography, ThemeShape, ThemeLayout, ThemeComponents } from "@/themes/types"
import { getBuiltinPresets } from "@/themes/presets"
import { generateThemeId } from "@/themes/validation"
import { defaultTypography, defaultShape, defaultLayout, defaultComponents } from "@/themes/defaults"
import { parseImportedTheme } from "@/themes/import-export"
import { ColorTokenRow } from "./ColorTokenRow"
import { ThemePreview } from "./ThemePreview"

/** The 17 RGB color token keys rendered via ColorPicker rows. */
const TOKEN_KEYS: ThemeRgbTokenKey[] = [
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

interface ThemeEditorModalProps {
  open: boolean
  onClose: () => void
  onSave: (theme: ThemeDefinition) => void
  onDelete?: (id: string) => void
  /** Pass an existing custom theme to edit, or undefined for create mode */
  editingTheme?: ThemeDefinition
  /** Current dark/light mode for preview */
  isDark: boolean
}

export function ThemeEditorModal({
  open,
  onClose,
  onSave,
  onDelete,
  editingTheme,
  isDark,
}: ThemeEditorModalProps) {
  const isEditing = !!editingTheme

  const defaultPreset = getBuiltinPresets()[0]
  const initialPalette = editingTheme?.palette ?? defaultPreset.palette

  const [name, setName] = useState(editingTheme?.name ?? "My Theme")
  const [description, setDescription] = useState(editingTheme?.description ?? "")
  const [lightTokens, setLightTokens] = useState<ThemeColorTokens>({ ...initialPalette.light })
  const [darkTokens, setDarkTokens] = useState<ThemeColorTokens>({ ...initialPalette.dark })
  const [typography, setTypography] = useState<ThemeTypography>(
    editingTheme?.typography ?? defaultTypography(),
  )
  const [shape, setShape] = useState<ThemeShape>(
    editingTheme?.shape ?? defaultShape(),
  )
  const [layout, setLayout] = useState<ThemeLayout>(
    editingTheme?.layout ?? defaultLayout(),
  )
  const [components, setComponents] = useState<ThemeComponents>(
    editingTheme?.components ?? defaultComponents(),
  )
  const [basePresetId, setBasePresetId] = useState<string | undefined>(
    editingTheme?.basePresetId,
  )

  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleLightChange = useCallback(
    (key: keyof ThemeColorTokens, value: string) => {
      setLightTokens((prev) => ({ ...prev, [key]: value }))
    },
    []
  )

  const handleDarkChange = useCallback(
    (key: keyof ThemeColorTokens, value: string) => {
      setDarkTokens((prev) => ({ ...prev, [key]: value }))
    },
    []
  )

  const handlePresetBase = useCallback(
    (presetId: string) => {
      const preset = getBuiltinPresets().find((p) => p.id === presetId)
      if (preset) {
        setLightTokens({ ...preset.palette.light })
        setDarkTokens({ ...preset.palette.dark })
      }
    },
    []
  )

  const handleSave = useCallback(() => {
    if (!name.trim()) {
      void message.warning("Please enter a theme name")
      return
    }
    const theme: ThemeDefinition = {
      id: editingTheme?.id ?? generateThemeId(name),
      name: name.trim(),
      description: description.trim() || undefined,
      version: 1,
      builtin: false,
      palette: { light: { ...lightTokens }, dark: { ...darkTokens } },
      typography,
      shape,
      layout,
      components,
      basePresetId,
    }
    onSave(theme)
    onClose()
  }, [name, description, lightTokens, darkTokens, typography, shape, layout, components, basePresetId, editingTheme, onSave, onClose])

  const handleDelete = useCallback(() => {
    if (editingTheme && onDelete) {
      Modal.confirm({
        title: "Delete theme?",
        content: `"${editingTheme.name}" will be permanently removed.`,
        okText: "Delete",
        okType: "danger",
        onOk: () => {
          onDelete(editingTheme.id)
          onClose()
        },
      })
    }
  }, [editingTheme, onDelete, onClose])

  const handleExport = useCallback(() => {
    const theme: ThemeDefinition = {
      id: editingTheme?.id ?? generateThemeId(name),
      name: name.trim() || "Exported Theme",
      description: description.trim() || undefined,
      version: 1,
      builtin: false,
      palette: { light: { ...lightTokens }, dark: { ...darkTokens } },
      typography,
      shape,
      layout,
      components,
      basePresetId,
    }
    const exportData = {
      tldw_theme: true,
      version: 1,
      exported_at: new Date().toISOString(),
      theme,
    }
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `theme-${theme.id}.json`
    a.click()
    URL.revokeObjectURL(url)
  }, [name, description, lightTokens, darkTokens, typography, shape, layout, components, basePresetId, editingTheme])

  const handleImport = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      if (!file) return
      const reader = new FileReader()
      reader.onload = (e) => {
        const jsonString = e.target?.result as string
        const result = parseImportedTheme(jsonString)

        if (result.valid === false) {
          void message.error(result.error)
          return
        }

        const { theme, warnings } = result
        setName(theme.name)
        setDescription(theme.description ?? "")
        setLightTokens({ ...theme.palette.light })
        setDarkTokens({ ...theme.palette.dark })
        setTypography(theme.typography ? { ...theme.typography } : defaultTypography())
        setShape(theme.shape ? { ...theme.shape } : defaultShape())
        setLayout(theme.layout ? { ...theme.layout } : defaultLayout())
        setComponents(theme.components ? { ...theme.components } : defaultComponents())
        setBasePresetId(theme.basePresetId)

        if (warnings.length > 0) {
          void message.warning(warnings.join(" "))
        }
        void message.success(`Theme "${theme.name}" imported`)
      }
      reader.readAsText(file)
      // Reset so the same file can be re-imported
      event.target.value = ""
    },
    []
  )

  const previewTokens = isDark ? darkTokens : lightTokens

  return (
    <Modal
      open={open}
      onCancel={onClose}
      title={isEditing ? "Edit Theme" : "Create Theme"}
      width={720}
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
            <Button onClick={onClose}>Cancel</Button>
            <Button type="primary" onClick={handleSave}>
              Save
            </Button>
          </div>
        </div>
      }
    >
      <div className="space-y-4">
        {/* Header controls */}
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
            <label className="text-xs text-text-muted block mb-1">Start from preset</label>
            <Select
              className="w-full"
              placeholder="Choose a base..."
              onChange={handlePresetBase}
              options={getBuiltinPresets().map((p) => ({ value: p.id, label: p.name }))}
            />
          </div>
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

        {/* Import/Export */}
        <div className="flex gap-2">
          <Button
            size="small"
            icon={<Download className="h-3.5 w-3.5" />}
            onClick={handleExport}
          >
            Export JSON
          </Button>
          <Button
            size="small"
            icon={<Upload className="h-3.5 w-3.5" />}
            onClick={() => fileInputRef.current?.click()}
          >
            Import JSON
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            className="hidden"
            onChange={handleImport}
          />
        </div>

        {/* Color editor columns */}
        <div className="grid grid-cols-2 gap-4">
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

        {/* Preview */}
        <div>
          <h4 className="text-xs font-medium text-text-muted mb-2">Preview ({isDark ? "Dark" : "Light"})</h4>
          <ThemePreview tokens={previewTokens} />
        </div>
      </div>
    </Modal>
  )
}
