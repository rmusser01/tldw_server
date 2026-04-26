import React, { useState, useRef, useCallback } from "react"
import { Segmented, Button, Modal, message } from "antd"
import {
  Monitor,
  Moon,
  Sun,
  Pencil,
  Trash2,
  Download,
  Upload,
  Palette,
  SlidersHorizontal,
  Copy,
} from "lucide-react"
import { useTheme } from "@/hooks/useTheme"
import { rgbTripleToHex } from "@/themes/antd-theme"
import type { ThemeDefinition } from "@/themes/types"
import type { ThemeValue } from "@/services/settings/ui-settings"
import { useTranslation } from "react-i18next"
import { ThemeQuickEditor } from "./ThemeQuickEditor"
import { ThemeAdvancedEditor } from "./ThemeAdvancedEditor"
import { downloadThemeJson, parseImportedTheme } from "@/themes/import-export"
import { duplicateTheme } from "@/themes/custom-themes"

const MODE_OPTIONS: { value: ThemeValue; icon: React.ReactNode; label: string }[] = [
  { value: "system", icon: <Monitor className="h-3.5 w-3.5" />, label: "System" },
  { value: "light", icon: <Sun className="h-3.5 w-3.5" />, label: "Light" },
  { value: "dark", icon: <Moon className="h-3.5 w-3.5" />, label: "Dark" },
]

const SWATCH_KEYS = ["bg", "primary", "accent", "surface", "text"] as const
const SWATCH_LABELS: Record<(typeof SWATCH_KEYS)[number], string> = {
  bg: "Background",
  primary: "Primary",
  accent: "Accent",
  surface: "Surface",
  text: "Text"
}

function ThemeSwatch({
  theme,
  isActive,
  isDark,
  onClick,
  onEdit,
  onDelete,
  onExport,
  onDuplicateExport,
}: {
  theme: ThemeDefinition
  isActive: boolean
  isDark: boolean
  onClick: () => void
  onEdit?: () => void
  onDelete?: () => void
  onExport?: () => void
  onDuplicateExport?: () => void
}) {
  const palette = isDark ? theme.palette.dark : theme.palette.light

  return (
    <div className="relative group">
      <button
        type="button"
        onClick={onClick}
        className={[
          "flex flex-col items-center gap-1.5 rounded-lg border p-2.5 transition-colors",
          isActive
            ? "border-primary bg-surface2 ring-2 ring-primary/30"
            : "border-border bg-surface hover:border-primary/50",
        ].join(" ")}
        title={theme.description ?? theme.name}
      >
        <div className="flex gap-1">
          {SWATCH_KEYS.map((key) => (
            <div
              key={key}
              className="h-4 w-4 rounded-full border border-border"
              style={{ backgroundColor: rgbTripleToHex(palette[key]) }}
              title={SWATCH_LABELS[key]}
            />
          ))}
        </div>
        <div className="flex flex-wrap justify-center gap-1 text-[10px] leading-none text-text-subtle">
          {SWATCH_KEYS.map((key) => (
            <span key={key}>{SWATCH_LABELS[key]}</span>
          ))}
        </div>
        <span className="text-xs text-text-muted leading-none">{theme.name}</span>
      </button>
      {/* Overlay icons on hover */}
      {(onEdit || onDelete || onExport || onDuplicateExport) && (
        <div className="absolute -top-1.5 -right-1.5 hidden group-hover:flex group-focus-within:flex gap-0.5">
          {onEdit && (
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); onEdit() }}
              className="h-5 w-5 rounded-full bg-surface border border-border flex items-center justify-center shadow-sm hover:bg-surface2"
              title="Edit theme"
            >
              <Pencil className="h-2.5 w-2.5 text-text-muted" />
            </button>
          )}
          {onExport && (
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); onExport() }}
              className="h-5 w-5 rounded-full bg-surface border border-border flex items-center justify-center shadow-sm hover:bg-surface2"
              title="Export theme"
            >
              <Download className="h-2.5 w-2.5 text-text-muted" />
            </button>
          )}
          {onDuplicateExport && (
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); onDuplicateExport() }}
              className="h-5 w-5 rounded-full bg-surface border border-border flex items-center justify-center shadow-sm hover:bg-surface2"
              title="Duplicate & Export"
            >
              <Copy className="h-2.5 w-2.5 text-text-muted" />
            </button>
          )}
          {onDelete && (
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); onDelete() }}
              className="h-5 w-5 rounded-full bg-surface border border-border flex items-center justify-center shadow-sm hover:bg-danger/10"
              title="Delete theme"
            >
              <Trash2 className="h-2.5 w-2.5 text-danger" />
            </button>
          )}
        </div>
      )}
    </div>
  )
}

export function ThemePicker() {
  const { t } = useTranslation("settings")
  const {
    mode,
    modePreference,
    setModePreference,
    themeId,
    setThemeId,
    themeDefinition,
    presets,
    customThemes,
    saveCustomTheme,
    deleteCustomTheme,
  } = useTheme()
  const isDark = mode === "dark"

  // Editor modal state
  const [quickEditorOpen, setQuickEditorOpen] = useState(false)
  const [advancedEditorOpen, setAdvancedEditorOpen] = useState(false)
  const [editingTheme, setEditingTheme] = useState<ThemeDefinition | undefined>()
  const fileInputRef = useRef<HTMLInputElement>(null)

  // --- Handlers ---

  const handleSave = useCallback(
    (theme: ThemeDefinition) => {
      saveCustomTheme(theme)
      setThemeId(theme.id)
    },
    [saveCustomTheme, setThemeId],
  )

  const handleDelete = useCallback(
    (id: string) => {
      const theme = customThemes.find((entry) => entry.id === id)
      Modal.confirm({
        title: t("generalSettings.settings.themePreset.delete.title", "Delete theme?"),
        content: t(
          "generalSettings.settings.themePreset.delete.description",
          `"${theme?.name ?? "This theme"}" will be permanently removed.`
        ),
        okText: t("common:delete", "Delete"),
        okType: "danger",
        cancelText: t("common:cancel", "Cancel"),
        onOk: () => {
          deleteCustomTheme(id)
        }
      })
    },
    [customThemes, deleteCustomTheme, t],
  )

  const handleEditCustom = useCallback((theme: ThemeDefinition) => {
    setEditingTheme(theme)
    setAdvancedEditorOpen(true)
  }, [])

  const handleExportCustom = useCallback((theme: ThemeDefinition) => {
    downloadThemeJson(theme)
  }, [])

  const handleDuplicateExport = useCallback(
    (preset: ThemeDefinition) => {
      const copy = duplicateTheme(preset, preset.name + " Copy")
      downloadThemeJson(copy)
    },
    [],
  )

  const handleImportClick = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  const handleImportFile = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (!file) return

      const reader = new FileReader()
      reader.onload = () => {
        const jsonString = reader.result as string
        const result = parseImportedTheme(jsonString)

        if (result.valid) {
          saveCustomTheme(result.theme)
          setThemeId(result.theme.id)
          if (result.warnings.length > 0) {
            void message.warning(result.warnings.join(" "))
          } else {
            void message.success(`Theme "${result.theme.name}" imported successfully.`)
          }
        } else {
          void message.error(result.error)
        }
      }
      reader.onerror = () => {
        void message.error("Failed to read theme file.")
      }
      reader.readAsText(file)

      // Reset file input so re-importing the same file triggers onChange
      e.target.value = ""
    },
    [saveCustomTheme, setThemeId],
  )

  const handleOpenQuick = useCallback(() => {
    setEditingTheme(undefined)
    setQuickEditorOpen(true)
  }, [])

  const handleOpenAdvanced = useCallback(() => {
    setEditingTheme(undefined)
    setAdvancedEditorOpen(true)
  }, [])

  /** Transition from quick editor → advanced editor, carrying over the preview theme */
  const handleQuickToAdvanced = useCallback((previewTheme: ThemeDefinition) => {
    setQuickEditorOpen(false)
    setEditingTheme(previewTheme)
    setAdvancedEditorOpen(true)
  }, [])

  // Split presets into builtin and custom for separate rendering
  const builtinPresets = presets.filter((p) => p.builtin)
  const customPresets = presets.filter((p) => !p.builtin)

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-2">
        <span className="text-sm text-text">
          {t("generalSettings.settings.darkMode.label", "Appearance")}
        </span>
        <Segmented
          value={modePreference}
          onChange={(val) => setModePreference(val as ThemeValue)}
          options={MODE_OPTIONS.map((opt) => ({
            value: opt.value,
            label: (
              <div className="flex items-center gap-1.5 px-1">
                {opt.icon}
                <span>{t(`generalSettings.settings.darkMode.modes.${opt.value}`, opt.label)}</span>
              </div>
            ),
          }))}
        />
      </div>

      <div className="flex flex-col gap-2">
        <span className="text-sm text-text">
          {t("generalSettings.settings.themePreset.label", "Theme")}
        </span>
        <div className="flex flex-wrap gap-2">
          {builtinPresets.map((preset) => (
            <ThemeSwatch
              key={preset.id}
              theme={preset}
              isActive={preset.id === themeId}
              isDark={isDark}
              onClick={() => setThemeId(preset.id)}
              onDuplicateExport={() => handleDuplicateExport(preset)}
            />
          ))}
          {customPresets.map((preset) => (
            <ThemeSwatch
              key={preset.id}
              theme={preset}
              isActive={preset.id === themeId}
              isDark={isDark}
              onClick={() => setThemeId(preset.id)}
              onEdit={() => handleEditCustom(preset)}
              onExport={() => handleExportCustom(preset)}
              onDelete={() => handleDelete(preset.id)}
            />
          ))}
        </div>

        {/* Action buttons row */}
        <div className="flex flex-wrap gap-2 mt-1">
          <Button
            size="small"
            icon={<Palette className="h-3.5 w-3.5" />}
            onClick={handleOpenQuick}
          >
            Quick Customize
          </Button>
          <Button
            size="small"
            icon={<SlidersHorizontal className="h-3.5 w-3.5" />}
            onClick={handleOpenAdvanced}
          >
            Advanced Editor
          </Button>
          <Button
            size="small"
            icon={<Upload className="h-3.5 w-3.5" />}
            onClick={handleImportClick}
          >
            Import Theme
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json,application/json"
            className="hidden"
            onChange={handleImportFile}
          />
        </div>
      </div>

      <ThemeQuickEditor
        open={quickEditorOpen}
        onClose={() => setQuickEditorOpen(false)}
        onSave={handleSave}
        isDark={isDark}
        editingTheme={editingTheme}
        activeTheme={themeDefinition}
        onOpenAdvanced={handleQuickToAdvanced}
      />
      <ThemeAdvancedEditor
        open={advancedEditorOpen}
        onClose={() => setAdvancedEditorOpen(false)}
        onSave={handleSave}
        onDelete={handleDelete}
        isDark={isDark}
        editingTheme={editingTheme}
        activeTheme={themeDefinition}
      />
    </div>
  )
}
