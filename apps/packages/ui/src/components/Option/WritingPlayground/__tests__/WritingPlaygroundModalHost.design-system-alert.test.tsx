// @vitest-environment jsdom
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { WritingPlaygroundModalHost } from "../WritingPlaygroundModalHost"
import type {
  WritingTemplateResponse,
  WritingThemeResponse,
} from "@/services/writing-playground"

type ModalHostProps = React.ComponentProps<typeof WritingPlaygroundModalHost>

const t = ((_key: string, defaultValue?: string) =>
  defaultValue ?? _key) as ModalHostProps["t"]

const makeProps = (overrides: Partial<ModalHostProps> = {}): ModalHostProps => ({
  t,
  settingsDisabled: false,
  supportsAdvancedCompat: true,
  extraBodyJsonModalOpen: false,
  setExtraBodyJsonModalOpen: vi.fn(),
  extraBodyJsonError: null,
  setExtraBodyJsonError: vi.fn(),
  extraBodyJsonDraft: "{}",
  setExtraBodyJsonDraft: vi.fn(),
  applyExtraBodyJsonDraft: vi.fn(),
  contextPreviewModalOpen: false,
  setContextPreviewModalOpen: vi.fn(),
  handleCopyContextPreview: vi.fn(async () => {}),
  handleExportContextPreview: vi.fn(),
  contextPreviewJson: "{}",
  templatesModalOpen: false,
  setTemplatesModalOpen: vi.fn(),
  templatesLoading: false,
  templatesError: null,
  templates: [],
  editingTemplate: null,
  templateImporting: false,
  templateRestoringDefaults: false,
  handleTemplateNew: vi.fn(),
  handleTemplateDuplicate: vi.fn(),
  templateDuplicateDisabled: true,
  templateFileInputRef: { current: null },
  templateExportDisabled: true,
  exportTemplate: vi.fn((_template: WritingTemplateResponse) => {}),
  templateRestoreDefaultsDisabled: true,
  handleTemplateRestoreDefaults: vi.fn(async () => {}),
  templateForm: {
    name: "",
    systemPrefix: "",
    systemSuffix: "",
    userPrefix: "",
    userSuffix: "",
    assistantPrefix: "",
    assistantSuffix: "",
    fimTemplate: "",
    isDefault: false,
  },
  templateFormDisabled: false,
  updateTemplateForm: vi.fn(),
  handleTemplateSelect: vi.fn(),
  templateSaveLoading: false,
  templateSaveDisabled: true,
  handleTemplateSave: vi.fn(),
  templateDeleteDisabled: true,
  deleteTemplateMutation: { isPending: false },
  confirmDeleteTemplate: vi.fn(),
  handleTemplateImport: vi.fn(),
  themesModalOpen: false,
  setThemesModalOpen: vi.fn(),
  themesLoading: false,
  themesError: null,
  themes: [],
  editingTheme: null,
  themeImporting: false,
  themeRestoringDefaults: false,
  handleThemeNew: vi.fn(),
  handleThemeDuplicate: vi.fn(),
  themeDuplicateDisabled: true,
  themeFileInputRef: { current: null },
  themeExportDisabled: true,
  exportTheme: vi.fn((_theme: WritingThemeResponse) => {}),
  themeRestoreDefaultsDisabled: true,
  handleThemeRestoreDefaults: vi.fn(async () => {}),
  themeForm: {
    name: "",
    className: "",
    css: "",
    order: 0,
    isDefault: false,
  },
  themeFormDisabled: false,
  updateThemeForm: vi.fn(),
  handleThemeSelect: vi.fn(),
  themeSaveLoading: false,
  themeSaveDisabled: true,
  handleThemeSave: vi.fn(),
  themeDeleteDisabled: true,
  deleteThemeMutation: { isPending: false },
  confirmDeleteTheme: vi.fn(),
  handleThemeImport: vi.fn(),
  createModalOpen: false,
  setCreateModalOpen: vi.fn(),
  createSessionMutation: {
    isPending: false,
    mutate: vi.fn(),
  },
  canCreateSession: false,
  newSessionName: "",
  setNewSessionName: vi.fn(),
  renameModalOpen: false,
  setRenameModalOpen: vi.fn(),
  renameTarget: null,
  renameSessionMutation: {
    isPending: false,
    mutate: vi.fn(),
  },
  canRenameSession: false,
  renameSessionName: "",
  setRenameSessionName: vi.fn(),
  ...overrides,
})

describe("WritingPlaygroundModalHost product-state alerts", () => {
  it("renders the extra_body JSON error through the design-system Alert", () => {
    render(
      <WritingPlaygroundModalHost
        {...makeProps({
          extraBodyJsonModalOpen: true,
          extraBodyJsonError: "Invalid JSON payload",
        })}
      />
    )

    const errorMessage = screen.getByText("Invalid JSON payload")
    expect(errorMessage.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
  })

  it("renders the template-load error through the design-system Alert", () => {
    render(
      <WritingPlaygroundModalHost
        {...makeProps({
          templatesModalOpen: true,
          templatesError: new Error("Template service failed"),
        })}
      />
    )

    const errorTitle = screen.getByText("Unable to load templates.")
    expect(errorTitle.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
  })

  it("renders the theme-load error through the design-system Alert", () => {
    render(
      <WritingPlaygroundModalHost
        {...makeProps({
          themesModalOpen: true,
          themesError: new Error("Theme service failed"),
        })}
      />
    )

    const errorTitle = screen.getByText("Unable to load themes.")
    expect(errorTitle.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
  })
})
