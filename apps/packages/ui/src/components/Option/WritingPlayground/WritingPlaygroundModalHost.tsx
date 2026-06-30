import React from "react"
import {
  Button,
  Checkbox,
  Empty,
  Input,
  InputNumber,
  Modal,
  Skeleton,
  Tag,
} from "antd"
import type { TFunction } from "i18next"
import { Alert } from "@/components/ui/primitives"
import type { WritingTemplateResponse, WritingThemeResponse } from "@/services/writing-playground"

type WritingPlaygroundModalHostProps = {
  t: TFunction
  settingsDisabled: boolean
  supportsAdvancedCompat: boolean
  extraBodyJsonModalOpen: boolean
  setExtraBodyJsonModalOpen: React.Dispatch<React.SetStateAction<boolean>>
  extraBodyJsonError: string | null
  setExtraBodyJsonError: React.Dispatch<React.SetStateAction<string | null>>
  extraBodyJsonDraft: string
  setExtraBodyJsonDraft: React.Dispatch<React.SetStateAction<string>>
  applyExtraBodyJsonDraft: () => void
  contextPreviewModalOpen: boolean
  setContextPreviewModalOpen: React.Dispatch<React.SetStateAction<boolean>>
  handleCopyContextPreview: () => Promise<void>
  handleExportContextPreview: () => void
  contextPreviewJson: string
  templatesModalOpen: boolean
  setTemplatesModalOpen: React.Dispatch<React.SetStateAction<boolean>>
  templatesLoading: boolean
  templatesError: unknown
  templates: WritingTemplateResponse[]
  editingTemplate: WritingTemplateResponse | null
  templateImporting: boolean
  templateRestoringDefaults: boolean
  handleTemplateNew: () => void
  handleTemplateDuplicate: () => void
  templateDuplicateDisabled: boolean
  templateFileInputRef: React.RefObject<HTMLInputElement | null>
  templateExportDisabled: boolean
  exportTemplate: (template: WritingTemplateResponse) => void
  templateRestoreDefaultsDisabled: boolean
  handleTemplateRestoreDefaults: () => Promise<void>
  templateForm: {
    name: string
    systemPrefix: string
    systemSuffix: string
    userPrefix: string
    userSuffix: string
    assistantPrefix: string
    assistantSuffix: string
    fimTemplate: string
    isDefault: boolean
  }
  templateFormDisabled: boolean
  updateTemplateForm: (patch: Record<string, unknown>) => void
  handleTemplateSelect: (template: WritingTemplateResponse) => void
  templateSaveLoading: boolean
  templateSaveDisabled: boolean
  handleTemplateSave: () => void
  templateDeleteDisabled: boolean
  deleteTemplateMutation: { isPending: boolean }
  confirmDeleteTemplate: (template: WritingTemplateResponse) => void
  handleTemplateImport: (event: React.ChangeEvent<HTMLInputElement>) => void
  themesModalOpen: boolean
  setThemesModalOpen: React.Dispatch<React.SetStateAction<boolean>>
  themesLoading: boolean
  themesError: unknown
  themes: WritingThemeResponse[]
  editingTheme: WritingThemeResponse | null
  themeImporting: boolean
  themeRestoringDefaults: boolean
  handleThemeNew: () => void
  handleThemeDuplicate: () => void
  themeDuplicateDisabled: boolean
  themeFileInputRef: React.RefObject<HTMLInputElement | null>
  themeExportDisabled: boolean
  exportTheme: (theme: WritingThemeResponse) => void
  themeRestoreDefaultsDisabled: boolean
  handleThemeRestoreDefaults: () => Promise<void>
  themeForm: {
    name: string
    className: string
    css: string
    order: number
    isDefault: boolean
  }
  themeFormDisabled: boolean
  updateThemeForm: (patch: Record<string, unknown>) => void
  handleThemeSelect: (theme: WritingThemeResponse) => void
  themeSaveLoading: boolean
  themeSaveDisabled: boolean
  handleThemeSave: () => void
  themeDeleteDisabled: boolean
  deleteThemeMutation: { isPending: boolean }
  confirmDeleteTheme: (theme: WritingThemeResponse) => void
  handleThemeImport: (event: React.ChangeEvent<HTMLInputElement>) => void
  createModalOpen: boolean
  setCreateModalOpen: React.Dispatch<React.SetStateAction<boolean>>
  createSessionMutation: {
    isPending: boolean
    mutate: (name: string) => void
  }
  canCreateSession: boolean
  newSessionName: string
  setNewSessionName: React.Dispatch<React.SetStateAction<string>>
  renameModalOpen: boolean
  setRenameModalOpen: React.Dispatch<React.SetStateAction<boolean>>
  renameTarget: WritingTemplateResponse | WritingThemeResponse | { id: string } | null
  renameSessionMutation: {
    isPending: boolean
    mutate: (payload: { session: unknown; name: string }) => void
  }
  canRenameSession: boolean
  renameSessionName: string
  setRenameSessionName: React.Dispatch<React.SetStateAction<string>>
}

export const WritingPlaygroundModalHost = ({
  t,
  settingsDisabled,
  supportsAdvancedCompat,
  extraBodyJsonModalOpen,
  setExtraBodyJsonModalOpen,
  extraBodyJsonError,
  setExtraBodyJsonError,
  extraBodyJsonDraft,
  setExtraBodyJsonDraft,
  applyExtraBodyJsonDraft,
  contextPreviewModalOpen,
  setContextPreviewModalOpen,
  handleCopyContextPreview,
  handleExportContextPreview,
  contextPreviewJson,
  templatesModalOpen,
  setTemplatesModalOpen,
  templatesLoading,
  templatesError,
  templates,
  editingTemplate,
  templateImporting,
  templateRestoringDefaults,
  handleTemplateNew,
  handleTemplateDuplicate,
  templateDuplicateDisabled,
  templateFileInputRef,
  templateExportDisabled,
  exportTemplate,
  templateRestoreDefaultsDisabled,
  handleTemplateRestoreDefaults,
  templateForm,
  templateFormDisabled,
  updateTemplateForm,
  handleTemplateSelect,
  templateSaveLoading,
  templateSaveDisabled,
  handleTemplateSave,
  templateDeleteDisabled,
  deleteTemplateMutation,
  confirmDeleteTemplate,
  handleTemplateImport,
  themesModalOpen,
  setThemesModalOpen,
  themesLoading,
  themesError,
  themes,
  editingTheme,
  themeImporting,
  themeRestoringDefaults,
  handleThemeNew,
  handleThemeDuplicate,
  themeDuplicateDisabled,
  themeFileInputRef,
  themeExportDisabled,
  exportTheme,
  themeRestoreDefaultsDisabled,
  handleThemeRestoreDefaults,
  themeForm,
  themeFormDisabled,
  updateThemeForm,
  handleThemeSelect,
  themeSaveLoading,
  themeSaveDisabled,
  handleThemeSave,
  themeDeleteDisabled,
  deleteThemeMutation,
  confirmDeleteTheme,
  handleThemeImport,
  createModalOpen,
  setCreateModalOpen,
  createSessionMutation,
  canCreateSession,
  newSessionName,
  setNewSessionName,
  renameModalOpen,
  setRenameModalOpen,
  renameTarget,
  renameSessionMutation,
  canRenameSession,
  renameSessionName,
  setRenameSessionName,
}: WritingPlaygroundModalHostProps) => {
  return (
    <>
      <Modal
        title={t("option:writingPlayground.extraBodyJsonModalTitle", "Edit extra_body JSON")}
        open={extraBodyJsonModalOpen}
        onCancel={() => {
          setExtraBodyJsonModalOpen(false)
          setExtraBodyJsonError(null)
        }}
        onOk={applyExtraBodyJsonDraft}
        okText={t("common:apply", "Apply")}
        cancelText={t("common:cancel", "Cancel")}
        okButtonProps={{ disabled: settingsDisabled || !supportsAdvancedCompat }}
        width={760}>
        <div className="flex flex-col gap-3">
          <span className="text-xs text-text-muted">
            {t(
              "option:writingPlayground.extraBodyJsonModalHint",
              "Provide a JSON object to merge advanced provider-specific parameters."
            )}
          </span>
          {extraBodyJsonError ? (
            <Alert variant="error" title={extraBodyJsonError} />
          ) : null}
          <Input.TextArea
            value={extraBodyJsonDraft}
            rows={14}
            disabled={settingsDisabled || !supportsAdvancedCompat}
            onChange={(event) => {
              setExtraBodyJsonDraft(event.target.value)
              if (extraBodyJsonError) {
                setExtraBodyJsonError(null)
              }
            }}
          />
        </div>
      </Modal>

      <Modal
        title={t("option:writingPlayground.contextPreviewTitle", "Context preview")}
        open={contextPreviewModalOpen}
        onCancel={() => setContextPreviewModalOpen(false)}
        footer={[
          <Button key="copy" size="small" onClick={() => void handleCopyContextPreview()}>
            {t("option:writingPlayground.contextPreviewCopyAction", "Copy JSON")}
          </Button>,
          <Button key="export" size="small" onClick={handleExportContextPreview}>
            {t("option:writingPlayground.contextPreviewExportAction", "Export JSON")}
          </Button>,
        ]}
        width={820}>
        <div className="flex flex-col gap-3">
          <span className="text-xs text-text-muted">
            {t(
              "option:writingPlayground.contextPreviewHint",
              "Preview of assembled messages after template parsing and context injection."
            )}
          </span>
          <Input.TextArea value={contextPreviewJson} readOnly rows={18} className="font-mono" />
        </div>
      </Modal>

      <Modal
        title={t("option:writingPlayground.templatesModalTitle", "Manage templates")}
        open={templatesModalOpen}
        onCancel={() => setTemplatesModalOpen(false)}
        footer={null}
        width={900}>
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-xs text-text-muted">
              {t("option:writingPlayground.templateImportHint", "Import JSON to add or update templates.")}
            </span>
            <div className="flex items-center gap-2">
              <Button size="small" onClick={handleTemplateNew}>
                {t("option:writingPlayground.templateNewAction", "New")}
              </Button>
              <Button size="small" disabled={templateDuplicateDisabled} onClick={handleTemplateDuplicate}>
                {t("option:writingPlayground.templateDuplicateAction", "Duplicate")}
              </Button>
              <Button
                size="small"
                onClick={() => templateFileInputRef.current?.click()}
                loading={templateImporting}>
                {t("option:writingPlayground.templateImportAction", "Import")}
              </Button>
              <Button
                size="small"
                disabled={templateExportDisabled}
                onClick={() => {
                  if (editingTemplate) exportTemplate(editingTemplate)
                }}>
                {t("option:writingPlayground.templateExportAction", "Export")}
              </Button>
              <Button
                size="small"
                disabled={templateRestoreDefaultsDisabled}
                loading={templateRestoringDefaults}
                onClick={() => {
                  void handleTemplateRestoreDefaults()
                }}>
                {t("option:writingPlayground.templateRestoreDefaultsAction", "Restore defaults")}
              </Button>
            </div>
          </div>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-[240px_1fr]">
            <div className="flex flex-col gap-2">
              <span className="text-xs text-text-muted">
                {t("option:writingPlayground.templateListTitle", "Templates")}
              </span>
              {templatesLoading ? (
                <Skeleton active />
              ) : templatesError ? (
                <Alert
                  variant="error"
                  title={t("option:writingPlayground.templateError", "Unable to load templates.")}
                />
              ) : templates.length === 0 ? (
                <Empty description={t("option:writingPlayground.templateListEmpty", "No templates yet.")} />
              ) : (
                <div className="flex flex-col gap-1">
                  {templates.map((template) => {
                    const isSelected = editingTemplate?.name === template.name
                    return (
                      <div
                        key={template.name}
                        className={`cursor-pointer rounded-md px-2 py-2 transition ${
                          isSelected ? "bg-surface-hover" : "hover:bg-surface-hover/60"
                        }`}
                        role="button"
                        tabIndex={0}
                        onClick={() => handleTemplateSelect(template)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault()
                            handleTemplateSelect(template)
                          }
                        }}>
                        <div className="flex w-full items-center justify-between gap-2">
                          <span className="text-sm font-medium text-text">{template.name}</span>
                          {template.is_default ? (
                            <Tag color="blue">
                              {t("option:writingPlayground.templateDefaultTag", "Default")}
                            </Tag>
                          ) : null}
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
            <div className="flex flex-col gap-3">
              <div className="flex flex-col gap-1">
                <span className="text-xs text-text-muted">
                  {t("option:writingPlayground.templateNameLabel", "Template name")}
                </span>
                <Input
                  value={templateForm.name}
                  disabled={templateFormDisabled}
                  onChange={(event) => updateTemplateForm({ name: event.target.value })}
                />
              </div>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div className="flex flex-col gap-1">
                  <span className="text-xs text-text-muted">
                    {t("option:writingPlayground.templateSystemPrefixLabel", "System prefix")}
                  </span>
                  <Input.TextArea
                    value={templateForm.systemPrefix}
                    disabled={templateFormDisabled}
                    onChange={(event) => updateTemplateForm({ systemPrefix: event.target.value })}
                    rows={2}
                    className="font-mono text-xs"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <span className="text-xs text-text-muted">
                    {t("option:writingPlayground.templateSystemSuffixLabel", "System suffix")}
                  </span>
                  <Input.TextArea
                    value={templateForm.systemSuffix}
                    disabled={templateFormDisabled}
                    onChange={(event) => updateTemplateForm({ systemSuffix: event.target.value })}
                    rows={2}
                    className="font-mono text-xs"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <span className="text-xs text-text-muted">
                    {t("option:writingPlayground.templateUserPrefixLabel", "User prefix")}
                  </span>
                  <Input.TextArea
                    value={templateForm.userPrefix}
                    disabled={templateFormDisabled}
                    onChange={(event) => updateTemplateForm({ userPrefix: event.target.value })}
                    rows={2}
                    className="font-mono text-xs"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <span className="text-xs text-text-muted">
                    {t("option:writingPlayground.templateUserSuffixLabel", "User suffix")}
                  </span>
                  <Input.TextArea
                    value={templateForm.userSuffix}
                    disabled={templateFormDisabled}
                    onChange={(event) => updateTemplateForm({ userSuffix: event.target.value })}
                    rows={2}
                    className="font-mono text-xs"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <span className="text-xs text-text-muted">
                    {t("option:writingPlayground.templateAssistantPrefixLabel", "Assistant prefix")}
                  </span>
                  <Input.TextArea
                    value={templateForm.assistantPrefix}
                    disabled={templateFormDisabled}
                    onChange={(event) => updateTemplateForm({ assistantPrefix: event.target.value })}
                    rows={2}
                    className="font-mono text-xs"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <span className="text-xs text-text-muted">
                    {t("option:writingPlayground.templateAssistantSuffixLabel", "Assistant suffix")}
                  </span>
                  <Input.TextArea
                    value={templateForm.assistantSuffix}
                    disabled={templateFormDisabled}
                    onChange={(event) => updateTemplateForm({ assistantSuffix: event.target.value })}
                    rows={2}
                    className="font-mono text-xs"
                  />
                </div>
              </div>
              <div className="flex flex-col gap-1">
                <span className="text-xs text-text-muted">
                  {t("option:writingPlayground.templateFimTemplateLabel", "FIM template")}
                </span>
                <Input.TextArea
                  value={templateForm.fimTemplate}
                  disabled={templateFormDisabled}
                  onChange={(event) => updateTemplateForm({ fimTemplate: event.target.value })}
                  rows={3}
                  className="font-mono text-xs"
                />
                <span className="text-xs text-text-muted">
                  {t("option:writingPlayground.templateFimTemplateHint", "Use {{prefix}} and {{suffix}} placeholders.")}
                </span>
              </div>
              <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
                <Checkbox
                  checked={templateForm.isDefault}
                  disabled={templateFormDisabled}
                  onChange={(event) => updateTemplateForm({ isDefault: event.target.checked })}>
                  {t("option:writingPlayground.templateDefaultLabel", "Default template")}
                </Checkbox>
                <div className="flex items-center gap-2">
                  <Button
                    type="primary"
                    size="small"
                    onClick={handleTemplateSave}
                    loading={templateSaveLoading}
                    disabled={templateSaveDisabled}>
                    {editingTemplate
                      ? t("common:save", "Save")
                      : t("option:writingPlayground.templateCreateAction", "Create")}
                  </Button>
                  <Button
                    size="small"
                    danger
                    disabled={templateDeleteDisabled}
                    loading={deleteTemplateMutation.isPending}
                    onClick={() => {
                      if (editingTemplate) {
                        confirmDeleteTemplate(editingTemplate)
                      }
                    }}>
                    {t("common:delete", "Delete")}
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
        <input
          ref={templateFileInputRef}
          type="file"
          accept=".json,application/json"
          onChange={handleTemplateImport}
          data-testid="writing-template-import"
          className="hidden"
        />
      </Modal>

      <Modal
        title={t("option:writingPlayground.themesModalTitle", "Manage themes")}
        open={themesModalOpen}
        onCancel={() => setThemesModalOpen(false)}
        footer={null}
        width={900}>
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-xs text-text-muted">
              {t("option:writingPlayground.themeImportHint", "Import JSON to add or update themes.")}
            </span>
            <div className="flex items-center gap-2">
              <Button size="small" onClick={handleThemeNew}>
                {t("option:writingPlayground.themeNewAction", "New")}
              </Button>
              <Button size="small" disabled={themeDuplicateDisabled} onClick={handleThemeDuplicate}>
                {t("option:writingPlayground.themeDuplicateAction", "Duplicate")}
              </Button>
              <Button
                size="small"
                onClick={() => themeFileInputRef.current?.click()}
                loading={themeImporting}>
                {t("option:writingPlayground.themeImportAction", "Import")}
              </Button>
              <Button
                size="small"
                disabled={themeExportDisabled}
                onClick={() => {
                  if (editingTheme) exportTheme(editingTheme)
                }}>
                {t("option:writingPlayground.themeExportAction", "Export")}
              </Button>
              <Button
                size="small"
                disabled={themeRestoreDefaultsDisabled}
                loading={themeRestoringDefaults}
                onClick={() => {
                  void handleThemeRestoreDefaults()
                }}>
                {t("option:writingPlayground.themeRestoreDefaultsAction", "Restore defaults")}
              </Button>
            </div>
          </div>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-[240px_1fr]">
            <div className="flex flex-col gap-2">
              <span className="text-xs text-text-muted">
                {t("option:writingPlayground.themeListTitle", "Themes")}
              </span>
              {themesLoading ? (
                <Skeleton active />
              ) : themesError ? (
                <Alert
                  variant="error"
                  title={t("option:writingPlayground.themeError", "Unable to load themes.")}
                />
              ) : themes.length === 0 ? (
                <Empty description={t("option:writingPlayground.themeListEmpty", "No themes yet.")} />
              ) : (
                <div className="flex flex-col gap-1">
                  {themes.map((theme) => {
                    const isSelected = editingTheme?.name === theme.name
                    return (
                      <div
                        key={theme.name}
                        className={`cursor-pointer rounded-md px-2 py-2 transition ${
                          isSelected ? "bg-surface-hover" : "hover:bg-surface-hover/60"
                        }`}
                        role="button"
                        tabIndex={0}
                        onClick={() => handleThemeSelect(theme)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault()
                            handleThemeSelect(theme)
                          }
                        }}>
                        <div className="flex w-full items-center justify-between gap-2">
                          <span className="text-sm font-medium text-text">{theme.name}</span>
                          {theme.is_default ? (
                            <Tag color="blue">
                              {t("option:writingPlayground.themeDefaultTag", "Default")}
                            </Tag>
                          ) : null}
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
            <div className="flex flex-col gap-3">
              <div className="flex flex-col gap-1">
                <span className="text-xs text-text-muted">
                  {t("option:writingPlayground.themeNameLabel", "Theme name")}
                </span>
                <Input
                  value={themeForm.name}
                  disabled={themeFormDisabled}
                  onChange={(event) => updateThemeForm({ name: event.target.value })}
                />
              </div>
              <div className="flex flex-col gap-1">
                <span className="text-xs text-text-muted">
                  {t("option:writingPlayground.themeClassLabel", "Theme class")}
                </span>
                <Input
                  value={themeForm.className}
                  disabled={themeFormDisabled}
                  onChange={(event) => updateThemeForm({ className: event.target.value })}
                  placeholder={t("option:writingPlayground.themeClassPlaceholder", "e.g. miku-dream")}
                />
              </div>
              <div className="flex flex-col gap-1">
                <span className="text-xs text-text-muted">
                  {t("option:writingPlayground.themeCssLabel", "Theme CSS")}
                </span>
                <Input.TextArea
                  value={themeForm.css}
                  disabled={themeFormDisabled}
                  onChange={(event) => updateThemeForm({ css: event.target.value })}
                  rows={6}
                  className="font-mono text-xs"
                />
                <span className="text-xs text-text-muted">
                  {t("option:writingPlayground.themeCssHint", "CSS is scoped to .writing-playground; @import and url() are stripped.")}
                </span>
              </div>
              <div className="flex flex-col gap-1">
                <span className="text-xs text-text-muted">
                  {t("option:writingPlayground.themeOrderLabel", "Order")}
                </span>
                <InputNumber
                  value={themeForm.order}
                  disabled={themeFormDisabled}
                  onChange={(value) => updateThemeForm({ order: typeof value === "number" ? value : 0 })}
                  className="w-full"
                />
              </div>
              <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
                <Checkbox
                  checked={themeForm.isDefault}
                  disabled={themeFormDisabled}
                  onChange={(event) => updateThemeForm({ isDefault: event.target.checked })}>
                  {t("option:writingPlayground.themeDefaultLabel", "Default theme")}
                </Checkbox>
                <div className="flex items-center gap-2">
                  <Button
                    type="primary"
                    size="small"
                    onClick={handleThemeSave}
                    loading={themeSaveLoading}
                    disabled={themeSaveDisabled}>
                    {editingTheme
                      ? t("common:save", "Save")
                      : t("option:writingPlayground.themeCreateAction", "Create")}
                  </Button>
                  <Button
                    size="small"
                    danger
                    disabled={themeDeleteDisabled}
                    loading={deleteThemeMutation.isPending}
                    onClick={() => {
                      if (editingTheme) {
                        confirmDeleteTheme(editingTheme)
                      }
                    }}>
                    {t("common:delete", "Delete")}
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
        <input
          ref={themeFileInputRef}
          type="file"
          accept=".json,application/json"
          onChange={handleThemeImport}
          data-testid="writing-theme-import"
          className="hidden"
        />
      </Modal>

      <Modal
        title={t("option:writingPlayground.createSessionTitle", "New session")}
        open={createModalOpen}
        onCancel={() => setCreateModalOpen(false)}
        onOk={() => createSessionMutation.mutate(newSessionName.trim())}
        okButtonProps={{ disabled: !canCreateSession, loading: createSessionMutation.isPending }}>
        <div className="flex flex-col gap-2">
          <span className="text-sm text-text-muted">
            {t("option:writingPlayground.createSessionLabel", "Session name")}
          </span>
          <Input
            value={newSessionName}
            onChange={(event) => setNewSessionName(event.target.value)}
            placeholder={t("option:writingPlayground.createSessionPlaceholder", "e.g. Draft ideas")}
            onPressEnter={() => {
              if (canCreateSession && !createSessionMutation.isPending) {
                createSessionMutation.mutate(newSessionName.trim())
              }
            }}
          />
        </div>
      </Modal>

      <Modal
        title={t("option:writingPlayground.renameSessionTitle", "Rename session")}
        open={renameModalOpen}
        onCancel={() => {
          setRenameModalOpen(false)
        }}
        onOk={() => {
          if (renameTarget) {
            renameSessionMutation.mutate({
              session: renameTarget,
              name: renameSessionName.trim(),
            })
          }
        }}
        okButtonProps={{ disabled: !canRenameSession, loading: renameSessionMutation.isPending }}>
        <div className="flex flex-col gap-2">
          <span className="text-sm text-text-muted">
            {t("option:writingPlayground.renameSessionLabel", "Session name")}
          </span>
          <Input
            value={renameSessionName}
            onChange={(event) => setRenameSessionName(event.target.value)}
            placeholder={t("option:writingPlayground.renameSessionPlaceholder", "e.g. Revised draft")}
            onPressEnter={() => {
              if (canRenameSession && renameTarget && !renameSessionMutation.isPending) {
                renameSessionMutation.mutate({
                  session: renameTarget,
                  name: renameSessionName.trim(),
                })
              }
            }}
          />
        </div>
      </Modal>
    </>
  )
}
