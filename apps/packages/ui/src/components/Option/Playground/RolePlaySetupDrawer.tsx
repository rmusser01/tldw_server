import React from "react"
import { Button, Drawer, Input, Skeleton, Switch } from "antd"
import { useTranslation } from "react-i18next"
import { X } from "lucide-react"
import { shallow } from "zustand/shallow"

import { AssistantSelect } from "@/components/Common/AssistantSelect"
import { useActorStore } from "@/store/actor"
import { createDefaultActorSettings, type ActorSettings } from "@/types/actor"
import type { AssistantSelection } from "@/types/assistant-selection"
import {
  getActorSettingsForChatWithCharacterFallback,
  saveActorSettingsForChat
} from "@/services/actor-settings"

import { RolePlaySetupPreview } from "./RolePlaySetupPreview"
import type { RolePlayState } from "./role-play-state"
import { getDefaultRolePlayGenerationStyle } from "./role-play-state"
import { SavedRolePlaySetupsPanel } from "./SavedRolePlaySetupsPanel"
import {
  PRESETS,
  SystemPromptTemplatesModal,
  type PresetKey,
  type PromptTemplate
} from "./playground-features"
import {
  clearRolePlayScene,
  resetRolePlayScene,
  summarizeRolePlayScene
} from "./role-play-scene"
import type {
  StartupTemplateBundle,
  StartupTemplateRolePlayMetadata
} from "./startup-template-bundles"

export type RolePlaySetupApplyPayload = {
  identitySelection?: AssistantSelection
  clearIdentity?: boolean
  clearBehavior?: boolean
  resetGenerationStyle?: boolean
  behaviorTemplate?: Pick<
    PromptTemplate,
    "id" | "title" | "content" | "category"
  >
  generationPresetKey?: PresetKey
  sceneSettings?: ActorSettings
}

export type RolePlaySetupSavePayload = {
  name: string
  rolePlay: StartupTemplateRolePlayMetadata
}

type RolePlaySetupDrawerProps = {
  open: boolean
  beforeState: RolePlayState
  historyId: string | null
  serverChatId: string | null
  characterId?: string | number | null
  currentSystemPrompt?: string | null
  ragPinnedResultIds?: string[]
  savedRolePlaySetups?: StartupTemplateBundle[]
  savedSetupDraftName?: string
  savedSetupNameFallback?: string
  onClose: () => void
  onApply: (payload: RolePlaySetupApplyPayload) => void | Promise<void>
  onSavedSetupDraftNameChange?: (name: string) => void
  onSaveRolePlaySetup?: (payload: RolePlaySetupSavePayload) => void
  onPreviewSavedSetup?: (id: string) => void
  onApplySavedSetup?: (setup: StartupTemplateBundle) => void | Promise<void>
  onRenameSavedSetup?: (id: string, name: string) => void
  onDeleteSavedSetup?: (id: string) => void
  returnFocusRef?: React.RefObject<HTMLElement>
}

const recomputeActive = (state: Omit<RolePlayState, "active">): RolePlayState => ({
  ...state,
  active: Boolean(
    state.identity ||
      state.behavior ||
      state.scene ||
      state.generationStyle ||
      state.context.pinnedCount > 0 ||
      state.context.hasExternalContext
  )
})

const getDraft = (draft: ActorSettings | null): ActorSettings =>
  draft ?? createDefaultActorSettings()

const isPresetKey = (value: string | null | undefined): value is PresetKey =>
  PRESETS.some((preset) => preset.key === value)

export const RolePlaySetupDrawer: React.FC<RolePlaySetupDrawerProps> = ({
  open,
  beforeState,
  historyId,
  serverChatId,
  characterId,
  currentSystemPrompt,
  ragPinnedResultIds = [],
  savedRolePlaySetups = [],
  savedSetupDraftName = "",
  savedSetupNameFallback = "New role-play setup",
  onClose,
  onApply,
  onSavedSetupDraftNameChange,
  onSaveRolePlaySetup,
  onPreviewSavedSetup,
  onApplySavedSetup,
  onRenameSavedSetup,
  onDeleteSavedSetup,
  returnFocusRef
}) => {
  const { t } = useTranslation(["playground", "common"])
  const { setSettings, setPreviewAndTokens } = useActorStore(
    (state) => ({
      setSettings: state.setSettings,
      setPreviewAndTokens: state.setPreviewAndTokens
    }),
    shallow
  )
  const [loading, setLoading] = React.useState(false)
  const [saving, setSaving] = React.useState(false)
  const applyingRef = React.useRef(false)
  const contentRef = React.useRef<HTMLFieldSetElement>(null)
  const originalSceneRef = React.useRef<ActorSettings | null>(null)
  const [sceneDraft, setSceneDraft] = React.useState<ActorSettings | null>(null)
  const [stagedIdentity, setStagedIdentity] =
    React.useState<AssistantSelection | null>(null)
  const [applyError, setApplyError] = React.useState<string | null>(null)
  const [clearIdentity, setClearIdentity] = React.useState(false)
  const [clearBehavior, setClearBehavior] = React.useState(false)
  const [resetGenerationStyle, setResetGenerationStyle] = React.useState(false)
  const [templatesOpen, setTemplatesOpen] = React.useState(false)
  const [sceneLoadError, setSceneLoadError] = React.useState<string | null>(null)
  const [stagedBehaviorTemplate, setStagedBehaviorTemplate] =
    React.useState<RolePlaySetupApplyPayload["behaviorTemplate"]>(null)
  const [stagedGenerationKey, setStagedGenerationKey] =
    React.useState<PresetKey | null>(null)

  const closeAndReturnFocus = React.useCallback(() => {
    if (applyingRef.current) return
    onClose()
    // The parent can unmount the drawer immediately; restore after its portal
    // has released background focus suppression.
    window.requestAnimationFrame(() => returnFocusRef?.current?.focus())
  }, [onClose, returnFocusRef])

  React.useEffect(() => {
    if (!open || saving || templatesOpen) return
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.isComposing) return
      const content = contentRef.current
      const dialog = content?.closest('[role="dialog"]')
      const target = event.target instanceof Element ? event.target : null
      // Shell capture handlers can stop Ant's bubble listener. Keep dismissal
      // local to this drawer, leaving nested menus and dialogs in control.
      if (!dialog || target?.closest('[role="dialog"]') !== dialog) return
      if (content?.querySelector('[aria-expanded="true"]')) return
      event.preventDefault()
      event.stopPropagation()
      closeAndReturnFocus()
    }
    window.addEventListener("keydown", handleEscape, true)
    return () => window.removeEventListener("keydown", handleEscape, true)
  }, [open, saving, templatesOpen, closeAndReturnFocus])

  React.useEffect(() => {
    if (!open) return

    let cancelled = false
    setLoading(true)
    setSceneDraft(null)
    originalSceneRef.current = null
    setStagedIdentity(null)
    setApplyError(null)
    setClearIdentity(false)
    setClearBehavior(false)
    setResetGenerationStyle(false)
    setStagedBehaviorTemplate(null)
    setStagedGenerationKey(null)
    setTemplatesOpen(false)
    setSceneLoadError(null)

    const load = async () => {
      try {
        const actor = await getActorSettingsForChatWithCharacterFallback({
          historyId,
          serverChatId,
          characterId
        })
        if (cancelled) return
        setSceneDraft(actor)
        originalSceneRef.current = actor
      } catch (error) {
        console.error("Failed to load role-play scene settings", error)
        if (!cancelled) {
          setSceneLoadError(
            t(
              "playground:composer.sceneLoadError",
              "Scene settings could not be loaded. Existing chat setup was left unchanged."
            )
          )
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void load()
    return () => {
      cancelled = true
    }
  }, [
    characterId,
    historyId,
    open,
    serverChatId,
    setPreviewAndTokens,
    setSettings
  ])

  const scenePreview = React.useMemo(
    () => summarizeRolePlayScene(sceneDraft),
    [sceneDraft]
  )

  const afterState = React.useMemo(() => {
    const stagedPreset = stagedGenerationKey
      ? PRESETS.find((preset) => preset.key === stagedGenerationKey)
      : null
    const next = {
      identity: clearIdentity ? null : stagedIdentity ?? beforeState.identity,
      behavior: clearBehavior
        ? null
        : stagedBehaviorTemplate
          ? {
              source: "template" as const,
              templateId: stagedBehaviorTemplate.id,
              title: stagedBehaviorTemplate.title,
              modified: false
            }
          : beforeState.behavior,
      scene: scenePreview.active
        ? {
            active: true,
            summary: scenePreview.summary
          }
        : null,
      generationStyle: resetGenerationStyle
        ? getDefaultRolePlayGenerationStyle()
        : stagedPreset
          ? {
              key: stagedPreset.key,
              label: String(
                t(
                  `playground:presets.${stagedPreset.key}.label`,
                  stagedPreset.label
                )
              )
            }
          : beforeState.generationStyle,
      context: beforeState.context
    }
    return recomputeActive(next)
  }, [
    beforeState.behavior,
    beforeState.context,
    beforeState.generationStyle,
    beforeState.identity,
    clearBehavior,
    clearIdentity,
    resetGenerationStyle,
    scenePreview.active,
    scenePreview.summary,
    stagedBehaviorTemplate,
    stagedGenerationKey,
    stagedIdentity,
    t
  ])

  const updateSceneEnabled = React.useCallback((checked: boolean) => {
    setSceneDraft((current) => ({
      ...getDraft(current),
      isEnabled: checked
    }))
  }, [])

  const updateSceneNotes = React.useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      setSceneDraft((current) => ({
        ...getDraft(current),
        notes: event.target.value
      }))
    },
    []
  )

  const updateAspectValue = React.useCallback(
    (aspectId: string, value: string) => {
      setSceneDraft((current) => {
        const base = getDraft(current)
        return {
          ...base,
          aspects: (base.aspects || []).map((aspect) =>
            aspect.id === aspectId ? { ...aspect, value } : aspect
          )
        }
      })
    },
    []
  )

  const applyWithScene = React.useCallback(
    async (
      scene: ActorSettings | null,
      applySettings: () => void | Promise<void>
    ) => {
      if (applyingRef.current || loading || sceneLoadError) return
      const originalScene = originalSceneRef.current
      applyingRef.current = true
      setSaving(true)
      setApplyError(null)
      let sceneSaved = false
      let applyingSettings = false
      try {
        if (scene) {
          const saved = await saveActorSettingsForChat({
            historyId,
            serverChatId,
            settings: scene
          })
          if (!saved) throw new Error("Scene settings save failed")
          sceneSaved = true
        }
        applyingSettings = true
        await applySettings()
        if (scene) {
          setSettings(scene)
          const preview = summarizeRolePlayScene(scene)
          setPreviewAndTokens(preview.prompt, preview.tokenCount)
        }
        applyingRef.current = false
        closeAndReturnFocus()
      } catch {
        let restored = true
        if (sceneSaved && originalScene) {
          try {
            restored = await saveActorSettingsForChat({
              historyId,
              serverChatId,
              settings: originalScene
            })
          } catch {
            restored = false
          }
        }
        setApplyError(
          !restored
            ? t(
                "playground:composer.rolePlayRestoreError",
                "Settings could not be applied, and the previous scene could not be restored. Keep this drawer open and retry Apply."
              )
            : applyingSettings
              ? t(
                  "playground:composer.rolePlayApplyError",
                  "Settings could not be applied. Your draft is still here; retry Apply."
                )
              : t(
                  "playground:composer.rolePlaySaveError",
                  "Scene settings could not be saved. Your draft is still here; retry Apply."
                )
        )
      } finally {
        applyingRef.current = false
        setSaving(false)
      }
    },
    [
      closeAndReturnFocus,
      historyId,
      loading,
      sceneLoadError,
      serverChatId,
      setPreviewAndTokens,
      setSettings,
      t
    ]
  )

  const handleApply = React.useCallback(async () => {
    const payload: RolePlaySetupApplyPayload = {}
    if (clearIdentity) payload.clearIdentity = true
    else if (stagedIdentity) payload.identitySelection = stagedIdentity
    if (clearBehavior) payload.clearBehavior = true
    if (!clearBehavior && stagedBehaviorTemplate) {
      payload.behaviorTemplate = stagedBehaviorTemplate
    }
    if (resetGenerationStyle) payload.resetGenerationStyle = true
    if (!resetGenerationStyle && stagedGenerationKey) {
      payload.generationPresetKey = stagedGenerationKey
    }

    if (sceneDraft) {
      payload.sceneSettings = sceneDraft
    }

    await applyWithScene(sceneDraft, () => onApply(payload))
  }, [
    applyWithScene,
    clearBehavior,
    clearIdentity,
    onApply,
    resetGenerationStyle,
    sceneDraft,
    stagedBehaviorTemplate,
    stagedGenerationKey,
    stagedIdentity
  ])

  const handleBehaviorTemplateSelect = React.useCallback((template: PromptTemplate) => {
    setStagedBehaviorTemplate({
      id: template.id,
      title: template.title,
      content: template.content,
      category: template.category
    })
    setClearBehavior(false)
    setTemplatesOpen(false)
  }, [])

  const selectGenerationPreset = React.useCallback((presetKey: PresetKey) => {
    setStagedGenerationKey(presetKey)
    setResetGenerationStyle(false)
  }, [])

  const buildRolePlaySetupSavePayload =
    React.useCallback((): RolePlaySetupSavePayload => {
      const identity =
        afterState.identity &&
        (afterState.identity.kind === "character" ||
          afterState.identity.kind === "persona") &&
        afterState.identity.id &&
        afterState.identity.name
          ? {
              kind: afterState.identity.kind,
              id: afterState.identity.id,
              name: afterState.identity.name
            }
          : null
      const behavior = afterState.behavior
        ? {
            source: afterState.behavior.source,
            templateId: afterState.behavior.templateId ?? null,
            templateTitle: afterState.behavior.title ?? null,
            templateCategory: stagedBehaviorTemplate?.category ?? null,
            systemPrompt:
              stagedBehaviorTemplate?.content ??
              String(currentSystemPrompt || "").trim(),
            modified: afterState.behavior.modified
          }
        : null
      const generationKey = isPresetKey(afterState.generationStyle?.key)
        ? afterState.generationStyle.key
        : null
      const generationPreset = generationKey
        ? PRESETS.find((preset) => preset.key === generationKey)
        : null
      const context =
        beforeState.context.pinnedCount > 0
          ? {
              ragPinnedCount: beforeState.context.pinnedCount,
              ragPinnedResultIds: ragPinnedResultIds.slice(0, 12)
            }
          : null
      const rolePlay: StartupTemplateRolePlayMetadata = {
        source: "role-play-setup",
        identity,
        behavior,
        scene: scenePreview.active ? getDraft(sceneDraft) : null,
        generation:
          generationKey && generationPreset
            ? {
                presetKey: generationKey,
                settings: generationPreset.settings
              }
            : null,
        context
      }
      return {
        name: savedSetupDraftName.trim() || savedSetupNameFallback,
        rolePlay
      }
    }, [
      afterState.behavior,
      afterState.generationStyle?.key,
      afterState.identity,
      beforeState.context.pinnedCount,
      currentSystemPrompt,
      ragPinnedResultIds,
      savedSetupDraftName,
      savedSetupNameFallback,
      sceneDraft,
      scenePreview.active,
      stagedBehaviorTemplate
    ])

  const handleSaveCurrentRolePlaySetup = React.useCallback(() => {
    if (!onSaveRolePlaySetup) return
    onSaveRolePlaySetup(buildRolePlaySetupSavePayload())
  }, [buildRolePlaySetupSavePayload, onSaveRolePlaySetup])

  const handleApplySavedSetup = React.useCallback(
    async (setup: StartupTemplateBundle) => {
      const nextScene =
        setup.rolePlay?.source === "role-play-setup"
          ? setup.rolePlay.scene ?? createDefaultActorSettings()
          : null
      await applyWithScene(nextScene, () => onApplySavedSetup?.(setup))
    },
    [applyWithScene, onApplySavedSetup]
  )

  const draft = getDraft(sceneDraft)
  const selectedIdentity: AssistantSelection | null = clearIdentity
    ? null
    : stagedIdentity ??
      (
        beforeState.identity?.kind !== "assistant" &&
        beforeState.identity?.id && beforeState.identity.name
          ? {
              kind: beforeState.identity.kind,
              id: beforeState.identity.id,
              name: beforeState.identity.name
            }
          : null
      )
  const visibleAspects = (draft.aspects ?? []).slice(0, 4)
  const activeGenerationKey =
    stagedGenerationKey ??
    (isPresetKey(beforeState.generationStyle?.key)
      ? beforeState.generationStyle.key
      : null)

  return (
    <Drawer
      placement="right"
      closeIcon={<X className="h-4 w-4 text-text-muted" aria-hidden="true" />}
      size={480}
      styles={{
        wrapper: { maxWidth: "100vw" },
        section: {
          background: "rgb(var(--color-surface))",
          color: "rgb(var(--color-text))"
        },
        header: { borderColor: "rgb(var(--color-border))" },
        body: { padding: 16 },
        footer: {
          borderColor: "rgb(var(--color-border))",
          padding: "12px 16px max(12px, env(safe-area-inset-bottom))"
        }
      }}
      open={open}
      onClose={closeAndReturnFocus}
      keyboard={!saving && !templatesOpen}
      afterOpenChange={(isOpen) => {
        if (!isOpen) returnFocusRef?.current?.focus()
      }}
      footer={
        <div className="space-y-2">
          {applyError ? (
            <p role="alert" className="rounded-md border border-danger/40 bg-danger/10 p-2 text-sm text-text">
              {applyError}
            </p>
          ) : null}
          <div className="flex justify-end gap-2">
            <Button disabled={saving} onClick={closeAndReturnFocus}>
              {t("common:cancel", "Cancel")}
            </Button>
            <Button
              type="primary"
              aria-label={t("common:apply", "Apply")}
              loading={saving}
              disabled={loading || Boolean(sceneLoadError)}
              onClick={handleApply}>
              {t("common:apply", "Apply")}
            </Button>
          </div>
        </div>
      }
      title={t("playground:composer.rolePlaySetup", "Role-play setup")}>
      <fieldset
        ref={contentRef}
        disabled={saving || loading || Boolean(sceneLoadError)}
        className="min-w-0 space-y-4"
        data-testid="role-play-setup-drawer">
        {loading && !sceneDraft ? (
          <div role="status" aria-live="polite">
            <Skeleton active />
            <span className="sr-only">
              {t(
                "playground:composer.sceneLoading",
                "Loading scene settings..."
              )}
            </span>
          </div>
        ) : null}
        {sceneLoadError ? (
          <p role="alert" className="rounded-md border border-warn/40 bg-warn/10 p-2 text-xs text-warn">
            {sceneLoadError}
          </p>
        ) : null}

        <RolePlaySetupPreview before={beforeState} after={afterState} />

        {onSaveRolePlaySetup &&
        onSavedSetupDraftNameChange &&
        onPreviewSavedSetup &&
        onApplySavedSetup &&
        onRenameSavedSetup &&
        onDeleteSavedSetup ? (
          <SavedRolePlaySetupsPanel
            setups={savedRolePlaySetups}
            draftName={savedSetupDraftName}
            nameFallback={savedSetupNameFallback}
            onDraftNameChange={onSavedSetupDraftNameChange}
            onSaveCurrent={handleSaveCurrentRolePlaySetup}
            onPreviewSetup={onPreviewSavedSetup}
            onApplySetup={handleApplySavedSetup}
            onRenameSetup={onRenameSavedSetup}
            onDeleteSetup={onDeleteSavedSetup}
            t={t}
          />
        ) : null}

        <section
          aria-label={t("playground:composer.rolePlayLayers", "Role-play layers")}
          className="space-y-2 rounded-md border border-border bg-surface p-3">
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-1">
              <div className="text-xs font-medium text-text">
                {t("playground:composer.context.character", "Character")}
              </div>
              <AssistantSelect
                variant="dropdown"
                selection={selectedIdentity}
                onSelectionChange={(selection) => {
                  setStagedIdentity(selection)
                  setClearIdentity(false)
                }}
                showLabel
                className="inline-flex min-h-9 w-full items-center justify-start gap-2 rounded-md border border-border bg-surface2 px-3 py-2 text-text"
                iconClassName="h-4 w-4"
              />
            </div>
            <div className="space-y-1">
              <div className="text-xs font-medium text-text">
                {t("playground:composer.context.behavior", "Behavior")}
              </div>
              <Button block onClick={() => setTemplatesOpen(true)}>
                {stagedBehaviorTemplate
                  ? stagedBehaviorTemplate.title
                  : t(
                      "playground:composer.chooseBehaviorTemplate",
                      "Choose behavior template"
                    )}
              </Button>
            </div>
          </div>

          <div
            role="radiogroup"
            aria-label={t(
              "playground:presets.generationStyle",
              "Generation style"
            )}
            className="grid gap-2 sm:grid-cols-4">
            {PRESETS.map((preset) => {
              const selected = activeGenerationKey === preset.key
              const label = String(
                t(
                  `playground:presets.${preset.key}.label`,
                  preset.label
                )
              )
              return (
                <label
                  key={preset.key}
                  className={`flex cursor-pointer items-center gap-2 rounded-md border px-3 py-2 text-left text-xs transition ${
                    selected
                      ? "border-primary bg-primary/10 text-primaryStrong"
                      : "border-border bg-surface2 text-text-muted hover:border-primary/50 hover:text-text"
                  }`}>
                  <input
                    type="radio"
                    name="role-play-generation-style"
                    value={preset.key}
                    checked={selected}
                    onChange={() => selectGenerationPreset(preset.key)}
                    className="h-3.5 w-3.5 accent-primary"
                  />
                  <span className="font-medium">
                    {label}
                  </span>
                </label>
              )
            })}
          </div>

          <div className="flex flex-wrap gap-2">
            <Button
              disabled={!afterState.identity}
              onClick={() => setClearIdentity(true)}>
              {t("playground:composer.clearIdentity", "Clear identity")}
            </Button>
            <Button
              disabled={!beforeState.behavior}
              onClick={() => setClearBehavior(true)}>
              {t("playground:composer.clearBehavior", "Clear behavior")}
            </Button>
            <Button
              disabled={!beforeState.generationStyle}
              onClick={() => setResetGenerationStyle(true)}>
              {t("playground:composer.resetGeneration", "Reset generation")}
            </Button>
          </div>
        </section>

        <section
          aria-label={t("playground:composer.context.scene", "Scene")}
          className="space-y-3 rounded-md border border-border bg-surface p-3">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-text">
                {t("playground:composer.context.scene", "Scene")}
              </h3>
              <p className="text-xs text-text-muted">
                {scenePreview.tokenCount > 0
                  ? `${scenePreview.tokenCount} tokens`
                  : scenePreview.active
                    ? t("playground:composer.sceneActive", "Scene active")
                    : t("playground:composer.sceneInactive", "No scene draft")}
              </p>
            </div>
            <Switch
              checked={draft.isEnabled}
              onChange={updateSceneEnabled}
              aria-label={t("playground:composer.sceneEnabled", "Scene enabled")}
            />
          </div>

          <Input.TextArea
            aria-label={t("playground:composer.sceneNotes", "Scene notes")}
            value={draft.notes}
            onChange={updateSceneNotes}
            rows={3}
            placeholder={t(
              "playground:composer.sceneNotesPlaceholder",
              "Describe the current scene context."
            )}
          />

          <div className="grid gap-2 sm:grid-cols-2">
            {visibleAspects.map((aspect) => (
              <label key={aspect.id} className="space-y-1 text-xs text-text-muted">
                <span>{aspect.name}</span>
                <Input
                  aria-label={aspect.name}
                  value={aspect.value}
                  onChange={(event) =>
                    updateAspectValue(aspect.id, event.target.value)
                  }
                />
              </label>
            ))}
          </div>

          <div className="flex flex-wrap gap-2">
            <Button onClick={() => setSceneDraft(clearRolePlayScene(sceneDraft))}>
              {t("playground:composer.clearScene", "Clear scene")}
            </Button>
            <Button onClick={() => setSceneDraft(resetRolePlayScene())}>
              {t("playground:composer.resetScene", "Reset scene")}
            </Button>
          </div>
        </section>

      </fieldset>
      <SystemPromptTemplatesModal
        open={templatesOpen}
        onClose={() => setTemplatesOpen(false)}
        onSelect={handleBehaviorTemplateSelect}
      />
    </Drawer>
  )
}
