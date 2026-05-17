import React from "react"
import { Button, Drawer, Input, Skeleton, Switch } from "antd"
import { useTranslation } from "react-i18next"
import { shallow } from "zustand/shallow"

import { AssistantSelect } from "@/components/Common/AssistantSelect"
import { useActorStore } from "@/store/actor"
import type { ActorSettings } from "@/types/actor"
import { createDefaultActorSettings } from "@/types/actor"
import {
  getActorSettingsForChatWithCharacterFallback,
  saveActorSettingsForChat
} from "@/services/actor-settings"

import { RolePlaySetupPreview } from "./RolePlaySetupPreview"
import type { RolePlayState } from "./role-play-state"
import { getDefaultRolePlayGenerationStyle } from "./role-play-state"
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

export type RolePlaySetupApplyPayload = {
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

type RolePlaySetupDrawerProps = {
  open: boolean
  beforeState: RolePlayState
  historyId: string | null
  serverChatId: string | null
  characterId?: string | number | null
  onClose: () => void
  onApply: (payload: RolePlaySetupApplyPayload) => void | Promise<void>
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
  onClose,
  onApply,
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
  const [sceneDraft, setSceneDraft] = React.useState<ActorSettings | null>(null)
  const [clearIdentity, setClearIdentity] = React.useState(false)
  const [clearBehavior, setClearBehavior] = React.useState(false)
  const [resetGenerationStyle, setResetGenerationStyle] = React.useState(false)
  const [templatesOpen, setTemplatesOpen] = React.useState(false)
  const [stagedBehaviorTemplate, setStagedBehaviorTemplate] =
    React.useState<RolePlaySetupApplyPayload["behaviorTemplate"]>(null)
  const [stagedGenerationKey, setStagedGenerationKey] =
    React.useState<PresetKey | null>(null)

  const closeAndReturnFocus = React.useCallback(() => {
    onClose()
    returnFocusRef?.current?.focus()
  }, [onClose, returnFocusRef])

  React.useEffect(() => {
    if (!open) return

    let cancelled = false
    setLoading(true)
    setClearIdentity(false)
    setClearBehavior(false)
    setResetGenerationStyle(false)
    setStagedBehaviorTemplate(null)
    setStagedGenerationKey(null)
    setTemplatesOpen(false)

    const load = async () => {
      try {
        const actor = await getActorSettingsForChatWithCharacterFallback({
          historyId,
          serverChatId,
          characterId
        })
        if (cancelled) return
        setSceneDraft(actor)
        setSettings(actor)
        const preview = summarizeRolePlayScene(actor)
        setPreviewAndTokens(preview.prompt, preview.tokenCount)
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

  React.useEffect(() => {
    if (!open) return
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closeAndReturnFocus()
      }
    }
    document.addEventListener("keydown", handler)
    return () => {
      document.removeEventListener("keydown", handler)
    }
  }, [closeAndReturnFocus, open])

  const scenePreview = React.useMemo(
    () => summarizeRolePlayScene(sceneDraft),
    [sceneDraft]
  )

  const afterState = React.useMemo(() => {
    const stagedPreset = stagedGenerationKey
      ? PRESETS.find((preset) => preset.key === stagedGenerationKey)
      : null
    const next = {
      identity: clearIdentity ? null : beforeState.identity,
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

  const handleApply = React.useCallback(async () => {
    const payload: RolePlaySetupApplyPayload = {}
    if (clearIdentity) payload.clearIdentity = true
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
      await saveActorSettingsForChat({
        historyId,
        serverChatId,
        settings: sceneDraft
      })
      setSettings(sceneDraft)
      const preview = summarizeRolePlayScene(sceneDraft)
      setPreviewAndTokens(preview.prompt, preview.tokenCount)
    }

    await onApply(payload)
    closeAndReturnFocus()
  }, [
    clearBehavior,
    clearIdentity,
    closeAndReturnFocus,
    historyId,
    onApply,
    resetGenerationStyle,
    sceneDraft,
    serverChatId,
    setPreviewAndTokens,
    setSettings,
    stagedBehaviorTemplate,
    stagedGenerationKey
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

  const draft = getDraft(sceneDraft)
  const visibleAspects = draft.aspects.slice(0, 4)
  const activeGenerationKey =
    stagedGenerationKey ??
    (isPresetKey(beforeState.generationStyle?.key)
      ? beforeState.generationStyle.key
      : null)

  return (
    <Drawer
      placement="right"
      size={480}
      open={open}
      onClose={closeAndReturnFocus}
      title={t("playground:composer.rolePlaySetup", "Role-play setup")}>
      <div className="space-y-4" data-testid="role-play-setup-drawer">
        {loading && !sceneDraft ? <Skeleton active /> : null}

        <RolePlaySetupPreview before={beforeState} after={afterState} />

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
              return (
                <button
                  key={preset.key}
                  type="button"
                  aria-pressed={selected}
                  onClick={() => selectGenerationPreset(preset.key)}
                  className={`rounded-md border px-3 py-2 text-left text-xs transition ${
                    selected
                      ? "border-primary bg-primary/10 text-primaryStrong"
                      : "border-border bg-surface2 text-text-muted hover:border-primary/50 hover:text-text"
                  }`}>
                  <span className="font-medium">
                    {t(
                      `playground:presets.${preset.key}.label`,
                      preset.label
                    )}
                  </span>
                </button>
              )
            })}
          </div>

          <div className="flex flex-wrap gap-2">
            <Button
              disabled={!beforeState.identity}
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

        <div className="flex justify-end gap-2 border-t border-border pt-3">
          <Button onClick={closeAndReturnFocus}>
            {t("common:cancel", "Cancel")}
          </Button>
          <Button type="primary" onClick={handleApply}>
            {t("common:apply", "Apply")}
          </Button>
        </div>
      </div>
      <SystemPromptTemplatesModal
        open={templatesOpen}
        onClose={() => setTemplatesOpen(false)}
        onSelect={handleBehaviorTemplateSelect}
      />
    </Drawer>
  )
}
