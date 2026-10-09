import React from "react"
import { Button, Drawer, Input, Modal, Skeleton, Switch } from "antd"
import { useTranslation } from "react-i18next"
import { X } from "lucide-react"
import { shallow } from "zustand/shallow"

import { AssistantSelect } from "@/components/Common/AssistantSelect"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { useActorStore } from "@/store/actor"
import { createDefaultActorSettings, type ActorSettings } from "@/types/actor"
import type { AssistantSelection } from "@/types/assistant-selection"
import { nativeCharacterPreflightMessage } from "@/hooks/chat/native-character-preflight"
import { captureChatDraftSource } from "@/hooks/chat/chat-draft-transition"
import { useHistorySelectionContext } from "@/hooks/chat/useHistorySelection"
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

export type RolePlayApplyContext = {
  /** Exact component-local proposal; never a persisted or server operation ID. */
  applicationId?: object
  isCurrent?: () => boolean
  onDestination?: (destination: {
    historyId: string | null
    serverChatId: string | null
    isCurrent: () => boolean
  }) => void
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
  onApply: (payload: RolePlaySetupApplyPayload, context?: RolePlayApplyContext) => boolean | void | Promise<boolean | void>
  /** Capture the initiating owner action after its synchronous identity transition. */
  captureApplyGuard?: () => () => boolean
  onSavedSetupDraftNameChange?: (name: string) => void
  onSaveRolePlaySetup?: (payload: RolePlaySetupSavePayload) => void
  onPreviewSavedSetup?: (id: string) => void
  onApplySavedSetup?: (setup: StartupTemplateBundle, context?: RolePlayApplyContext) => boolean | void | Promise<boolean | void>
  onSavedSetupApplied?: () => void
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
  captureApplyGuard,
  onSavedSetupDraftNameChange,
  onSaveRolePlaySetup,
  onPreviewSavedSetup,
  onApplySavedSetup,
  onSavedSetupApplied,
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
  const [modal, modalContextHolder] = Modal.useModal()
  const editRevisionRef = React.useRef(0)
  const baselineSceneRef = React.useRef<string | undefined>(undefined)
  const [loading, setLoading] = React.useState(false)
  const [saving, setSaving] = React.useState(false)
  const applyingRef = React.useRef(false)
  const applyingDestinationRef = React.useRef<{
    historyId: string | null
    serverChatId: string | null
    isCurrent: () => boolean
  } | null>(null)
  const acceptedSettingsRef = React.useRef<{
    intent: string
    destination: NonNullable<typeof applyingDestinationRef.current>
  } | null>(null)
  const settingsApplicationRef = React.useRef<{
    id: object
    intent: string
    destination: NonNullable<typeof applyingDestinationRef.current> | null
  } | null>(null)
  const activeAttemptRef = React.useRef<object | null>(null)
  const mountedRef = React.useRef(true)
  const lifetimeRef = React.useRef({ open, revision: 0 })
  if (lifetimeRef.current.open !== open) {
    lifetimeRef.current = { open, revision: lifetimeRef.current.revision + 1 }
    acceptedSettingsRef.current = null
    settingsApplicationRef.current = null
    applyingDestinationRef.current = null
    activeAttemptRef.current = null
    applyingRef.current = false
  }
  const scopeRef = React.useRef({ open, historyId, serverChatId })
  scopeRef.current = { open, historyId, serverChatId }
  React.useEffect(() => {
    mountedRef.current = true
    return () => { mountedRef.current = false }
  }, [])
  const contentRef = React.useRef<HTMLFieldSetElement>(null)
  const [sceneDraft, setSceneDraft] = React.useState<ActorSettings | null>(null)
  const [stagedIdentity, setStagedIdentity] =
    React.useState<AssistantSelection | null>(null)
  const [applyError, setApplyError] = React.useState<string | null>(null)
  const [clearIdentity, setClearIdentity] = React.useState(false)
  const [clearBehavior, setClearBehavior] = React.useState(false)
  const [resetGenerationStyle, setResetGenerationStyle] = React.useState(false)
  const [templatesOpen, setTemplatesOpen] = React.useState(false)
  const [sceneLoadError, setSceneLoadError] = React.useState<string | null>(null)
  const [sceneReadRevision, setSceneReadRevision] = React.useState(0)
  const readSourceRef = React.useRef<(() => boolean) | null>(null)
  const dismissalSourceRef = React.useRef<(() => boolean) | null>(null)
  const loadedSceneRef = React.useRef<(() => boolean) | null>(null)
  const draftRevision = useStoreMessageOption((state) => state.chatDraftRevision)
  const restoreRevision = usePlaygroundSessionStore(
    (state) => state.restoreRevision
  )
  const selectionRevision = usePlaygroundSessionStore(
    (state) => state.sourceSelectionRevision
  )
  const historySelection = useHistorySelectionContext()
  const selectionSignal = historySelection?.getSignal()
  const captureSelectionFence = historySelection?.fence
  const sceneView = {
    open,
    historyId,
    serverChatId,
    characterId,
    draftRevision,
    restoreRevision,
    selectionRevision,
    sceneReadRevision,
    selectionSignal,
    captureSelectionFence,
    captureApplyGuard
  }
  const sceneViewRef = React.useRef(sceneView)
  if (
    Object.keys(sceneView).some(
      (key) =>
        sceneView[key as keyof typeof sceneView] !==
        sceneViewRef.current[key as keyof typeof sceneView]
    )
  )
    sceneViewRef.current = sceneView
  const renderedSceneView = sceneViewRef.current

  const [stagedBehaviorTemplate, setStagedBehaviorTemplate] =
    React.useState<RolePlaySetupApplyPayload["behaviorTemplate"]>(null)
  const [stagedGenerationKey, setStagedGenerationKey] =
    React.useState<PresetKey | null>(null)

  const dismissalSource = dismissalSourceRef.current
  const loadedScene = loadedSceneRef.current
  const isDraftCurrent = React.useCallback(() => mountedRef.current &&
    !applyingRef.current && sceneViewRef.current === renderedSceneView &&
    Boolean(loadedScene?.()), [loadedScene, renderedSceneView])
  const stage = React.useCallback((change: () => void) => {
    if (!isDraftCurrent()) return
    editRevisionRef.current++
    change()
  }, [isDraftCurrent])

  const closeAndReturnFocus = React.useCallback(() => {
    if (applyingRef.current) return
    onClose()
    // The parent can unmount the drawer immediately; restore after its portal
    // has released background focus suppression.
    window.requestAnimationFrame(() => returnFocusRef?.current?.focus())
  }, [onClose, returnFocusRef])

  const requestClose = React.useCallback(() => {
    if (
      !mountedRef.current ||
      !scopeRef.current.open ||
      applyingRef.current ||
      sceneViewRef.current !== renderedSceneView ||
      dismissalSourceRef.current !== dismissalSource ||
      !dismissalSource?.()
    )
      return
    const dirty =
      stagedIdentity !== null ||
      clearIdentity ||
      clearBehavior ||
      resetGenerationStyle ||
      stagedBehaviorTemplate !== null ||
      stagedGenerationKey !== null ||
      (baselineSceneRef.current !== undefined &&
        JSON.stringify(sceneDraft) !== baselineSceneRef.current)
    if (!dirty) return closeAndReturnFocus()
    const revision = editRevisionRef.current
    const lifetimeRevision = lifetimeRef.current.revision
    const canDismiss = () =>
      dismissalSourceRef.current === dismissalSource &&
      dismissalSource() &&
      mountedRef.current &&
      scopeRef.current.open &&
      !applyingRef.current &&
      sceneViewRef.current === renderedSceneView &&
      lifetimeRef.current.revision === lifetimeRevision &&
      editRevisionRef.current === revision
    modal.confirm({
      title: t(
        "playground:composer.discardSetupTitle",
        "Discard unsaved role-play setup?"
      ),
      content: t(
        "playground:composer.discardSetupHelp",
        "Your staged setup will be discarded. Identity and behavior already applied remain in effect."
      ),
      okText: t("common:discard", "Discard"),
      cancelText: t("common:keepEditing", "Keep editing"),
      onOk: () => {
        if (canDismiss()) closeAndReturnFocus()
      }
    })
  }, [
    clearBehavior,
    clearIdentity,
    closeAndReturnFocus,
    dismissalSource,
    modal,
    renderedSceneView,
    resetGenerationStyle,
    sceneDraft,
    stagedBehaviorTemplate,
    stagedGenerationKey,
    stagedIdentity,
    t
  ])

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
      requestClose()
    }
    window.addEventListener("keydown", handleEscape, true)
    return () => window.removeEventListener("keydown", handleEscape, true)
  }, [open, saving, templatesOpen, requestClose])

  React.useEffect(() => {
    if (!open) return
    const ownDestination = applyingDestinationRef.current
    if (
      ownDestination?.historyId === historyId &&
      ownDestination.serverChatId === serverChatId &&
      ownDestination.isCurrent()
    ) {
      return
    }

    acceptedSettingsRef.current = null
    settingsApplicationRef.current = null
    applyingDestinationRef.current = null
    activeAttemptRef.current = null
    applyingRef.current = false
    setSaving(false)
    let cancelled = false
    const applyIsCurrent = captureApplyGuard?.() ?? (() => true)
    const hadAuthority = applyIsCurrent()
    const selectionIsCurrent = captureSelectionFence?.() ?? (() => true)
    const sourceView = sceneViewRef.current
    const { isCurrent: sourceIsCurrent } = captureChatDraftSource(
      () =>
        !cancelled &&
        selectionIsCurrent() &&
        sceneViewRef.current === sourceView &&
        usePlaygroundSessionStore.getState().restoreRevision ===
          restoreRevision &&
        usePlaygroundSessionStore.getState().sourceSelectionRevision ===
          selectionRevision
    )
    const isCurrent = () => sourceIsCurrent() && applyIsCurrent()
    readSourceRef.current = isCurrent
    // Unverified error panels can dismiss without granting settings authority.
    dismissalSourceRef.current = () =>
      sourceIsCurrent() && (!hadAuthority || applyIsCurrent())
    loadedSceneRef.current = null
    baselineSceneRef.current = undefined
    setApplyError(null)
    if (!isCurrent()) {
      setLoading(false)
      setSceneLoadError(
        t(
          "playground:composer.sceneLoadError",
          "Scene settings could not be loaded. Existing chat setup was left unchanged."
        )
      )
      return () => {
        cancelled = true
      }
    }
    setLoading(true)
    setSceneDraft(null)
    setStagedIdentity(null)
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
        if (!isCurrent()) return
        loadedSceneRef.current = isCurrent
        baselineSceneRef.current = JSON.stringify(actor)
        setSceneDraft(actor)
      } catch (error) {
        console.error("Failed to load role-play scene settings", error)
        if (isCurrent()) {
          setSceneLoadError(
            t(
              "playground:composer.sceneLoadError",
              "Scene settings could not be loaded. Existing chat setup was left unchanged."
            )
          )
        }
      } finally {
        if (isCurrent()) {
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
    draftRevision,
    restoreRevision,
    selectionRevision,
    sceneReadRevision,
    selectionSignal,
    captureSelectionFence,
    captureApplyGuard,
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
    stage(() => setSceneDraft((current) => ({ ...getDraft(current), isEnabled: checked })))
  }, [stage])

  const updateSceneNotes = React.useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      const value = event.target.value
      stage(() => setSceneDraft((current) => ({ ...getDraft(current), notes: value })))
    },
    [stage]
  )

  const updateAspectValue = React.useCallback(
    (aspectId: string, value: string) => {
      stage(() => setSceneDraft((current) => {
        const base = getDraft(current)
        return {
          ...base,
          aspects: (base.aspects || []).map((aspect) =>
            aspect.id === aspectId ? { ...aspect, value } : aspect
          )
        }
      }))
    },
    [stage]
  )

  const applyWithScene = React.useCallback(
    async (
      scene: ActorSettings | null,
      intent: string,
      applySettings: (
        context: RolePlayApplyContext
      ) => boolean | void | Promise<boolean | void>,
      onApplied?: () => void
    ) => {
      if (
        sceneViewRef.current !== renderedSceneView ||
        applyingRef.current ||
        loading ||
        sceneLoadError ||
        !loadedSceneRef.current?.()
      )
        return
      const attempt = {}
      activeAttemptRef.current = attempt
      applyingRef.current = true
      setSaving(true)
      setApplyError(null)
      const lifetimeRevision = lifetimeRef.current.revision
      let settingsAccepted = false
      const sourceIsCurrent = loadedSceneRef.current
      let isCurrent = () =>
        Boolean(sourceIsCurrent?.()) &&
        mountedRef.current &&
        lifetimeRef.current.revision === lifetimeRevision &&
        scopeRef.current.open &&
        scopeRef.current.historyId === historyId &&
        scopeRef.current.serverChatId === serverChatId
      try {
        let application = settingsApplicationRef.current
        if (
          !application ||
          application.intent !== intent ||
          (application.destination && !application.destination.isCurrent())
        ) {
          application = { id: {}, intent, destination: null }
          settingsApplicationRef.current = application
        }
        const proposal = application
        let destination = { historyId, serverChatId }
        let notified = false
        const onDestination: NonNullable<
          RolePlayApplyContext["onDestination"]
        > = (next) => {
          notified = true
          destination = next
          const restoreRevision =
            usePlaygroundSessionStore.getState().restoreRevision
          isCurrent = () => {
            const current = useStoreMessageOption.getState()
            const props = scopeRef.current
            const propsMatch =
              (props.historyId === historyId &&
                props.serverChatId === serverChatId) ||
              (props.historyId === next.historyId &&
                props.serverChatId === next.serverChatId)
            return (
              mountedRef.current &&
              lifetimeRef.current.revision === lifetimeRevision &&
              next.isCurrent() &&
              props.open &&
              propsMatch &&
              current.historyId === next.historyId &&
              current.serverChatId === next.serverChatId &&
              usePlaygroundSessionStore.getState().restoreRevision ===
                restoreRevision
            )
          }
          if (isCurrent()) {
            // Model retirement blocks writes, but must still allow this source to dismiss.
            const applyIsCurrent = captureApplyGuard?.() ?? (() => true)
            const selectionIsCurrent = captureSelectionFence?.() ?? (() => true)
            dismissalSourceRef.current = captureChatDraftSource(
              () => applyIsCurrent() && selectionIsCurrent()
            ).isCurrent
          }
          proposal.destination = { ...next, isCurrent }
          applyingDestinationRef.current = proposal.destination
          loadedSceneRef.current = isCurrent
        }
        const accepted = acceptedSettingsRef.current
        if (accepted?.intent === intent && accepted.destination.isCurrent()) {
          destination = accepted.destination
          isCurrent = accepted.destination.isCurrent
          applyingDestinationRef.current = accepted.destination
        } else {
          acceptedSettingsRef.current = null
          const applied = applySettings({
            applicationId: proposal.id,
            isCurrent: () => Boolean(isCurrent()),
            onDestination
          })
          // Existing synchronous callers may not emit the optional notification.
          if (!notified) {
            const current = useStoreMessageOption.getState()
            onDestination({
              historyId: current.historyId,
              serverChatId: current.serverChatId,
              isCurrent: captureApplyGuard?.() ?? (() => true)
            })
          }
          if ((await applied) === false || !isCurrent()) return
          acceptedSettingsRef.current = {
            intent,
            destination: { ...destination, isCurrent }
          }
        }
        settingsAccepted = true
        if (scene) {
          const saved = await saveActorSettingsForChat({
            historyId: destination.historyId,
            serverChatId: destination.serverChatId,
            settings: scene
          })
          if (!isCurrent()) return
          if (!saved) throw new Error("Scene settings save failed")
          setSettings(scene)
          if (!isCurrent()) return
          const preview = summarizeRolePlayScene(scene)
          setPreviewAndTokens(preview.prompt, preview.tokenCount)
          if (!isCurrent()) return
        }
        if (!isCurrent()) return
        onApplied?.()
        if (!isCurrent()) return
        acceptedSettingsRef.current = null
        settingsApplicationRef.current = null
        applyingRef.current = false
        closeAndReturnFocus()
      } catch (error) {
        if (!isCurrent()) return
        setApplyError(
          nativeCharacterPreflightMessage(error, t) ??
            (settingsAccepted
              ? t(
                  "playground:composer.rolePlayScenePartialSaveError",
                  "Identity and behavior were applied, but scene settings could not be saved. Your draft is still here; retry Apply."
                )
              : t(
                  "playground:composer.rolePlayApplyError",
                  "Settings could not be applied. Your draft is still here; retry Apply."
                ))
        )
      } finally {
        if (activeAttemptRef.current === attempt) {
          activeAttemptRef.current = null
          applyingDestinationRef.current =
            acceptedSettingsRef.current?.destination.isCurrent()
              ? acceptedSettingsRef.current.destination
              : settingsApplicationRef.current?.destination?.isCurrent()
                ? settingsApplicationRef.current.destination
                : null
          applyingRef.current = false
          if (
            mountedRef.current &&
            lifetimeRef.current.revision === lifetimeRevision
          )
            setSaving(false)
        }
      }
    },
    [
      closeAndReturnFocus,
      captureApplyGuard,
      captureSelectionFence,
      historyId,
      loading,
      sceneLoadError,
      renderedSceneView,
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

    const intent = JSON.stringify(payload)
    if (sceneDraft) {
      payload.sceneSettings = sceneDraft
    }

    await applyWithScene(sceneDraft, intent, (context) => onApply(payload, context))
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
    stage(() => {
    setStagedBehaviorTemplate({
      id: template.id,
      title: template.title,
      content: template.content,
      category: template.category
    })
    setClearBehavior(false)
    setTemplatesOpen(false)
    })
  }, [stage])

  const selectGenerationPreset = React.useCallback((presetKey: PresetKey) => {
    stage(() => {
      setStagedGenerationKey(presetKey)
      setResetGenerationStyle(false)
    })
  }, [stage])

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
    if (!onSaveRolePlaySetup || !isDraftCurrent()) return
    onSaveRolePlaySetup(buildRolePlaySetupSavePayload())
  }, [buildRolePlaySetupSavePayload, isDraftCurrent, onSaveRolePlaySetup])

  const handleApplySavedSetup = React.useCallback(
    async (setup: StartupTemplateBundle) => {
      const nextScene =
        setup.rolePlay?.source === "role-play-setup"
          ? setup.rolePlay.scene ?? createDefaultActorSettings()
          : null
      const intent = JSON.stringify({
        ...setup,
        rolePlay: setup.rolePlay ? { ...setup.rolePlay, scene: null } : null
      })
      await applyWithScene(
        nextScene,
        intent,
        (context) => onApplySavedSetup?.(setup, context),
        onSavedSetupApplied
      )
    },
    [applyWithScene, onApplySavedSetup, onSavedSetupApplied]
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
      onClose={requestClose}
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
          {sceneLoadError && !readSourceRef.current?.() ? (
            <p>
              {t(
                "playground:composer.sceneSourceUnavailable",
                "Close this panel and check your account verification and selected chat before reopening scene settings."
              )}
            </p>
          ) : null}
          <div className="flex justify-end gap-2">
            {sceneLoadError && readSourceRef.current?.() ? (
              <Button
                onClick={() => {
                  if (
                    sceneViewRef.current === renderedSceneView &&
                    readSourceRef.current?.()
                  ) setSceneReadRevision(value => value + 1)
                }}>
                {t("common:retry", "Retry")}
              </Button>
            ) : null}
            <Button disabled={saving} onClick={closeAndReturnFocus}>
              {t("common:cancel", "Cancel")}
            </Button>
            <Button
              type="primary"
              aria-label={t("common:apply", "Apply")}
              loading={saving}
              disabled={loading || Boolean(sceneLoadError) || !loadedSceneRef.current?.()}
              onClick={handleApply}>
              {t("common:apply", "Apply")}
            </Button>
          </div>
        </div>
      }
      title={t("playground:composer.rolePlaySetup", "Role-play setup")}>
      {modalContextHolder}
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
                onSelectionChange={(selection) => stage(() => {
                  setStagedIdentity(selection)
                  setClearIdentity(false)
                })}
                showLabel
                className="inline-flex min-h-9 w-full items-center justify-start gap-2 rounded-md border border-border bg-surface2 px-3 py-2 text-text"
                iconClassName="h-4 w-4"
              />
            </div>
            <div className="space-y-1">
              <div className="text-xs font-medium text-text">
                {t("playground:composer.context.behavior", "Behavior")}
              </div>
              <Button block onClick={() => { if (isDraftCurrent()) setTemplatesOpen(true) }}>
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
              onClick={() => stage(() => setClearIdentity(true))}>
              {t("playground:composer.clearIdentity", "Clear identity")}
            </Button>
            <Button
              disabled={!beforeState.behavior}
              onClick={() => stage(() => setClearBehavior(true))}>
              {t("playground:composer.clearBehavior", "Clear behavior")}
            </Button>
            <Button
              disabled={!beforeState.generationStyle}
              onClick={() => stage(() => setResetGenerationStyle(true))}>
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
                    ? t("playground:composer.sceneDraftEnabled", "Scene draft enabled")
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
            <Button onClick={() => stage(() => setSceneDraft(clearRolePlayScene(sceneDraft)))}>
              {t("playground:composer.clearScene", "Clear scene")}
            </Button>
            <Button onClick={() => stage(() => setSceneDraft(resetRolePlayScene()))}>
              {t("playground:composer.resetScene", "Reset scene")}
            </Button>
          </div>
        </section>

      </fieldset>
      <SystemPromptTemplatesModal
        open={templatesOpen}
        onClose={() => { if (isDraftCurrent()) setTemplatesOpen(false) }}
        onSelect={handleBehaviorTemplateSelect}
      />
    </Drawer>
  )
}
