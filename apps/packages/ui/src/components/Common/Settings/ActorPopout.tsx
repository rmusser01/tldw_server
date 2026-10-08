import React from "react"
import { Button, Drawer, Form, Modal, Segmented, Skeleton, Switch } from "antd"
import { useTranslation } from "react-i18next"
import { useMessageOption } from "@/hooks/useMessageOption"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { captureChatDraftSource } from "@/hooks/chat/chat-draft-transition"
import { useHistorySelectionContext } from "@/hooks/chat/useHistorySelection"
import type { ActorEditorMode, ActorSettings, ActorTarget } from "@/types/actor"
import { createDefaultActorSettings } from "@/types/actor"
import {
  buildActorPrompt,
  buildActorSettingsFromForm,
  estimateActorTokens
} from "@/utils/actor"
import { ActorEditor } from "@/components/Common/Settings/ActorEditor"
import { useActorEditorPrefs, useActorStore } from "@/store/actor"
import { shallow } from "zustand/shallow"
import type { Character } from "@/types/character"
import { useSelectedAssistant } from "@/hooks/useSelectedAssistant"
import { useSelectedCharacter } from "@/hooks/useSelectedCharacter"

type Props = {
  open: boolean
  setOpen: (open: boolean) => void
}

const loadActorSettings = () => import("@/services/actor-settings")

export const ActorPopout: React.FC<Props> = ({ open, setOpen }) => {
  const { t } = useTranslation(["playground", "common"])
  const { historyId, serverChatId, serverChatAssistantKind } =
    useMessageOption()
  const [selectedCharacter] = useSelectedCharacter<Character | null>(null)
  const [selectedAssistant, , assistantMeta] = useSelectedAssistant(null)
  const accountIsCurrent = assistantMeta?.isCurrent
  const historySelection = useHistorySelectionContext()
  const selectionSignal = historySelection?.getSignal()
  const captureSelectionFence = historySelection?.fence
  const [form] = Form.useForm()
  const [modal, modalContextHolder] = Modal.useModal()
  const editRevisionRef = React.useRef(0)
  const baselineRef = React.useRef<{ settings: ActorSettings; fields: Record<string, unknown> } | null>(null)
  const [settings, setSettingsState] = React.useState<ActorSettings | null>(null)
  const [{ preview, tokenCount }, setPreviewState] = React.useState({ preview: "", tokenCount: 0 })
  const { commitSettings, commitPreview } =
    useActorStore(
      (state) => ({
        commitSettings: state.setSettings,
        commitPreview: state.setPreviewAndTokens
      }),
      shallow
    )
  const { editorMode, setEditorMode } = useActorEditorPrefs()
  const [loading, setLoading] = React.useState(false)
  const [newAspectTarget, setNewAspectTarget] =
    React.useState<ActorTarget>("user")
  const [newAspectName, setNewAspectName] = React.useState("")
  const actorPositionValue = Form.useWatch("actorChatPosition", form)
  const timeoutRef = React.useRef<number | undefined>()
  const personaChatActive =
    serverChatAssistantKind === "persona" ||
    selectedAssistant?.kind === "persona"

  const draftRevision = useStoreMessageOption(
    (state) => state.chatDraftRevision
  )
  const restoreRevision = usePlaygroundSessionStore(
    (state) => state.restoreRevision
  )
  const selectionRevision = usePlaygroundSessionStore(
    (state) => state.sourceSelectionRevision
  )
  const [readRevision, setReadRevision] = React.useState(0)
  const [loadError, setLoadError] = React.useState(false)
  const [saveError, setSaveError] = React.useState(false)
  const [saving, setSaving] = React.useState(false)
  const saveAttemptRef = React.useRef<object | null>(null)
  const readSourceRef = React.useRef<(() => boolean) | null>(null)
  const loadedSourceRef = React.useRef<(() => boolean) | null>(null)
  const view = {
    open,
    historyId,
    serverChatId,
    characterId: selectedCharacter?.id,
    personaChatActive,
    draftRevision,
    restoreRevision,
    selectionRevision,
    readRevision,
    selectionSignal,
    captureSelectionFence,
    accountIsCurrent
  }
  const viewRef = React.useRef(view)
  if (
    Object.keys(view).some(
      (key) =>
        view[key as keyof typeof view] !==
        viewRef.current[key as keyof typeof view]
    )
  )
    viewRef.current = view
  const renderedView = viewRef.current
  const loadedSource = loadedSourceRef.current
  const isEditorCurrent = React.useCallback(() => !saveAttemptRef.current && viewRef.current === renderedView && Boolean(loadedSource?.()), [loadedSource, renderedView])
  const captureEditGuard = React.useCallback(() => {
    const revision = editRevisionRef.current
    return () => isEditorCurrent() && editRevisionRef.current === revision
  }, [isEditorCurrent])
  const setSettings = React.useCallback((next: ActorSettings) => {
    if (!isEditorCurrent()) return
    editRevisionRef.current++
    setSettingsState(next)
  }, [isEditorCurrent])
  const setPreviewAndTokens = React.useCallback((preview: string, tokenCount: number) => {
    if (isEditorCurrent()) setPreviewState({ preview, tokenCount })
  }, [isEditorCurrent])
  const requestClose = () => {
    if (saveAttemptRef.current || saving || viewRef.current !== renderedView) return
    const baseline = baselineRef.current
    const dirty = baseline && (JSON.stringify(settings) !== JSON.stringify(baseline.settings) ||
      Object.entries(baseline.fields).some(([key, value]) => JSON.stringify(form.getFieldValue(key)) !== JSON.stringify(value)))
    if (!dirty) return setOpen(false)
    const canDiscard = captureEditGuard()
    if (!canDiscard()) return
    modal.confirm({
      title: t("common:modelSettings.discardSceneTitle", "Discard unsaved scene settings?"),
      content: t("common:modelSettings.discardSceneHelp", "Your staged scene edits will be discarded."),
      okText: t("common:discard", "Discard"),
      cancelText: t("common:keepEditing", "Keep editing"),
      onOk: () => { if (canDiscard()) setOpen(false) }
    })
  }

  React.useEffect(() => {
    loadedSourceRef.current = null
    baselineRef.current = null
    saveAttemptRef.current = null
    setSaving(false)
    setSaveError(false)
    if (!open || personaChatActive) return
    let active = true
    const sourceView = viewRef.current
    const selectionIsCurrent = captureSelectionFence?.() ?? (() => true)
    const { isCurrent } = captureChatDraftSource(
      () =>
        active &&
        selectionIsCurrent() &&
        (accountIsCurrent?.() ?? true) &&
        viewRef.current === sourceView &&
        usePlaygroundSessionStore.getState().restoreRevision ===
          restoreRevision &&
        usePlaygroundSessionStore.getState().sourceSelectionRevision ===
          selectionRevision
    )
    readSourceRef.current = isCurrent
    if (!isCurrent()) {
      setLoading(false)
      setLoadError(true)
      return () => {
        active = false
      }
    }
    setLoading(true)
    setLoadError(false)
    const hydrate = async () => {
      try {
        const { getActorSettingsForChatWithCharacterFallback } =
          await loadActorSettings()
        if (!isCurrent()) return
        const actor = await getActorSettingsForChatWithCharacterFallback({
          historyId,
          serverChatId,
          characterId: selectedCharacter?.id ?? null
        })
        if (!isCurrent()) return
        loadedSourceRef.current = isCurrent
        setSettingsState(actor)
        commitSettings(actor)
        if (!isCurrent()) return
        const fields: Record<string, unknown> = {
          actorEnabled: actor.isEnabled,
          actorNotes: actor.notes,
          actorNotesGmOnly: actor.notesGmOnly ?? false,
          actorChatPosition: actor.chatPosition,
          actorChatDepth: actor.chatDepth,
          actorChatRole: actor.chatRole,
          actorTemplateMode: actor.templateMode ?? "merge",
          actorAppendable: actor.appendable ?? false
        }
        for (const aspect of actor.aspects || []) {
          fields[`actor_${aspect.id}`] = aspect.value
          fields[`actor_key_${aspect.id}`] = aspect.key
        }
        form.resetFields()
        if (!isCurrent()) return
        form.setFieldsValue(fields)
        if (!isCurrent()) return
        baselineRef.current = { settings: structuredClone(actor), fields: structuredClone(form.getFieldsValue(true)) }
        const text = buildActorPrompt(actor)
        setPreviewState({ preview: text, tokenCount: estimateActorTokens(text) })
        commitPreview(text, estimateActorTokens(text))
        if (!isCurrent()) return
      } catch {
        if (isCurrent()) setLoadError(true)
      } finally {
        if (isCurrent()) setLoading(false)
      }
    }
    void hydrate()
    return () => {
      active = false
    }
  }, [
    form,
    open,
    personaChatActive,
    historyId,
    serverChatId,
    selectedCharacter?.id,
    draftRevision,
    restoreRevision,
    selectionRevision,
    readRevision,
    selectionSignal,
    captureSelectionFence,
    accountIsCurrent,
    commitPreview,
    commitSettings
  ])

  React.useEffect(() => {
    return () => {
      if (timeoutRef.current !== undefined) {
        window.clearTimeout(timeoutRef.current)
      }
    }
  }, [renderedView, settings])

  const recompute = React.useCallback(() => {
    if (!isEditorCurrent()) return
    const base = settings ?? createDefaultActorSettings()
    const values = form.getFieldsValue()

    const next: ActorSettings = buildActorSettingsFromForm(base, values)

    const text = buildActorPrompt(next)
    setPreviewAndTokens(text, estimateActorTokens(text))
  }, [form, isEditorCurrent, setPreviewAndTokens, settings])

  const debouncedRecompute = React.useMemo(() => {
    return () => {
      if (!isEditorCurrent()) return
      const revision = editRevisionRef.current
      if (timeoutRef.current !== undefined) window.clearTimeout(timeoutRef.current)
      const timer = window.setTimeout(() => {
        if (timeoutRef.current !== timer || editRevisionRef.current !== revision || !isEditorCurrent()) return
        timeoutRef.current = undefined
        recompute()
      }, 150)
      timeoutRef.current = timer
    }
  }, [isEditorCurrent, recompute])
  React.useEffect(() => { recompute() }, [recompute])

  const handleSave = async (values: any) => {
    const isCurrent = loadedSourceRef.current
    if (
      viewRef.current !== renderedView ||
      !isCurrent?.() ||
      saveAttemptRef.current
    )
      return
    const attempt = {}
    const revision = editRevisionRef.current
    const canCommit = () => saveAttemptRef.current === attempt &&
      viewRef.current === renderedView && isCurrent() && editRevisionRef.current === revision
    saveAttemptRef.current = attempt
    setSaving(true)
    setSaveError(false)
    const next = buildActorSettingsFromForm(
      settings ?? createDefaultActorSettings(),
      { ...form.getFieldsValue(true), ...values }
    )
    try {
      const { saveActorSettingsForChat } = await loadActorSettings()
      if (!canCommit()) return
      const saved = await saveActorSettingsForChat({
        historyId,
        serverChatId,
        settings: next
      })
      if (!canCommit()) return
      if (!saved) throw new Error("actor_settings_save_failed")
      commitSettings(next)
      if (!canCommit()) return
      const text = buildActorPrompt(next)
      commitPreview(text, estimateActorTokens(text))
      if (!canCommit()) return
      setOpen(false)
    } catch {
      if (canCommit()) setSaveError(true)
    } finally {
      if (saveAttemptRef.current === attempt) {
        saveAttemptRef.current = null
        if (isCurrent()) setSaving(false)
      }
    }
  }

  return (
    <Drawer
      placement="right"
      size={420}
      open={open}
      onClose={requestClose}
      title={t("playground:composer.actorTitle", "Scene Director (Actor)")}>
      {modalContextHolder}
      {personaChatActive ? (
        <div className="rounded-lg border border-border bg-surface2 p-4 text-sm text-text-muted">
          {t(
            "playground:composer.actorPersonaUnsupported",
            "Scene Director is currently available only for character-backed chats."
          )}
        </div>
      ) : null}
      {!personaChatActive && loadError ? (
        <div className="space-y-2">
          <p role="alert">
            {t(
              "playground:composer.sceneLoadError",
              "Scene settings could not be loaded. Existing chat setup was left unchanged."
            )}
          </p>
          {readSourceRef.current?.() ? (
            <Button
              onClick={() => {
                if (
                  viewRef.current === renderedView &&
                  readSourceRef.current?.()
                )
                  setReadRevision((value) => value + 1)
              }}>
              {t("common:retry", "Retry")}
            </Button>
          ) : (
            <p>
              {t(
                "playground:composer.sceneSourceUnavailable",
                "Close this panel and check your account verification and selected chat before reopening scene settings."
              )}
            </p>
          )}
        </div>
      ) : null}
      {saveError ? (
        <p role="alert">
          {t(
            "playground:composer.sceneSaveError",
            "Scene settings could not be saved. Your edits are still here; retry Save."
          )}
        </p>
      ) : null}
      {!personaChatActive && loading ? (
        <Skeleton active />
      ) : !personaChatActive && !loadError ? (
        <Form
          form={form}
          layout="vertical"
          disabled={saving || !loadedSourceRef.current?.()}
          onFinish={handleSave}
          onValuesChange={(changed) => {
            editRevisionRef.current++
            const keys = Object.keys(changed || {})
            const shouldUpdate = keys.some(
              (k) =>
                k === "actorEnabled" ||
                k === "actorNotes" ||
                k === "actorNotesGmOnly" ||
                k === "actorChatPosition" ||
                k === "actorChatDepth" ||
                k === "actorChatRole" ||
                k === "actorAppendable" ||
                k.startsWith("actor_")
            )
            if (shouldUpdate) {
              debouncedRecompute()
            }
          }}>
          <fieldset
            className="space-y-3 min-w-0 border-0 p-0"
            disabled={saving || !loadedSourceRef.current?.()}>
            <div className="flex items-center justify-between">
              <div className="flex flex-col">
                <span className="font-medium text-text">
                  {t(
                    "playground:composer.actorTitle",
                    "Scene Director (Actor)"
                  )}
                </span>
                <span className="text-xs text-text-subtle">
                  {t(
                    "playground:composer.actorHelp",
                    "Configure per-chat scene context: roles, mood, world, goals, and notes."
                  )}
                </span>
              </div>
              <Form.Item
                name="actorEnabled"
                valuePropName="checked"
                className="mb-0">
                <Switch />
              </Form.Item>
            </div>

            <div className="flex items-center justify-between pt-2">
              <Segmented
                size="small"
                value={editorMode}
                onChange={(val) => setEditorMode(val as ActorEditorMode)}
                options={[
                  {
                    value: "simple",
                    label: t("playground:actor.modeSimple", "Simple")
                  },
                  {
                    value: "advanced",
                    label: t("playground:actor.modeAdvanced", "Advanced")
                  }
                ]}
              />
            </div>

            <ActorEditor
              form={form}
              settings={settings}
              setSettings={setSettings}
              isCurrent={isEditorCurrent}
              captureEditGuard={captureEditGuard}
              actorPreview={preview}
              actorTokenCount={tokenCount}
              onRecompute={recompute}
              newAspectTarget={newAspectTarget}
              setNewAspectTarget={setNewAspectTarget}
              newAspectName={newAspectName}
              setNewAspectName={setNewAspectName}
              actorPositionValue={actorPositionValue}
              editorMode={editorMode}
              onModeChange={setEditorMode}
            />
          </fieldset>
          <div className="pt-2 flex justify-end gap-2">
            <Button onClick={() => setOpen(false)}>
              {t("common:cancel", "Cancel")}
            </Button>
            <Button
              type="primary"
              htmlType="submit"
              loading={saving}
              disabled={loading || loadError || !loadedSourceRef.current?.()}>
              {t("common:save", "Save")}
            </Button>
          </div>
        </Form>
      ) : null}
    </Drawer>
  )
}
