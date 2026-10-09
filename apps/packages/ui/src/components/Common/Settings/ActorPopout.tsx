import React from "react"
import { Button, Drawer, Form, Segmented, Skeleton, Switch } from "antd"
import { useTranslation } from "react-i18next"
import { useStoreMessageOption } from "@/store/option"
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
    useStoreMessageOption(
      (state) => ({
        historyId: state.historyId,
        serverChatId: state.serverChatId,
        serverChatAssistantKind: state.serverChatAssistantKind
      }),
      shallow
    )
  const [selectedCharacter] = useSelectedCharacter<Character | null>(null)
  const [selectedAssistant] = useSelectedAssistant(null)
  const [form] = Form.useForm()
  const {
    settings,
    setSettings,
    preview,
    tokenCount,
    setPreviewAndTokens
  } = useActorStore(
    (state) => ({
      settings: state.settings,
      setSettings: state.setSettings,
      preview: state.preview,
      tokenCount: state.tokenCount,
      setPreviewAndTokens: state.setPreviewAndTokens
    }),
    shallow
  )
  const { editorMode, setEditorMode } = useActorEditorPrefs()
  const [loading, setLoading] = React.useState(false)
  const [newAspectTarget, setNewAspectTarget] =
    React.useState<ActorTarget>("user")
  const [newAspectName, setNewAspectName] = React.useState("")
  const actorPositionValue = Form.useWatch("actorChatPosition", form)
  const hydratedRef = React.useRef(false)
  const timeoutRef = React.useRef<number | undefined>()
  const personaChatActive =
    serverChatAssistantKind === "persona" || selectedAssistant?.kind === "persona"

  React.useEffect(() => {
    if (!open || settings) return
    if (import.meta?.env?.DEV) {
      console.count("ActorPopout/initSettings")
    }
    const base = createDefaultActorSettings()
    setSettings(base)
    const baseFields: Record<string, any> = {
      actorEnabled: base.isEnabled,
      actorNotes: base.notes,
      actorNotesGmOnly: base.notesGmOnly ?? false,
      actorChatPosition: base.chatPosition,
      actorChatDepth: base.chatDepth,
      actorChatRole: base.chatRole,
      actorTemplateMode: base.templateMode ?? "merge",
      actorAppendable: base.appendable ?? false
    }
    for (const aspect of base.aspects || []) {
      baseFields[`actor_${aspect.id}`] = aspect.value
      baseFields[`actor_key_${aspect.id}`] = aspect.key
    }
    form.setFieldsValue(baseFields)
    const text = buildActorPrompt(base)
    setPreviewAndTokens(text, estimateActorTokens(text))
  }, [form, open, setPreviewAndTokens, setSettings, settings])

  const hydrate = React.useCallback(async () => {
    if (!open || personaChatActive) return
    setLoading(true)
    try {
      const { getActorSettingsForChatWithCharacterFallback } =
        await loadActorSettings()
      const actor = await getActorSettingsForChatWithCharacterFallback({
        historyId,
        serverChatId,
        characterId: selectedCharacter?.id ?? null
      })
      setSettings(actor)

      const baseFields: Record<string, any> = {
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
        baseFields[`actor_${aspect.id}`] = aspect.value
        baseFields[`actor_key_${aspect.id}`] = aspect.key
      }
      form.setFieldsValue(baseFields)

      const text = buildActorPrompt(actor)
      setPreviewAndTokens(text, estimateActorTokens(text))
    } finally {
      setLoading(false)
    }
  }, [
    form,
    historyId,
    personaChatActive,
    selectedCharacter?.id,
    serverChatId,
    setPreviewAndTokens,
    setSettings
  ])

  React.useEffect(() => {
    if (personaChatActive) {
      hydratedRef.current = false
      return
    }
    if (open && !hydratedRef.current) {
      if (import.meta?.env?.DEV) {
        console.count("ActorPopout/hydrate")
      }
      hydratedRef.current = true
      void hydrate()
    }
    if (!open) {
      hydratedRef.current = false
    }
  }, [open, hydrate, personaChatActive])

  React.useEffect(() => {
    return () => {
      if (timeoutRef.current !== undefined) {
        window.clearTimeout(timeoutRef.current)
      }
    }
  }, [])

  const recompute = React.useCallback(() => {
    const base = settings ?? createDefaultActorSettings()
    const values = form.getFieldsValue()

    const next: ActorSettings = buildActorSettingsFromForm(base, values)

    const text = buildActorPrompt(next)
    setPreviewAndTokens(text, estimateActorTokens(text))
  }, [form, setPreviewAndTokens, settings])

  const debouncedRecompute = React.useMemo(() => {
    return () => {
      if (timeoutRef.current !== undefined) {
        window.clearTimeout(timeoutRef.current)
      }
      timeoutRef.current = window.setTimeout(() => {
        recompute()
      }, 150)
    }
  }, [recompute])

  const handleSave = async (values: any) => {
    const base = settings ?? createDefaultActorSettings()
    const next: ActorSettings = buildActorSettingsFromForm(base, values)
    setSettings(next)
    const { saveActorSettingsForChat } = await loadActorSettings()
    await saveActorSettingsForChat({
      historyId,
      serverChatId,
      settings: next
    })
    setOpen(false)
  }

  return (
    <Drawer
      placement="right"
      size={420}
      open={open}
      onClose={() => setOpen(false)}
      title={t("playground:composer.actorTitle", "Scene Director (Actor)")}>
      {personaChatActive ? (
        <div className="rounded-lg border border-border bg-surface2 p-4 text-sm text-text-muted">
          {t(
            "playground:composer.actorPersonaUnsupported",
            "Scene Director is currently available only for character-backed chats."
          )}
        </div>
      ) : null}
      {!personaChatActive && loading && !settings ? (
        <Skeleton active />
      ) : !personaChatActive ? (
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSave}
          onValuesChange={(changed) => {
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
          <div className="space-y-3">
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

            <div className="pt-2 flex justify-end gap-2">
              <Button onClick={() => setOpen(false)}>
                {t("common:cancel", "Cancel")}
              </Button>
              <Button type="primary" htmlType="submit">
                {t("common:save", "Save")}
              </Button>
            </div>
          </div>
        </Form>
      ) : null}
    </Drawer>
  )
}
