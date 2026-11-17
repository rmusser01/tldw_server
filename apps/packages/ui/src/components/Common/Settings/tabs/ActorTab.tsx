import { Form, Segmented, Switch } from "antd"
import type { FormInstance } from "antd"
import { useTranslation } from "react-i18next"
import { ActorEditor } from "@/components/Common/Settings/ActorEditor"
import { useActorEditorPrefs } from "@/store/actor"
import type { ActorEditorMode, ActorSettings, ActorTarget } from "@/types/actor"

interface ActorTabProps {
  form: FormInstance
  actorSettings: ActorSettings | null
  setActorSettings: (settings: ActorSettings) => void
  isCurrent: () => boolean
  captureEditGuard: () => () => boolean
  actorPreview: string
  actorTokenCount: number
  onRecompute: () => void
  newAspectTarget: ActorTarget
  setNewAspectTarget: (target: ActorTarget) => void
  newAspectName: string
  setNewAspectName: (name: string) => void
  actorPositionValue: string | undefined
}

export function ActorTab({
  form,
  actorSettings,
  setActorSettings,
  isCurrent,
  captureEditGuard,
  actorPreview,
  actorTokenCount,
  onRecompute,
  newAspectTarget,
  setNewAspectTarget,
  newAspectName,
  setNewAspectName,
  actorPositionValue
}: ActorTabProps) {
  const { t } = useTranslation("playground")
  const { editorMode, setEditorMode } = useActorEditorPrefs()

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex flex-col">
          <span className="font-medium text-text">
            {t("composer.actorTitle", "Scene Director (Actor)")}
          </span>
          <span className="text-xs text-text-subtle">
            {t(
              "composer.actorHelp",
              "Configure per-chat scene context: roles, mood, world, goals, and notes."
            )}
          </span>
        </div>
        <Form.Item name="actorEnabled" valuePropName="checked" className="mb-0">
          <Switch />
        </Form.Item>
      </div>

      <div className="flex items-center justify-between">
        <Segmented
          size="small"
          value={editorMode}
          onChange={(val) => setEditorMode(val as ActorEditorMode)}
          options={[
            {
              value: "simple",
              label: t("actor.modeSimple", "Simple")
            },
            {
              value: "advanced",
              label: t("actor.modeAdvanced", "Advanced")
            }
          ]}
        />
      </div>

      {actorSettings && (
        <ActorEditor
          form={form}
          settings={actorSettings}
          setSettings={setActorSettings}
          isCurrent={isCurrent}
          captureEditGuard={captureEditGuard}
          actorPreview={actorPreview}
          actorTokenCount={actorTokenCount}
          onRecompute={onRecompute}
          newAspectTarget={newAspectTarget}
          setNewAspectTarget={setNewAspectTarget}
          newAspectName={newAspectName}
          setNewAspectName={setNewAspectName}
          actorPositionValue={actorPositionValue}
          editorMode={editorMode}
          onModeChange={setEditorMode}
        />
      )}
    </div>
  )
}
