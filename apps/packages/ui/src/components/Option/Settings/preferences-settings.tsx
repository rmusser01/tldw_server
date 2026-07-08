import { useRef } from "react"
import { Modal, Radio, Select, Switch } from "antd"
import { useTranslation } from "react-i18next"
import { useStorage } from "@plasmohq/storage/hook"

import { useAntdNotification } from "@/hooks/useAntdNotification"
import { useConnectionActions, useConnectionState } from "@/hooks/useConnectionState"
import { useI18n } from "@/hooks/useI18n"
import { useSetting } from "@/hooks/useSetting"
import {
  getDefaultShortcutsForPersona
} from "@/components/Layouts/header-shortcut-items"
import {
  HEADER_SHORTCUT_SELECTION_SETTING
} from "@/services/settings/ui-settings"
import { useTutorialCompletion } from "@/store/tutorials"
import type { UserPersona } from "@/types/connection"
import { SearchModeSettings } from "./search-mode"

export const PreferencesSettings = () => {
  const { t } = useTranslation("settings")
  const notification = useAntdNotification()
  const { changeLocale, locale, supportLanguage } = useI18n()
  const { userPersona } = useConnectionState()
  const { setUserPersona } = useConnectionActions()
  const { completedTutorials, resetProgress: resetTutorialProgress } =
    useTutorialCompletion()
  const [, setShortcutSelection] = useSetting(HEADER_SHORTCUT_SELECTION_SETTING)
  const personaSeqRef = useRef(0)

  const [onboardingAutoFinish, setOnboardingAutoFinish] = useStorage(
    "onboardingAutoFinish",
    false
  )
  const [sendNotificationAfterIndexing, setSendNotificationAfterIndexing] =
    useStorage("sendNotificationAfterIndexing", false)
  const [checkOllamaStatus, setCheckOllamaStatus] = useStorage(
    "checkOllamaStatus",
    true
  )

  const handlePersonaChange = async (nextPersona: UserPersona) => {
    const seq = ++personaSeqRef.current
    await setUserPersona(nextPersona)
    if (seq !== personaSeqRef.current) return
    await setShortcutSelection(getDefaultShortcutsForPersona(nextPersona))
  }

  return (
    <dl className="flex flex-col space-y-6 text-sm">
      <div>
        <h2 className="text-base font-semibold leading-7 text-text">
          {t("preferencesSettings.title", "General preferences")}
        </h2>
        <div className="border-b border-border mt-3" />
      </div>

      <div className="flex flex-row justify-between">
        <span className="text-text">
          {t("generalSettings.settings.language.label")}
        </span>

        <Select
          aria-label={t("generalSettings.settings.language.label")}
          allowClear
          filterOption={(input, option) =>
            option!.label.toLowerCase().indexOf(input.toLowerCase()) >= 0 ||
            option!.value.toLowerCase().indexOf(input.toLowerCase()) >= 0
          }
          onChange={(value) => {
            changeLocale(value)
          }}
          options={supportLanguage}
          placeholder={t("generalSettings.settings.language.placeholder")}
          showSearch
          style={{ width: "200px" }}
          value={locale}
        />
      </div>

      <div className="flex flex-row justify-between">
        <div>
          <span className="text-text">
            {t("generalSettings.settings.sendNotificationAfterIndexing.label")}
          </span>
        </div>

        <Switch
          aria-label={t("generalSettings.settings.sendNotificationAfterIndexing.label")}
          checked={sendNotificationAfterIndexing}
          onChange={setSendNotificationAfterIndexing}
        />
      </div>

      <div className="flex flex-row justify-between">
        <div>
          <span className="text-text">
            {t("generalSettings.settings.ollamaStatus.label")}
          </span>
        </div>

        <Switch
          aria-label={t("generalSettings.settings.ollamaStatus.label")}
          checked={checkOllamaStatus}
          onChange={setCheckOllamaStatus}
        />
      </div>

      <div className="flex flex-row justify-between">
        <div>
          <span className="text-text">
            {t(
              "generalSettings.settings.onboardingAutoFinish.label",
              "Auto-finish onboarding after successful connection"
            )}
          </span>
        </div>

        <Switch
          aria-label={t(
            "generalSettings.settings.onboardingAutoFinish.label",
            "Auto-finish onboarding after successful connection"
          )}
          checked={onboardingAutoFinish}
          onChange={setOnboardingAutoFinish}
        />
      </div>

      <div className="flex flex-row justify-between">
        <div className="inline-flex items-center gap-2">
          <span className="text-text">
            {t(
              "generalSettings.settings.resetTutorials.label",
              "Reset tutorial progress"
            )}
          </span>
          {completedTutorials.length > 0 && (
            <span className="text-xs text-text-muted">
              ({completedTutorials.length}{" "}
              {t(
                "generalSettings.settings.resetTutorials.completed",
                "completed"
              )}
              )
            </span>
          )}
        </div>

        <button
          className="text-xs text-primary hover:text-primaryStrong disabled:cursor-not-allowed disabled:opacity-50"
          disabled={completedTutorials.length === 0}
          onClick={() => {
            Modal.confirm({
              content: t(
                "generalSettings.settings.resetTutorials.confirmMessage",
                "This will mark all tutorials as incomplete so you can replay them."
              ),
              onOk: () => {
                resetTutorialProgress()
                notification.success({
                  message: t(
                    "generalSettings.settings.resetTutorials.toast",
                    "Tutorial progress has been reset"
                  )
                })
              },
              title: t(
                "generalSettings.settings.resetTutorials.confirmTitle",
                "Reset tutorial progress?"
              )
            })
          }}
          type="button"
        >
          {t(
            "generalSettings.settings.resetTutorials.button",
            "Reset tutorials"
          )}
        </button>
      </div>

      <div>
        <h2 className="text-base font-semibold leading-7 text-text">
          {t("generalSettings.persona.title", "Persona")}
        </h2>
        <p className="mt-1 mb-3 text-xs text-text-muted">
          {t(
            "generalSettings.persona.description",
            "Your persona controls which features are shown in the navigation. Change it at any time."
          )}
        </p>
        <div className="border-b border-border mb-3" />
        <Radio.Group
          className="flex flex-col gap-2"
          onChange={(event) => {
            const value = event.target.value as string
            handlePersonaChange(
              value === "explorer" ? null : (value as UserPersona)
            ).catch((error) => {
              console.error("[PreferencesSettings] Failed to change persona", error)
              notification.error({
                message: t(
                  "generalSettings.persona.changeError",
                  "Could not update persona"
                )
              })
            })
          }}
          value={userPersona ?? "explorer"}
        >
          <Radio value="researcher">
            {t("generalSettings.persona.researcher", "Researcher")}
            <span className="ml-2 text-xs text-text-muted">
              {t(
                "generalSettings.persona.researcherHint",
                "Focus on research, knowledge, and evaluation tools"
              )}
            </span>
          </Radio>
          <Radio value="family">
            {t("generalSettings.persona.family", "Family")}
            <span className="ml-2 text-xs text-text-muted">
              {t(
                "generalSettings.persona.familyHint",
                "Simplified view with safety and content controls"
              )}
            </span>
          </Radio>
          <Radio value="explorer">
            {t("generalSettings.persona.explorer", "Explorer / All features")}
            <span className="ml-2 text-xs text-text-muted">
              {t(
                "generalSettings.persona.explorerHint",
                "Show every feature in the navigation"
              )}
            </span>
          </Radio>
        </Radio.Group>
      </div>

      <SearchModeSettings />
    </dl>
  )
}

export default PreferencesSettings
