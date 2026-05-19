import { useRef } from "react"
import { Alert, Modal, Radio, Select, Switch } from "antd"
import { useNavigate } from "react-router-dom"
import { SearchModeSettings } from "./search-mode"
import { useTranslation } from "react-i18next"
import { useI18n } from "@/hooks/useI18n"
import { useStorage } from "@plasmohq/storage/hook"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { SystemSettings } from "./system-settings"
import { ThemePicker } from "@/components/Common/Settings/ThemePicker"
import { getDefaultOcrLanguage, ocrLanguages } from "@/data/ocr-language"
import { useServerOnline } from "@/hooks/useServerOnline"
import { useConnectionState, useConnectionActions } from "@/hooks/useConnectionState"
import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import ConnectFeatureBanner from "@/components/Common/ConnectFeatureBanner"
import { useTutorialCompletion } from "@/store/tutorials"
import { isExtensionRuntime } from "@/utils/browser-runtime"
import { useSetting } from "@/hooks/useSetting"
import { HEADER_SHORTCUT_SELECTION_SETTING } from "@/services/settings/ui-settings"
import { getDefaultShortcutsForPersona } from "@/components/Layouts/header-shortcut-items"
import type { UserPersona } from "@/types/connection"

export const GeneralSettings = () => {
  // Persisted preference: auto-finish onboarding when connection & RAG are healthy
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

  const [defaultOCRLanguage, setDefaultOCRLanguage] = useStorage(
    "defaultOCRLanguage",
    getDefaultOcrLanguage()
  )
  const [enableOcrAssets, setEnableOcrAssets] = useStorage(
    "enableOcrAssets",
    false
  )

  const [settingsIntroDismissed, setSettingsIntroDismissed] = useStorage(
    "settingsIntroDismissed",
    false
  )

  const { t } = useTranslation("settings")
  const notification = useAntdNotification()
  const { changeLocale, locale, supportLanguage } = useI18n()
  const isOnline = useServerOnline()
  const navigate = useNavigate()
  const { serverUrl: connectedServerUrl, userPersona } = useConnectionState()
  const { restartOnboarding, setUserPersona } = useConnectionActions()
  const { completedTutorials, resetProgress: resetTutorialProgress } = useTutorialCompletion()
  const [, setShortcutSelection] = useSetting(HEADER_SHORTCUT_SELECTION_SETTING)
  const personaSeqRef = useRef(0)

  const handlePersonaChange = async (nextPersona: UserPersona) => {
    const seq = ++personaSeqRef.current
    await setUserPersona(nextPersona)
    if (seq !== personaSeqRef.current) return // stale
    const shortcuts = getDefaultShortcutsForPersona(nextPersona)
    await setShortcutSelection(shortcuts)
  }

  return (
    <dl className="flex flex-col space-y-6 text-sm">
      {!isOnline && (
        <div>
          <ConnectFeatureBanner
            title={t("generalSettings.empty.connectTitle", {
              defaultValue: "Connect tldw Assistant to your server"
            })}
            description={t("generalSettings.empty.connectDescription", {
              defaultValue:
                "Some settings only take effect when your tldw server is reachable. Connect your server to get the full experience."
            })}
            examples={[
              t("generalSettings.empty.connectExample1", {
                defaultValue:
                  "Open Settings → tldw server to add your server URL and API key."
              }),
              t("generalSettings.empty.connectExample2", {
                defaultValue:
                  "Use Diagnostics to confirm your server is healthy before trying advanced tools."
              })
            ]}
          />
        </div>
      )}

      {isOnline && !settingsIntroDismissed && (
        <div>
          <FeatureEmptyState
            title={t("generalSettings.empty.title", {
              defaultValue: "Tune how tldw Assistant behaves"
            })}
            description={t("generalSettings.empty.description", {
              defaultValue:
                "Adjust defaults for the Web UI, sidepanel, speech, search, and data handling from one place."
            })}
            examples={[
              t("generalSettings.empty.example1", {
                defaultValue:
                  "Choose your default language, theme, and chat resume behavior."
              }),
              t("generalSettings.empty.example2", {
                defaultValue:
                  "Control whether chats are temporary, how large pastes are handled, and how reasoning is displayed."
              })
            ]}
            primaryActionLabel={t("generalSettings.empty.primaryCta", {
              defaultValue: "Configure server & auth"
            })}
            onPrimaryAction={() => navigate("/settings/tldw")}
            secondaryActionLabel={t("generalSettings.empty.secondaryCta", {
              defaultValue: "Dismiss"
            })}
            onSecondaryAction={() => setSettingsIntroDismissed(true)}
          />
        </div>
      )}
      {/* Connection info (A1: read-only server URL) */}
      <div>
        <h2 className="text-base font-semibold leading-7 text-text">
          {t("generalSettings.connection.title", "Connection")}
        </h2>
        <div className="border-b border-border mt-3 mb-3"></div>
        <div className="flex flex-row items-center justify-between">
          <span className="text-text">
            {t("generalSettings.connection.serverUrl", "Server URL")}
          </span>
          <div className="flex items-center gap-2">
            <code className="rounded border border-border bg-surface2 px-2 py-0.5 text-xs text-text-muted select-all">
              {connectedServerUrl || t("generalSettings.connection.notConfigured", "Not configured")}
            </code>
            {isOnline ? (
              <span className="inline-flex h-2 w-2 rounded-full bg-success" title={t("generalSettings.connection.online", "Online")} />
            ) : (
              <span className="inline-flex h-2 w-2 rounded-full bg-danger" title={t("generalSettings.connection.offline", "Offline")} />
            )}
          </div>
        </div>
        <p className="mt-1 text-[11px] text-text-subtle">
          {isExtensionRuntime()
            ? t(
                "generalSettings.connection.changeHintExtension",
                "To change, update the server URL in extension settings."
              )
            : t(
                "generalSettings.connection.changeHint",
                "To change, go to Settings > tldw Server, or update NEXT_PUBLIC_API_URL and rebuild."
              )}
        </p>
      </div>

      <div>
        <h2 className="text-base font-semibold leading-7 text-text">
          {t("generalSettings.title")}
        </h2>
        <div className="border-b border-border mt-3"></div>
      </div>

      <div className="flex flex-row justify-between">
        <span className="text-text">
          {t("generalSettings.settings.language.label")}
        </span>

        <Select
          aria-label={t("generalSettings.settings.language.label")}
          placeholder={t("generalSettings.settings.language.placeholder")}
          allowClear
          showSearch
          style={{ width: "200px" }}
          options={supportLanguage}
          value={locale}
          filterOption={(input, option) =>
            option!.label.toLowerCase().indexOf(input.toLowerCase()) >= 0 ||
            option!.value.toLowerCase().indexOf(input.toLowerCase()) >= 0
          }
          onChange={(value) => {
            changeLocale(value)
          }}
        />
      </div>
      <div className="flex flex-row justify-between">
        <div className="inline-flex items-center gap-2">
          <span className="text-text">
            {t("generalSettings.settings.sendNotificationAfterIndexing.label")}
          </span>
        </div>

        <Switch
          checked={sendNotificationAfterIndexing}
          onChange={setSendNotificationAfterIndexing}
          aria-label={t("generalSettings.settings.sendNotificationAfterIndexing.label")}
        />
      </div>

      <div className="flex flex-row justify-between">
        <div className="inline-flex items-center gap-2">
          <span className="text-text">
            {t("generalSettings.settings.ollamaStatus.label")}
          </span>
        </div>

        <Switch
          checked={checkOllamaStatus}
          onChange={(checked) => setCheckOllamaStatus(checked)}
          aria-label={t("generalSettings.settings.ollamaStatus.label")}
        />
      </div>

      <div className="flex flex-row justify-between">
        <div className="inline-flex items-center gap-2">
          <span className="text-text">
            {t(
              "generalSettings.settings.onboardingAutoFinish.label",
              "Auto-finish onboarding after successful connection"
            )}
          </span>
        </div>

        <Switch
          checked={onboardingAutoFinish}
          onChange={(checked) => setOnboardingAutoFinish(checked)}
          aria-label={t("generalSettings.settings.onboardingAutoFinish.label", "Auto-finish onboarding after successful connection")}
        />
      </div>

      <div className="flex flex-row justify-between">
        <div className="inline-flex items-center gap-2">
          <span className="text-text">
            {t(
              "generalSettings.settings.restartOnboarding.label",
              "Restart onboarding from the beginning"
            )}
          </span>
        </div>

        <button
          type="button"
          className="text-xs text-primary hover:text-primaryStrong"
          onClick={() => {
            Modal.confirm({
              title: t(
                "generalSettings.settings.restartOnboarding.confirmTitle",
                "Restart onboarding?"
              ),
              content: t(
                "generalSettings.settings.restartOnboarding.confirmMessage",
                "This will reset your onboarding state and take you back to the setup flow."
              ),
              onOk: async () => {
                try {
                  await restartOnboarding()
                  notification.success({
                    message: t(
                      "generalSettings.settings.restartOnboarding.toast",
                      "Onboarding has been reset"
                    )
                  })
                  navigate("/")
                } catch (err) {
                  console.error("Failed to restart onboarding:", err)
                  notification.error({
                    message: t(
                      "generalSettings.settings.restartOnboarding.error",
                      "Failed to restart onboarding. Please try again."
                    )
                  })
                }
              }
            })
          }}
        >
          {t(
            "generalSettings.settings.restartOnboarding.button",
            "Restart onboarding"
          )}
        </button>
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
          type="button"
          className="text-xs text-primary hover:text-primaryStrong disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={completedTutorials.length === 0}
          onClick={() => {
            Modal.confirm({
              title: t(
                "generalSettings.settings.resetTutorials.confirmTitle",
                "Reset tutorial progress?"
              ),
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
              }
            })
          }}
        >
          {t(
            "generalSettings.settings.resetTutorials.button",
            "Reset tutorials"
          )}
        </button>
      </div>

      {/* Persona selection */}
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
        <div className="border-b border-border mb-3"></div>
        <Radio.Group
          value={userPersona ?? "explorer"}
          onChange={(e) => {
            const val = e.target.value as string
            handlePersonaChange(val === "explorer" ? null : (val as UserPersona)).catch(
              (err) => console.error("[GeneralSettings] Failed to change persona", err)
            )
          }}
          className="flex flex-col gap-2"
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

      {/* Browser extension promotion (webui only) */}
      {!isExtensionRuntime() && (
        <Alert
          type="info"
          showIcon
          message={t(
            "generalSettings.extensionPromo.title",
            "Browser Extension Available"
          )}
          description={t(
            "generalSettings.extensionPromo.description",
            "Get the tldw browser extension for quick access to chat, ingestion, and more from any tab."
          )}
          action={
            <a
              href="https://github.com/rmusser01/tldw_server"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center rounded border border-border bg-surface2 px-2 py-1 text-xs text-primary hover:bg-surface3"
            >
              {t("generalSettings.extensionPromo.cta", "Learn More")}
            </a>
          }
        />
      )}

      <div className="space-y-2">
        <div className="flex flex-row justify-between">
          <span className="text-text">
            {t("generalSettings.settings.enableOcrAssets.label")}
          </span>

          <Switch
            checked={enableOcrAssets}
            onChange={(checked) => setEnableOcrAssets(checked)}
            aria-label={t("generalSettings.settings.enableOcrAssets.label")}
          />
        </div>
        {!enableOcrAssets && (
          <Alert
            type="info"
            showIcon
            title={t(
              "generalSettings.settings.enableOcrAssets.downloadNotice",
              "Enable to download OCR language assets for image text recognition"
            )}
            className="!py-1.5 !text-xs"
          />
        )}
        {enableOcrAssets && (
          <Alert
            type="success"
            showIcon
            title={t(
              "generalSettings.settings.enableOcrAssets.assetsEnabled",
              "OCR assets enabled and ready"
            )}
            className="!py-1.5 !text-xs"
          />
        )}
      </div>

      <div className="flex flex-row justify-between">
        <span className="text-text">
          {t("generalSettings.settings.ocrLanguage.label")}
        </span>

        <Select
          aria-label={t("generalSettings.settings.ocrLanguage.label")}
          placeholder={t("generalSettings.settings.ocrLanguage.placeholder")}
          showSearch
          style={{ width: "200px" }}
          options={ocrLanguages}
          value={defaultOCRLanguage}
          filterOption={(input, option) =>
            option!.label.toLowerCase().indexOf(input.toLowerCase()) >= 0 ||
            option!.value.toLowerCase().indexOf(input.toLowerCase()) >= 0
          }
          onChange={(value) => {
            setDefaultOCRLanguage(value)
          }}
        />
      </div>

      <ThemePicker />
      <SearchModeSettings />
      <SystemSettings />
    </dl>
  )
}
