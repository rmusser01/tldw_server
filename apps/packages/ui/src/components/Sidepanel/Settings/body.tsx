import { Switch } from "antd"
import { useTranslation } from "react-i18next"
import { useStorage } from "@plasmohq/storage/hook"

import { ThemePicker } from "@/components/Common/Settings/ThemePicker"

const shortcuts = [
  {
    path: "/settings",
    label: "Setup & Recovery",
    description: "Connection, auth, provider keys, models, and diagnostics."
  },
  {
    path: "/settings/preferences",
    label: "Preferences",
    description: "Language, notifications, persona, tutorials, and web search."
  },
  {
    path: "/settings/ui",
    label: "UI customization",
    description: "Theme, shortcuts, display defaults, and visual behavior."
  },
  {
    path: "/settings/tldw",
    label: "Server & auth",
    description: "Change the server URL or API key."
  },
  {
    path: "/settings/health",
    label: "Full diagnostics",
    description: "Open detailed server and subsystem checks."
  }
] as const

type BrowserRuntimeGlobal = typeof globalThis & {
  browser?: {
    runtime?: {
      getURL?: (path: string) => string
    }
  }
}

const resolveOptionsHref = (path: string) => {
  try {
    const runtime = (globalThis as BrowserRuntimeGlobal).browser?.runtime
    if (runtime?.getURL) {
      return runtime.getURL(`/options.html#${path}`)
    }
  } catch {
    // Fall back to the hosted WebUI path when extension APIs are unavailable.
  }

  return path
}

export const SettingsBody = () => {
  const { t } = useTranslation("settings")
  const [copilotResumeLastChat, setCopilotResumeLastChat] = useStorage(
    "copilotResumeLastChat",
    false
  )
  const [hideCurrentChatModelSettings, setHideCurrentChatModelSettings] =
    useStorage("hideCurrentChatModelSettings", false)

  return (
    <div className="flex flex-col gap-4 p-4">
      <section className="space-y-3">
        <div>
          <h2 className="text-base font-semibold text-text">
            {t("sidepanelSettings.shortcuts.title", "Settings shortcuts")}
          </h2>
          <p className="mt-1 text-xs text-text-muted">
            {t(
              "sidepanelSettings.shortcuts.description",
              "Jump to the full settings page that owns the change."
            )}
          </p>
        </div>
        <div className="space-y-2">
          {shortcuts.map((shortcut) => (
            <a
              aria-label={shortcut.label}
              className="block rounded-md border border-border bg-surface p-3 text-text transition hover:bg-surface2"
              href={resolveOptionsHref(shortcut.path)}
              key={shortcut.path}
              rel="noreferrer"
              target="_blank"
            >
              <span className="text-sm font-medium">{shortcut.label}</span>
              <span className="mt-1 block text-xs text-text-muted">
                {shortcut.description}
              </span>
            </a>
          ))}
        </div>
      </section>

      <section className="space-y-3 rounded-md border border-border bg-surface p-3">
        <h3 className="text-sm font-semibold text-text">
          {t("sidepanelSettings.local.title", "Sidepanel behavior")}
        </h3>
        <div className="flex items-center justify-between gap-3">
          <span className="text-sm text-text">
            {t("generalSettings.settings.copilotResumeLastChat.label")}
          </span>
          <Switch
            checked={copilotResumeLastChat}
            onChange={setCopilotResumeLastChat}
          />
        </div>
        <div className="flex items-center justify-between gap-3">
          <span className="text-sm text-text">
            {t("generalSettings.settings.hideCurrentChatModelSettings.label")}
          </span>
          <Switch
            checked={hideCurrentChatModelSettings}
            onChange={setHideCurrentChatModelSettings}
          />
        </div>
      </section>

      <ThemePicker />
    </div>
  )
}
