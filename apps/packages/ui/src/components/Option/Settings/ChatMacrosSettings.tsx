import React from "react"

import {
  cloneChatMacro,
  getChatMacroSettings,
  listChatMacros,
  setChatMacroEnabled,
  updateChatMacroSettings,
  validateChatMacro,
  type ChatMacroSummary
} from "@/services/chat-macros"

const stringifySettings = (settings: Record<string, unknown>): string =>
  JSON.stringify(settings, null, 2)

const responseError = (status: number, error?: string): string =>
  error || `Request failed (${status})`

export const ChatMacrosSettings = () => {
  const [macros, setMacros] = React.useState<ChatMacroSummary[]>([])
  const [loading, setLoading] = React.useState(true)
  const [busyMacro, setBusyMacro] = React.useState<string | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const [cloneName, setCloneName] = React.useState("")
  const [settingsText, setSettingsText] = React.useState("{}")
  const [settingsMessage, setSettingsMessage] = React.useState<string | null>(null)
  const [validateRaw, setValidateRaw] = React.useState("")
  const [validationMessage, setValidationMessage] = React.useState<string | null>(null)

  const loadData = React.useCallback(async () => {
    setLoading(true)
    setError(null)
    const [macroResponse, settingsResponse] = await Promise.all([
      listChatMacros(),
      getChatMacroSettings()
    ])

    if (!macroResponse.ok || !macroResponse.data) {
      setError(responseError(macroResponse.status, macroResponse.error))
    } else {
      setMacros(macroResponse.data.macros)
    }

    if (settingsResponse.ok && settingsResponse.data) {
      setSettingsText(stringifySettings(settingsResponse.data.settings))
    }
    setLoading(false)
  }, [])

  React.useEffect(() => {
    let cancelled = false

    const run = async () => {
      setLoading(true)
      setError(null)
      const [macroResponse, settingsResponse] = await Promise.all([
        listChatMacros(),
        getChatMacroSettings()
      ])
      if (cancelled) return

      if (!macroResponse.ok || !macroResponse.data) {
        setError(responseError(macroResponse.status, macroResponse.error))
      } else {
        setMacros(macroResponse.data.macros)
      }

      if (settingsResponse.ok && settingsResponse.data) {
        setSettingsText(stringifySettings(settingsResponse.data.settings))
      }
      setLoading(false)
    }

    void run()
    return () => {
      cancelled = true
    }
  }, [])

  const toggleMacro = React.useCallback(
    async (macro: ChatMacroSummary) => {
      setBusyMacro(macro.name)
      setError(null)
      const response = await setChatMacroEnabled(macro.name, !macro.enabled)
      setBusyMacro(null)
      if (!response.ok) {
        setError(responseError(response.status, response.error))
        return
      }
      void loadData()
    },
    [loadData]
  )

  const cloneWrapup = React.useCallback(
    async (macro: ChatMacroSummary) => {
      const trimmed = cloneName.trim()
      if (!trimmed) return

      setBusyMacro(macro.name)
      setError(null)
      const response = await cloneChatMacro(macro.name, {
        name: trimmed,
        command: trimmed
      })
      setBusyMacro(null)
      if (!response.ok) {
        setError(responseError(response.status, response.error))
        return
      }
      setCloneName("")
      void loadData()
    },
    [cloneName, loadData]
  )

  const saveSettings = React.useCallback(async () => {
    setSettingsMessage(null)
    setError(null)
    let parsed: Record<string, unknown>
    try {
      const loaded = JSON.parse(settingsText)
      if (!loaded || typeof loaded !== "object" || Array.isArray(loaded)) {
        throw new Error("Macro settings JSON must be an object")
      }
      parsed = loaded as Record<string, unknown>
    } catch (err) {
      setSettingsMessage(err instanceof Error ? err.message : "Invalid settings JSON")
      return
    }

    const response = await updateChatMacroSettings(parsed)
    if (!response.ok) {
      setSettingsMessage(responseError(response.status, response.error))
      return
    }
    setSettingsMessage("Macro settings saved")
  }, [settingsText])

  const validateRawMacro = React.useCallback(async () => {
    setValidationMessage(null)
    const response = await validateChatMacro(validateRaw)
    if (!response.ok || !response.data) {
      setValidationMessage(responseError(response.status, response.error))
      return
    }
    if (response.data.valid) {
      setValidationMessage("Macro YAML is valid")
      return
    }
    setValidationMessage(response.data.error || "Macro YAML is invalid")
  }, [validateRaw])

  const firstMacro = macros[0] ?? null

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-6 px-4 py-4 text-text">
      <header>
        <h1 className="text-xl font-semibold">Chat macros</h1>
      </header>

      {error ? (
        <p className="rounded-md border border-danger/40 bg-danger/10 px-3 py-2 text-sm font-medium text-danger" role="alert">
          {error}
        </p>
      ) : null}

      <section className="space-y-3">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold">Macros</h2>
          {loading ? <span className="text-xs text-text-muted">Loading</span> : null}
        </div>

        <div className="overflow-hidden rounded-md border border-border">
          <table className="w-full border-collapse text-sm">
            <thead className="bg-surface2 text-left text-xs uppercase text-text-muted">
              <tr>
                <th className="px-3 py-2 font-medium">Command</th>
                <th className="px-3 py-2 font-medium">Source</th>
                <th className="px-3 py-2 font-medium">Description</th>
                <th className="px-3 py-2 text-right font-medium">Enabled</th>
              </tr>
            </thead>
            <tbody>
              {macros.map((macro) => (
                <tr key={`${macro.source}:${macro.name}`} className="border-t border-border">
                  <td className="px-3 py-2 font-medium">/{macro.command}</td>
                  <td className="px-3 py-2 text-text-muted">{macro.source}</td>
                  <td className="px-3 py-2 text-text-muted">
                    {macro.description || "No description"}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <button
                      type="button"
                      role="switch"
                      aria-checked={macro.enabled}
                      aria-label={`Toggle /${macro.command}`}
                      disabled={busyMacro === macro.name}
                      onClick={() => void toggleMacro(macro)}
                      className={[
                        "inline-flex h-6 w-11 items-center rounded-full border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50",
                        macro.enabled
                          ? "border-primary bg-primary"
                          : "border-border bg-surface"
                      ].join(" ")}
                    >
                      <span
                        className={[
                          "block h-4 w-4 rounded-full bg-white transition-transform",
                          macro.enabled ? "translate-x-5" : "translate-x-1"
                        ].join(" ")}
                      />
                    </button>
                  </td>
                </tr>
              ))}
              {!loading && macros.length === 0 ? (
                <tr>
                  <td className="px-3 py-3 text-sm text-text-muted" colSpan={4}>
                    No macros found.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-md border border-border bg-surface px-3 py-3">
          <h2 className="text-sm font-semibold">Clone a macro</h2>
          <label className="mt-3 block text-sm font-medium" htmlFor="chat-macro-clone-name">
            Clone macro name
          </label>
          <input
            id="chat-macro-clone-name"
            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-focus"
            value={cloneName}
            onChange={(event) => setCloneName(event.target.value)}
          />
          <button
            type="button"
            className="mt-3 inline-flex min-h-[36px] items-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-primaryStrong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50"
            disabled={!firstMacro || !cloneName.trim()}
            onClick={() => firstMacro && void cloneWrapup(firstMacro)}
          >
            Clone /{firstMacro?.command || "macro"}
          </button>
        </div>

        <div className="rounded-md border border-border bg-surface px-3 py-3">
          <h2 className="text-sm font-semibold">Validate YAML</h2>
          <label className="mt-3 block text-sm font-medium" htmlFor="chat-macro-validate-yaml">
            Validate macro YAML
          </label>
          <textarea
            id="chat-macro-validate-yaml"
            className="mt-1 min-h-[118px] w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-sm outline-none focus:border-primary focus:ring-2 focus:ring-focus"
            value={validateRaw}
            onChange={(event) => setValidateRaw(event.target.value)}
          />
          <button
            type="button"
            className="mt-3 inline-flex min-h-[36px] items-center rounded-md border border-border px-4 py-2 text-sm font-medium text-text transition-colors hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
            onClick={() => void validateRawMacro()}
          >
            Validate macro
          </button>
          {validationMessage ? (
            <p className="mt-2 text-sm text-text-muted">{validationMessage}</p>
          ) : null}
        </div>
      </section>

      <section className="rounded-md border border-border bg-surface px-3 py-3">
        <h2 className="text-sm font-semibold">Output profiles</h2>
        <label className="mt-3 block text-sm font-medium" htmlFor="chat-macro-settings-json">
          Macro settings JSON
        </label>
        <textarea
          id="chat-macro-settings-json"
          className="mt-1 min-h-[180px] w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-sm outline-none focus:border-primary focus:ring-2 focus:ring-focus"
          value={settingsText}
          onChange={(event) => setSettingsText(event.target.value)}
        />
        <button
          type="button"
          className="mt-3 inline-flex min-h-[36px] items-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-primaryStrong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          onClick={() => void saveSettings()}
        >
          Save macro settings
        </button>
        {settingsMessage ? (
          <p className="mt-2 text-sm text-text-muted">{settingsMessage}</p>
        ) : null}
      </section>
    </div>
  )
}

export default ChatMacrosSettings
