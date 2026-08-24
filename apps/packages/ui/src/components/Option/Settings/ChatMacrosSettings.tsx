import React from "react"
import { Tooltip } from "antd"
import { Plus, RefreshCw, Upload } from "lucide-react"

import {
  cloneChatMacro,
  getChatMacroSettings,
  listChatMacros,
  setChatMacroEnabled,
  type ChatMacroSettings,
  type ChatMacroSummary
} from "@/services/chat-macros"
import { ChatMacroEditor } from "./ChatMacroEditor"
import { OutputProfileEditor } from "./OutputProfileEditor"

type ActiveTab = "macros" | "profiles"
type Selection = { kind: "macro"; name: string } | { kind: "new" } | null

const responseError = (status: number, error?: string): string =>
  error || `Request failed (${status})`

const headerButtonClassName =
  "inline-flex h-9 items-center gap-1.5 rounded-md border border-border bg-surface px-3 text-sm font-medium text-text transition-colors hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50"

const iconButtonClassName =
  "inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-border bg-surface text-text transition-colors hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50"

const tabClassName = (active: boolean): string =>
  [
    "inline-flex h-9 items-center border-b-2 px-3 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus",
    active
      ? "border-primary text-primary"
      : "border-transparent text-text-muted hover:border-border hover:text-text"
  ].join(" ")

const selectionKey = (macro: ChatMacroSummary): string => `${macro.source}:${macro.name}`

export const ChatMacrosSettings = () => {
  const [activeTab, setActiveTab] = React.useState<ActiveTab>("macros")
  const [macros, setMacros] = React.useState<ChatMacroSummary[]>([])
  const [catalogLoading, setCatalogLoading] = React.useState(true)
  const [catalogError, setCatalogError] = React.useState<string | null>(null)
  const [settings, setSettings] = React.useState<ChatMacroSettings | null>(null)
  const [settingsLoading, setSettingsLoading] = React.useState(true)
  const [settingsError, setSettingsError] = React.useState<string | null>(null)
  const [selection, setSelection] = React.useState<Selection>(null)
  const [busyMacro, setBusyMacro] = React.useState<string | null>(null)
  const [cloneSource, setCloneSource] = React.useState<ChatMacroSummary | null>(null)
  const [cloneName, setCloneName] = React.useState("")
  const [cloneError, setCloneError] = React.useState<string | null>(null)
  const [cloneBusy, setCloneBusy] = React.useState(false)
  const [importRequest, setImportRequest] = React.useState(0)
  const editorHostRef = React.useRef<HTMLDivElement>(null)
  const mountedRef = React.useRef(false)
  const catalogRequestRef = React.useRef(0)
  const settingsRequestRef = React.useRef(0)

  const refreshCatalog = React.useCallback(async (preferredName?: string) => {
    const request = ++catalogRequestRef.current
    setCatalogLoading(true)
    setCatalogError(null)

    try {
      const response = await listChatMacros()
      if (!mountedRef.current || request !== catalogRequestRef.current) return

      if (!response.ok || !response.data) {
        setCatalogError(responseError(response.status, response.error))
        return
      }

      const nextMacros = response.data.macros
      setMacros(nextMacros)
      setSelection((current) => {
        if (preferredName && nextMacros.some((macro) => macro.name === preferredName)) {
          return { kind: "macro", name: preferredName }
        }
        if (current?.kind === "new") return current
        if (current?.kind === "macro" && nextMacros.some((macro) => macro.name === current.name)) {
          return current
        }
        return nextMacros[0] ? { kind: "macro", name: nextMacros[0].name } : { kind: "new" }
      })
    } catch (error) {
      if (mountedRef.current && request === catalogRequestRef.current) {
        setCatalogError(error instanceof Error ? error.message : "Unable to load macro catalog.")
      }
    } finally {
      if (mountedRef.current && request === catalogRequestRef.current) {
        setCatalogLoading(false)
      }
    }
  }, [])

  const refreshSettings = React.useCallback(async () => {
    const request = ++settingsRequestRef.current
    setSettingsLoading(true)
    setSettingsError(null)

    try {
      const response = await getChatMacroSettings()
      if (!mountedRef.current || request !== settingsRequestRef.current) return

      if (!response.ok || !response.data?.settings) {
        setSettingsError(responseError(response.status, response.error))
        return
      }

      setSettings(response.data.settings)
    } catch (error) {
      if (mountedRef.current && request === settingsRequestRef.current) {
        setSettingsError(error instanceof Error ? error.message : "Unable to load macro settings.")
      }
    } finally {
      if (mountedRef.current && request === settingsRequestRef.current) {
        setSettingsLoading(false)
      }
    }
  }, [])

  React.useEffect(() => {
    mountedRef.current = true
    void refreshCatalog()
    void refreshSettings()

    return () => {
      mountedRef.current = false
      catalogRequestRef.current += 1
      settingsRequestRef.current += 1
    }
  }, [refreshCatalog, refreshSettings])

  React.useEffect(() => {
    if (importRequest === 0) return
    editorHostRef.current?.querySelector<HTMLInputElement>('input[type="file"]')?.click()
  }, [importRequest])

  const selectedMacro = selection?.kind === "macro"
    ? macros.find((macro) => macro.name === selection.name) || null
    : null
  const outputProfileNames = settings
    ? Object.keys(settings.output_profiles)
    : ["default"]

  const selectMacro = React.useCallback((macro: ChatMacroSummary) => {
    setCloneSource(null)
    setCloneName("")
    setCloneError(null)
    setSelection({ kind: "macro", name: macro.name })
  }, [])

  const openNewMacro = React.useCallback(() => {
    setCloneSource(null)
    setCloneName("")
    setCloneError(null)
    setSelection({ kind: "new" })
  }, [])

  const importMacro = React.useCallback(() => {
    openNewMacro()
    setImportRequest((current) => current + 1)
  }, [openNewMacro])

  const toggleMacro = React.useCallback(async (macro: ChatMacroSummary) => {
    setBusyMacro(macro.name)
    setCatalogError(null)

    try {
      const response = await setChatMacroEnabled(macro.name, !macro.enabled)
      if (!mountedRef.current) return
      if (!response.ok) {
        setCatalogError(responseError(response.status, response.error))
        return
      }
      await refreshCatalog(macro.name)
    } catch (error) {
      if (mountedRef.current) {
        setCatalogError(error instanceof Error ? error.message : "Unable to update macro state.")
      }
    } finally {
      if (mountedRef.current) setBusyMacro(null)
    }
  }, [refreshCatalog])

  const requestClone = React.useCallback((macro: ChatMacroSummary) => {
    setCloneSource(macro)
    setCloneName("")
    setCloneError(null)
  }, [])

  const cloneMacro = React.useCallback(async () => {
    if (!cloneSource) return
    const name = cloneName.trim()
    if (!name) {
      setCloneError("Clone macro name is required.")
      return
    }

    setCloneBusy(true)
    setCloneError(null)
    try {
      const response = await cloneChatMacro(cloneSource.name, { name, command: name })
      if (!mountedRef.current) return
      if (!response.ok) {
        setCloneError(responseError(response.status, response.error))
        return
      }
      setCloneSource(null)
      setCloneName("")
      await refreshCatalog(name)
    } catch (error) {
      if (mountedRef.current) {
        setCloneError(error instanceof Error ? error.message : "Unable to clone macro.")
      }
    } finally {
      if (mountedRef.current) setCloneBusy(false)
    }
  }, [cloneName, cloneSource, refreshCatalog])

  const handleSaved = React.useCallback((name: string) => {
    if (!mountedRef.current) return
    setCloneSource(null)
    setCloneName("")
    setSelection({ kind: "macro", name })
    void refreshCatalog(name)
  }, [refreshCatalog])

  const handleDeleted = React.useCallback(() => {
    if (!mountedRef.current) return
    setCloneSource(null)
    setCloneName("")
    setSelection(null)
    void refreshCatalog()
  }, [refreshCatalog])

  const handleSettingsSaved = React.useCallback((nextSettings: ChatMacroSettings) => {
    if (mountedRef.current) setSettings(nextSettings)
  }, [])

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-5 px-4 py-4 text-text">
      <header className="flex flex-col gap-3 border-b border-border pb-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-xl font-semibold">Chat macros</h1>
          <p className="mt-1 max-w-3xl text-sm text-text-muted">
            Author reusable chat workflows and shape how their results are returned.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2" data-testid="chat-macro-header-actions">
          <button type="button" className={headerButtonClassName} onClick={openNewMacro}>
            <Plus aria-hidden="true" size={16} />
            New macro
          </button>
          <button
            type="button"
            aria-label="Import macro"
            className={headerButtonClassName}
            onClick={importMacro}
          >
            <Upload aria-hidden="true" size={16} />
            Import
          </button>
          <Tooltip title="Refresh macros">
            <button
              type="button"
              aria-label="Refresh macros"
              className={iconButtonClassName}
              onClick={() => {
                void refreshCatalog()
                void refreshSettings()
              }}
            >
              <RefreshCw aria-hidden="true" size={16} />
            </button>
          </Tooltip>
        </div>
      </header>

      <div className="flex border-b border-border" role="tablist" aria-label="Chat macro settings views">
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "macros"}
          className={tabClassName(activeTab === "macros")}
          onClick={() => setActiveTab("macros")}
        >
          Macros
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "profiles"}
          className={tabClassName(activeTab === "profiles")}
          onClick={() => setActiveTab("profiles")}
        >
          Output profiles
        </button>
      </div>

      {activeTab === "macros" ? (
        <div className="grid min-w-0 gap-5 xl:grid-cols-[minmax(220px,300px)_minmax(0,1fr)]">
          <aside className="min-w-0 border-b border-border pb-4 xl:border-b-0 xl:border-r xl:pb-0 xl:pr-4" aria-label="Macro catalog">
            <div className="mb-2 flex items-center justify-between gap-3">
              <h2 className="text-sm font-semibold">Macros</h2>
              {catalogLoading ? <span className="text-xs text-text-muted">Loading macros</span> : null}
            </div>

            {catalogError ? (
              <div className="mb-3 flex flex-wrap items-center gap-2 rounded-md border border-danger/40 bg-danger/10 px-3 py-2" role="alert">
                <span className="min-w-0 flex-1 text-sm font-medium text-danger">{catalogError}</span>
                <button
                  type="button"
                  className="text-sm font-medium text-danger underline decoration-danger/50 underline-offset-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                  onClick={() => void refreshCatalog()}
                >
                  Retry macro list
                </button>
              </div>
            ) : null}

            <div className="divide-y divide-border border-y border-border">
              {macros.map((macro) => {
                const selected = selectedMacro?.name === macro.name
                return (
                  <div
                    key={selectionKey(macro)}
                    className={[
                      "grid min-w-0 grid-cols-[minmax(0,1fr)_auto] gap-2 px-2 py-2",
                      selected ? "bg-surface2/70" : "bg-transparent"
                    ].join(" ")}
                  >
                    <button
                      type="button"
                      aria-label={`Select /${macro.command}`}
                      aria-pressed={selected}
                      className="min-w-0 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                      onClick={() => selectMacro(macro)}
                    >
                      <span className="block truncate text-sm font-medium text-text">/{macro.command}</span>
                      <span className="mt-1 flex flex-wrap gap-x-2 gap-y-1 text-xs text-text-muted">
                        <span>{macro.source}</span>
                        <span>{macro.enabled ? "Enabled" : "Disabled"}</span>
                        <span className="text-success">Valid</span>
                      </span>
                    </button>
                    <button
                      type="button"
                      role="switch"
                      aria-checked={macro.enabled}
                      aria-label={`Toggle /${macro.command}`}
                      disabled={busyMacro === macro.name}
                      onClick={() => void toggleMacro(macro)}
                      className={[
                        "mt-1 inline-flex h-6 w-11 items-center rounded-full border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50",
                        macro.enabled ? "border-primary bg-primary" : "border-border bg-surface"
                      ].join(" ")}
                    >
                      <span
                        className={[
                          "block h-4 w-4 rounded-full bg-white transition-transform",
                          macro.enabled ? "translate-x-5" : "translate-x-1"
                        ].join(" ")}
                      />
                    </button>
                  </div>
                )
              })}
              {!catalogLoading && macros.length === 0 ? (
                <p className="px-2 py-3 text-sm text-text-muted">No macros found.</p>
              ) : null}
            </div>
          </aside>

          <div ref={editorHostRef} className="min-w-0">
            {cloneSource ? (
              <section className="mb-4 border-y border-border py-3" aria-labelledby="chat-macro-clone-title">
                <h2 id="chat-macro-clone-title" className="mb-3 text-sm font-semibold text-text">
                  Clone /{cloneSource.command}
                </h2>
                <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
                  <label className="min-w-0 flex-1 text-sm font-medium text-text" htmlFor="chat-macro-clone-name">
                    Clone macro name
                    <input
                      id="chat-macro-clone-name"
                      className="mt-1 h-9 w-full rounded-md border border-border bg-background px-2.5 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-focus disabled:cursor-not-allowed disabled:opacity-60"
                      value={cloneName}
                      disabled={cloneBusy}
                      onChange={(event) => {
                        setCloneName(event.target.value)
                        setCloneError(null)
                      }}
                    />
                  </label>
                  <button
                    type="button"
                    className="inline-flex h-9 items-center justify-center rounded-md bg-primary px-3 text-sm font-medium text-white transition-colors hover:bg-primaryStrong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50"
                    disabled={cloneBusy || !cloneName.trim()}
                    onClick={() => void cloneMacro()}
                  >
                    {cloneBusy ? "Cloning" : `Clone /${cloneSource.command}`}
                  </button>
                </div>
                {cloneError ? <p className="mt-2 text-sm font-medium text-danger" role="alert">{cloneError}</p> : null}
              </section>
            ) : null}

            <ChatMacroEditor
              selected={selectedMacro}
              outputProfileNames={outputProfileNames}
              onSaved={handleSaved}
              onDeleted={handleDeleted}
              onCloneRequested={requestClone}
            />
          </div>
        </div>
      ) : null}

      {activeTab === "profiles" ? (
        <section aria-label="Output profile manager">
          {settingsLoading ? <p className="text-sm text-text-muted">Loading output profiles</p> : null}
          {!settingsLoading && settingsError ? (
            <div className="flex flex-wrap items-center gap-3 rounded-md border border-danger/40 bg-danger/10 px-3 py-2" role="alert">
              <span className="min-w-0 flex-1 text-sm font-medium text-danger">{settingsError}</span>
              <button
                type="button"
                className="text-sm font-medium text-danger underline decoration-danger/50 underline-offset-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                onClick={() => void refreshSettings()}
              >
                Retry settings
              </button>
            </div>
          ) : null}
          {!settingsLoading && !settingsError && settings ? (
            <OutputProfileEditor settings={settings} onSaved={handleSettingsSaved} />
          ) : null}
        </section>
      ) : null}
    </div>
  )
}

export default ChatMacrosSettings
