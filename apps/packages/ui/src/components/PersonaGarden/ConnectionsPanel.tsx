import React from "react"
import { useTranslation } from "react-i18next"

import { tldwClient } from "@/services/tldw/TldwApiClient"
import { toAllowedPath } from "@/services/tldw/path-utils"

type PersonaConnection = {
  id: string
  persona_id: string
  name: string
  base_url: string
  auth_type: string
  headers_template?: Record<string, string>
  timeout_ms?: number
  allowed_hosts?: string[]
  secret_configured?: boolean
  key_hint?: string | null
  created_at?: string | null
  last_modified?: string | null
}

type PersonaConnectionTestResult = {
  ok: boolean
  status_code?: number | null
  body_preview?: unknown
  latency_ms?: number | null
  error?: string | null
}

type ConnectionFormState = {
  name: string
  baseUrl: string
  authType: string
  secret: string
  headersTemplateText: string
  timeoutMs: string
}

type ConnectionsPanelProps = {
  selectedPersonaId: string
  selectedPersonaName: string
  isActive?: boolean
  onConnectionSaved?: () => void
  onConnectionTestSucceeded?: () => void
  handoffFocusRequest?: {
    section: "connection_form" | "saved_connections"
    token: number
    connectionId?: string | null
    connectionName?: string | null
  } | null
  onSetupHandoffFocusConsumed?: (token: number) => void
}

const DEFAULT_FORM_STATE: ConnectionFormState = {
  name: "",
  baseUrl: "",
  authType: "none",
  secret: "",
  headersTemplateText: "{}",
  timeoutMs: "15000"
}

const parseHeadersTemplate = (
  rawValue: string
): { ok: true; value: Record<string, string> } | { ok: false; error: string } => {
  const trimmed = rawValue.trim()
  if (!trimmed) return { ok: true, value: {} }
  try {
    const parsed = JSON.parse(trimmed)
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { ok: false, error: "Headers template must be a JSON object." }
    }
    const next: Record<string, string> = {}
    for (const [key, value] of Object.entries(parsed)) {
      next[String(key)] = String(value)
    }
    return { ok: true, value: next }
  } catch {
    return { ok: false, error: "Headers template must be valid JSON." }
  }
}

const formatHeadersTemplate = (value?: Record<string, string>) =>
  JSON.stringify(value ?? {}, null, 2)

const isPersonaConnection = (value: unknown): value is PersonaConnection => {
  if (!value || typeof value !== "object") return false
  const record = value as Record<string, unknown>
  return (
    typeof record.id === "string" &&
    typeof record.persona_id === "string" &&
    typeof record.name === "string" &&
    typeof record.base_url === "string" &&
    typeof record.auth_type === "string"
  )
}

const summarizeTestBodyPreview = (value: unknown): string | null => {
  if (typeof value === "string") {
    const trimmed = value.trim()
    return trimmed || null
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return "No response items"
    return `${value.length} response item${value.length === 1 ? "" : "s"}`
  }
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>
    for (const key of ["message", "detail", "result", "status", "response_text"]) {
      const candidate = record[key]
      if (typeof candidate === "string" && candidate.trim()) {
        return candidate.trim()
      }
    }
    const serialized = JSON.stringify(record)
    return serialized.length > 180 ? `${serialized.slice(0, 177)}...` : serialized
  }
  return null
}

export const ConnectionsPanel: React.FC<ConnectionsPanelProps> = ({
  selectedPersonaId,
  selectedPersonaName,
  isActive = false,
  onConnectionSaved,
  onConnectionTestSucceeded,
  handoffFocusRequest = null,
  onSetupHandoffFocusConsumed
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const [connections, setConnections] = React.useState<PersonaConnection[]>([])
  const [connectionsLoaded, setConnectionsLoaded] = React.useState(false)
  const [loading, setLoading] = React.useState(false)
  const [saving, setSaving] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [validationError, setValidationError] = React.useState<string | null>(null)
  const [editingConnectionId, setEditingConnectionId] = React.useState<string | null>(null)
  const [deletingConnectionId, setDeletingConnectionId] = React.useState<string | null>(null)
  const [testingConnectionId, setTestingConnectionId] = React.useState<string | null>(null)
  const [testResults, setTestResults] = React.useState<Record<string, PersonaConnectionTestResult>>({})
  const [formState, setFormState] =
    React.useState<ConnectionFormState>(DEFAULT_FORM_STATE)
  const connectionNameInputRef = React.useRef<HTMLInputElement | null>(null)
  const connectionEditButtonRefs = React.useRef<Record<string, HTMLButtonElement | null>>({})
  const lastHandledHandoffTokenRef = React.useRef<number | null>(null)

  const resetConnectionUiState = React.useCallback(() => {
    setEditingConnectionId(null)
    setDeletingConnectionId(null)
    setTestingConnectionId(null)
    setTestResults({})
    setFormState(DEFAULT_FORM_STATE)
    setValidationError(null)
  }, [])

  React.useEffect(() => {
    let cancelled = false

    const load = async () => {
      if (!isActive || !selectedPersonaId) {
        setConnections([])
        setConnectionsLoaded(false)
        setError(null)
        resetConnectionUiState()
        return
      }
      setLoading(true)
      setConnectionsLoaded(false)
      try {
        const response = await tldwClient.fetchWithAuth(
          toAllowedPath(
            `/api/v1/persona/profiles/${encodeURIComponent(selectedPersonaId)}/connections`
          ),
          { method: "GET" }
        )
        if (!response.ok) {
          throw new Error(
            response.error ||
              t("sidepanel:personaGarden.connections.loadError", {
                defaultValue: "Failed to load persona connections."
              })
          )
        }
        const payload = await response.json()
        const nextRows = Array.isArray(payload)
          ? payload.filter(isPersonaConnection)
          : []
        if (!cancelled) {
          setConnections(nextRows)
          resetConnectionUiState()
          setError(null)
        }
      } catch (loadError) {
        if (!cancelled) {
          setConnections([])
          resetConnectionUiState()
          setError(
            loadError instanceof Error
              ? loadError.message
              : t("sidepanel:personaGarden.connections.loadError", {
                  defaultValue: "Failed to load persona connections."
                })
          )
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
          setConnectionsLoaded(true)
        }
      }
    }

    void load()
    return () => {
      cancelled = true
    }
  }, [isActive, resetConnectionUiState, selectedPersonaId])

  React.useEffect(() => {
    if (!isActive || !selectedPersonaId || !handoffFocusRequest || loading) return
    if (lastHandledHandoffTokenRef.current === handoffFocusRequest.token) return

    const consume = () => {
      lastHandledHandoffTokenRef.current = handoffFocusRequest.token
      onSetupHandoffFocusConsumed?.(handoffFocusRequest.token)
    }

    const focusConnectionForm = () => {
      const formTarget = connectionNameInputRef.current
      if (!formTarget) return false
      formTarget.scrollIntoView?.({ block: "nearest", behavior: "smooth" })
      formTarget.focus()
      consume()
      return true
    }

    if (handoffFocusRequest.section === "connection_form") {
      focusConnectionForm()
      return
    }

    if (!connectionsLoaded) return

    const normalizedConnectionId = String(handoffFocusRequest.connectionId || "").trim()
    const normalizedConnectionName = String(handoffFocusRequest.connectionName || "").trim()

    const matchedConnection = connections.find((connection) => {
      if (normalizedConnectionId && connection.id === normalizedConnectionId) return true
      return normalizedConnectionName.length > 0 && connection.name === normalizedConnectionName
    })

    if (matchedConnection) {
      const editButton = connectionEditButtonRefs.current[matchedConnection.id]
      if (!editButton) return
      editButton.scrollIntoView?.({ block: "nearest", behavior: "smooth" })
      editButton.focus()
      consume()
      return
    }

    focusConnectionForm()
  }, [
    connections,
    connectionsLoaded,
    handoffFocusRequest,
    isActive,
    loading,
    onSetupHandoffFocusConsumed,
    selectedPersonaId
  ])

  const updateField = React.useCallback(
    (field: keyof ConnectionFormState, value: string) => {
      setFormState((current) => ({
        ...current,
        [field]: value
      }))
    },
    []
  )

  const handleReset = React.useCallback(() => {
    setEditingConnectionId(null)
    setFormState(DEFAULT_FORM_STATE)
    setValidationError(null)
  }, [])

  const loadConnectionIntoForm = React.useCallback((connection: PersonaConnection) => {
    setEditingConnectionId(connection.id)
    setValidationError(null)
    setError(null)
    setFormState({
      name: connection.name ?? "",
      baseUrl: connection.base_url ?? "",
      authType: connection.auth_type ?? "none",
      secret: "",
      headersTemplateText: formatHeadersTemplate(connection.headers_template),
      timeoutMs: String(connection.timeout_ms ?? 15000)
    })
  }, [])

  const handleSave = React.useCallback(async () => {
    if (!selectedPersonaId) return
    const name = formState.name.trim()
    const baseUrl = formState.baseUrl.trim()
    if (!name) {
      setValidationError("Connection name is required.")
      return
    }
    if (!baseUrl) {
      setValidationError("Base URL is required.")
      return
    }
    const headersTemplateResult = parseHeadersTemplate(formState.headersTemplateText)
    if (headersTemplateResult.ok === false) {
      setValidationError(headersTemplateResult.error)
      return
    }

    const payload: Record<string, unknown> = {
      name,
      base_url: baseUrl,
      auth_type: formState.authType,
      headers_template: headersTemplateResult.value,
      timeout_ms: Number.parseInt(formState.timeoutMs, 10) || 15000
    }
    const trimmedSecret = formState.secret.trim()
    if (trimmedSecret) {
      payload.secret = trimmedSecret
    }

    setSaving(true)
    setValidationError(null)
    setError(null)
    try {
      const requestPath = editingConnectionId
        ? `/api/v1/persona/profiles/${encodeURIComponent(selectedPersonaId)}/connections/${encodeURIComponent(editingConnectionId)}`
        : `/api/v1/persona/profiles/${encodeURIComponent(selectedPersonaId)}/connections`
      const response = await tldwClient.fetchWithAuth(
        toAllowedPath(requestPath),
        {
          method: editingConnectionId ? "PUT" : "POST",
          body: payload
        }
      )
      if (!response.ok) {
        throw new Error(
          response.error ||
            (editingConnectionId
              ? "Failed to update persona connection."
              : "Failed to create persona connection.")
        )
      }
      const saved = (await response.json()) as PersonaConnection
      setConnections((current) => [saved, ...current.filter((item) => item.id !== saved.id)])
      handleReset()
      onConnectionSaved?.()
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : editingConnectionId
            ? "Failed to update persona connection."
            : "Failed to create persona connection."
      )
    } finally {
      setSaving(false)
    }
  }, [editingConnectionId, formState, handleReset, onConnectionSaved, selectedPersonaId])

  const handleDelete = React.useCallback(async (connectionId: string) => {
    if (!selectedPersonaId) return
    if (
      typeof window !== "undefined" &&
      !window.confirm("Delete this persona connection?")
    ) {
      return
    }
    setDeletingConnectionId(connectionId)
    setError(null)
    try {
      const response = await tldwClient.fetchWithAuth(
        toAllowedPath(
          `/api/v1/persona/profiles/${encodeURIComponent(selectedPersonaId)}/connections/${encodeURIComponent(connectionId)}`
        ),
        {
          method: "DELETE"
        }
      )
      if (!response.ok) {
        throw new Error(response.error || "Failed to delete persona connection.")
      }
      setConnections((current) => current.filter((item) => item.id !== connectionId))
      setTestResults((current) => {
        const next = { ...current }
        delete next[connectionId]
        return next
      })
      if (editingConnectionId === connectionId) {
        handleReset()
      }
    } catch (deleteError) {
      setError(
        deleteError instanceof Error
          ? deleteError.message
          : "Failed to delete persona connection."
      )
    } finally {
      setDeletingConnectionId(null)
    }
  }, [editingConnectionId, handleReset, selectedPersonaId])

  const handleTest = React.useCallback(async (connectionId: string) => {
    if (!selectedPersonaId) return
    setTestingConnectionId(connectionId)
    setError(null)
    try {
      const response = await tldwClient.fetchWithAuth(
        toAllowedPath(
          `/api/v1/persona/profiles/${encodeURIComponent(selectedPersonaId)}/connections/${encodeURIComponent(connectionId)}/test`
        ),
        {
          method: "POST",
          body: {}
        }
      )
      if (!response.ok) {
        throw new Error(response.error || "Failed to test persona connection.")
      }
      const result = (await response.json()) as PersonaConnectionTestResult
      setTestResults((current) => ({
        ...current,
        [connectionId]: result
      }))
      if (result.ok) {
        onConnectionTestSucceeded?.()
      }
    } catch (testError) {
      setTestResults((current) => ({
        ...current,
        [connectionId]: {
          ok: false,
          error:
            testError instanceof Error
              ? testError.message
              : "Failed to test persona connection."
        }
      }))
    } finally {
      setTestingConnectionId(null)
    }
  }, [onConnectionTestSucceeded, selectedPersonaId])

  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
        {t("sidepanel:personaGarden.connections.heading", {
          defaultValue: "Connections"
        })}
      </div>
      <div className="mt-2 space-y-3 text-sm text-text">
        <p className="text-xs text-text-muted">
          {selectedPersonaId
            ? t("sidepanel:personaGarden.connections.description", {
                defaultValue:
                  "Create reusable API connection records for {{personaName}}. Commands can reference these records instead of storing secrets inline.",
                personaName:
                  selectedPersonaName ||
                  selectedPersonaId ||
                  t("sidepanel:personaGarden.connections.currentPersona", {
                    defaultValue: "this persona"
                  })
              })
            : t("sidepanel:personaGarden.connections.noPersona", {
                defaultValue:
                  "Select a persona to manage reusable external connections."
              })}
        </p>

        {error ? (
          <div className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-700">
            {error}
          </div>
        ) : null}
        {validationError ? (
          <div className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-700">
            {validationError}
          </div>
        ) : null}

        {selectedPersonaId ? (
          <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
            <div className="rounded-md border border-border bg-bg p-3">
              <div className="text-sm font-medium text-text">
                {editingConnectionId
                  ? t("sidepanel:personaGarden.connections.editHeading", {
                      defaultValue: "Edit connection"
                    })
                  : t("sidepanel:personaGarden.connections.createHeading", {
                      defaultValue: "Create connection"
                    })}
              </div>

              <div className="mt-3 space-y-3">
                <label className="block text-xs text-text-muted">
                  {t("sidepanel:personaGarden.connections.name", {
                    defaultValue: "Connection name"
                  })}
                  <input
                    ref={connectionNameInputRef}
                    data-testid="persona-connections-name-input"
                    className="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-text"
                    value={formState.name}
                    onChange={(event) => updateField("name", event.target.value)}
                    placeholder="Slack Alerts"
                  />
                </label>

                <label className="block text-xs text-text-muted">
                  {t("sidepanel:personaGarden.connections.baseUrl", {
                    defaultValue: "Base URL"
                  })}
                  <input
                    data-testid="persona-connections-base-url-input"
                    className="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-text"
                    value={formState.baseUrl}
                    onChange={(event) => updateField("baseUrl", event.target.value)}
                    placeholder="https://api.example.com"
                  />
                </label>

                <div className="grid gap-3 md:grid-cols-2">
                  <label className="block text-xs text-text-muted">
                    {t("sidepanel:personaGarden.connections.authType", {
                      defaultValue: "Auth type"
                    })}
                    <select
                      data-testid="persona-connections-auth-type-select"
                      className="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-text"
                      value={formState.authType}
                      onChange={(event) => updateField("authType", event.target.value)}
                    >
                      <option value="none">none</option>
                      <option value="bearer">bearer</option>
                      <option value="api_key">api_key</option>
                      <option value="basic">basic</option>
                      <option value="custom_header">custom_header</option>
                    </select>
                  </label>

                  <label className="block text-xs text-text-muted">
                    {t("sidepanel:personaGarden.connections.timeoutMs", {
                      defaultValue: "Timeout (ms)"
                    })}
                    <input
                      data-testid="persona-connections-timeout-input"
                      className="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-text"
                      value={formState.timeoutMs}
                      onChange={(event) => updateField("timeoutMs", event.target.value)}
                      inputMode="numeric"
                      placeholder="15000"
                    />
                  </label>
                </div>

                <label className="block text-xs text-text-muted">
                  {t("sidepanel:personaGarden.connections.secret", {
                    defaultValue: "Secret"
                  })}
                  <input
                    data-testid="persona-connections-secret-input"
                    className="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-text"
                    type="password"
                    value={formState.secret}
                    onChange={(event) => updateField("secret", event.target.value)}
                    placeholder="Paste bearer token or API key"
                  />
                </label>

                <label className="block text-xs text-text-muted">
                  {t("sidepanel:personaGarden.connections.headersTemplate", {
                    defaultValue: "Headers template (JSON)"
                  })}
                  <textarea
                    data-testid="persona-connections-headers-input"
                    className="mt-1 min-h-[96px] w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-text"
                    value={formState.headersTemplateText}
                    onChange={(event) =>
                      updateField("headersTemplateText", event.target.value)
                    }
                  />
                </label>

                <div className="rounded-md border border-sky-500/30 bg-sky-500/10 px-3 py-2 text-xs text-sky-800">
                  {t("sidepanel:personaGarden.connections.writeOnlyHint", {
                    defaultValue:
                      "Secrets are write-only. Leave the secret field blank while editing to keep the current value. Saved connections can be tested or deleted here."
                  })}
                </div>

                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    data-testid="persona-connections-save"
                    className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
                    disabled={saving}
                    onClick={() => {
                      void handleSave()
                    }}
                  >
                    {saving
                      ? t("common:saving", "Saving...")
                      : editingConnectionId
                        ? t("common:save", "Save")
                        : t("common:create", "Create")}
                  </button>
                  <button
                    type="button"
                    className="rounded-md border border-border px-3 py-2 text-sm text-text transition hover:bg-surface2"
                    onClick={handleReset}
                  >
                    {editingConnectionId
                      ? t("common:cancel", "Cancel")
                      : t("common:clear", "Clear")}
                  </button>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="text-xs font-medium text-text">
                  {t("sidepanel:personaGarden.connections.savedConnections", {
                    defaultValue: "Saved connections"
                  })}
                </div>
                {loading ? (
                  <span className="text-xs text-text-muted">
                    {t("common:loading.title", "Loading...")}
                  </span>
                ) : null}
              </div>

              {connections.length > 0 ? (
                connections.map((connection) => (
                  <div
                    key={connection.id}
                    className="rounded-md border border-border bg-bg p-3"
                  >
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div>
                        <div className="font-medium text-text">
                          {connection.name}
                        </div>
                        <div className="text-xs text-text-muted">
                          {connection.base_url}
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-2 text-[11px]">
                        <span className="rounded-full border border-border px-2 py-0.5 text-text-muted">
                          {connection.auth_type}
                        </span>
                        {connection.secret_configured ? (
                          <span
                            data-testid={`persona-connections-secret-configured-${connection.id}`}
                            className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2 py-0.5 text-emerald-700"
                          >
                            secret configured
                          </span>
                        ) : (
                          <span className="rounded-full border border-border px-2 py-0.5 text-text-muted">
                            no secret
                          </span>
                        )}
                      </div>
                    </div>
                    {connection.allowed_hosts?.length ? (
                      <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-text-muted">
                        {connection.allowed_hosts.map((host) => (
                          <span
                            key={`${connection.id}-${host}`}
                            className="rounded-full border border-border px-2 py-0.5"
                          >
                            {host}
                          </span>
                        ))}
                      </div>
                    ) : null}
                    {connection.key_hint ? (
                      <div className="mt-2 text-xs text-text-muted">
                        {t("sidepanel:personaGarden.connections.keyHint", {
                          defaultValue: "Key hint: {{hint}}",
                          hint: connection.key_hint
                        })}
                      </div>
                    ) : null}
                    <div className="mt-3 flex flex-wrap gap-2">
                      <button
                        type="button"
                        ref={(node) => {
                          connectionEditButtonRefs.current[connection.id] = node
                        }}
                        data-testid={`persona-connections-edit-${connection.id}`}
                        className="rounded-md border border-border px-2 py-1 text-xs text-text transition hover:bg-surface2"
                        onClick={() => loadConnectionIntoForm(connection)}
                      >
                        {t("common:edit", "Edit")}
                      </button>
                      <button
                        type="button"
                        data-testid={`persona-connections-test-${connection.id}`}
                        className="rounded-md border border-border px-2 py-1 text-xs text-text transition hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-60"
                        disabled={testingConnectionId === connection.id}
                        onClick={() => {
                          void handleTest(connection.id)
                        }}
                      >
                        {testingConnectionId === connection.id
                          ? t("common:loading.title", "Loading...")
                          : t("sidepanel:personaGarden.connections.test", {
                              defaultValue: "Test"
                            })}
                      </button>
                      <button
                        type="button"
                        data-testid={`persona-connections-delete-${connection.id}`}
                        className="rounded-md border border-red-500/40 px-2 py-1 text-xs text-red-700 transition hover:bg-red-500/10 disabled:cursor-not-allowed disabled:opacity-60"
                        disabled={deletingConnectionId === connection.id}
                        onClick={() => {
                          void handleDelete(connection.id)
                        }}
                      >
                        {deletingConnectionId === connection.id
                          ? t("common:loading.title", "Loading...")
                          : t("common:delete", "Delete")}
                      </button>
                    </div>
                    {testResults[connection.id] ? (
                      <div
                        className={`mt-3 rounded-md border px-3 py-2 text-xs ${
                          testResults[connection.id]?.ok
                            ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-800"
                            : "border-amber-500/40 bg-amber-500/10 text-amber-800"
                        }`}
                      >
                        <div className="font-medium">
                          {testResults[connection.id]?.ok
                            ? `Test passed (${testResults[connection.id]?.status_code ?? "ok"})`
                            : t("sidepanel:personaGarden.connections.testFailed", {
                                defaultValue: "Test failed"
                              })}
                        </div>
                        {summarizeTestBodyPreview(testResults[connection.id]?.body_preview) ? (
                          <div className="mt-1">
                            {summarizeTestBodyPreview(testResults[connection.id]?.body_preview)}
                          </div>
                        ) : null}
                        {testResults[connection.id]?.error ? (
                          <div className="mt-1">{testResults[connection.id]?.error}</div>
                        ) : null}
                        {typeof testResults[connection.id]?.latency_ms === "number" ? (
                          <div className="mt-1 text-[11px] opacity-80">
                            {t("sidepanel:personaGarden.connections.latency", {
                              defaultValue: "Latency: {{latency}} ms",
                              latency: testResults[connection.id]?.latency_ms
                            })}
                          </div>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                ))
              ) : (
                <div
                  data-testid="persona-connections-empty"
                  className="rounded-md border border-dashed border-border px-3 py-4 text-xs text-text-muted"
                >
                  {loading
                    ? t("sidepanel:personaGarden.connections.loading", {
                        defaultValue: "Loading connections..."
                      })
                    : t("sidepanel:personaGarden.connections.empty", {
                        defaultValue:
                          "No reusable connections yet. Create one here, then reference it from a command."
                      })}
                </div>
              )}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  )
}
