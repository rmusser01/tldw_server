import React, { useCallback, useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Button, Input, Modal, Select, Table, Tag, Tooltip, message } from "antd"
import { Key, Plus, RefreshCw, Trash2 } from "lucide-react"
import { Alert } from "@/components/ui/primitives"
import { tldwClient } from "@/services/tldw/TldwApiClient"

type ProviderKey = {
  provider: string
  has_key: boolean
  source: string
  key_hint: string | null
  auth_source: string | null
  last_used_at: string | null
}

const PROVIDER_OPTIONS = [
  "openai",
  "anthropic",
  "google",
  "groq",
  "deepseek",
  "mistral",
  "cohere",
  "openrouter",
  "huggingface",
  "moonshot",
  "qwen",
  "together",
  "elevenlabs",
  "bedrock",
  "voyage",
  "zai",
  "poe",
  "novita",
  "custom-openai-api",
  "custom-openai-api-2",
]

export const ProviderKeysSettings = () => {
  const { t } = useTranslation(["settings", "common"])
  const [keys, setKeys] = useState<ProviderKey[]>([])
  const [loading, setLoading] = useState(true)
  const [byokUnavailable, setByokUnavailable] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const loadGenerationRef = useRef(0)

  // Add form state
  const [showAddForm, setShowAddForm] = useState(false)
  const [addProvider, setAddProvider] = useState<string>("")
  const [addApiKey, setAddApiKey] = useState("")
  const [saving, setSaving] = useState(false)

  const loadKeys = useCallback(async () => {
    // Refreshes and locale changes can overlap; only the latest load owns state.
    const generation = ++loadGenerationRef.current
    setLoading(true)
    setError(null)
    setByokUnavailable(false)
    try {
      const res = await tldwClient.listUserProviderKeys()
      if (generation !== loadGenerationRef.current) return
      setKeys(res.items ?? [])
      setByokUnavailable(false)
    } catch (err: unknown) {
      if (generation !== loadGenerationRef.current) return
      const failure = err as {
        status?: number
        details?: { detail?: unknown }
      } | null
      if (
        failure?.status === 403 &&
        failure.details?.detail === "BYOK is disabled in this deployment"
      ) {
        setByokUnavailable(true)
      } else {
        setError(
          failure?.status === 403
            ? t(
                "settings:providerKeys.accessDenied",
                "Access to provider keys was denied. Contact your server administrator to check your permissions."
              )
            : t("settings:providerKeys.fetchError", "Failed to load provider keys")
        )
      }
    } finally {
      if (generation === loadGenerationRef.current) setLoading(false)
    }
  }, [t])

  useEffect(() => {
    void loadKeys()
    return () => {
      loadGenerationRef.current += 1
    }
  }, [loadKeys])

  const handleSave = useCallback(async () => {
    if (!addProvider || !addApiKey) return
    setSaving(true)
    try {
      await tldwClient.upsertUserProviderKey(addProvider, addApiKey)
      message.success(t("settings:providerKeys.saveSuccess", { provider: addProvider, defaultValue: `${addProvider} key saved and validated` }))
      setShowAddForm(false)
      setAddProvider("")
      setAddApiKey("")
      void loadKeys()
    } catch {
      message.error(t("settings:providerKeys.saveError", "Failed to save key — check that the key is valid"))
    } finally {
      setSaving(false)
    }
  }, [addProvider, addApiKey, loadKeys, t])

  const handleDelete = useCallback(
    (provider: string) => {
      Modal.confirm({
        title: t("settings:providerKeys.deleteConfirmTitle", { provider, defaultValue: `Remove ${provider} key?` }),
        content: t("settings:providerKeys.deleteConfirmContent", "Your key will be removed and the server default key (if any) will be used instead."),
        okText: t("common:delete", "Delete"),
        okType: "danger",
        cancelText: t("common:cancel", "Cancel"),
        onOk: async () => {
          try {
            await tldwClient.deleteUserProviderKey(provider)
            message.success(t("settings:providerKeys.deleteSuccess", { provider, defaultValue: `${provider} key removed` }))
            void loadKeys()
          } catch {
            message.error(t("settings:providerKeys.deleteError", "Failed to remove key"))
          }
        },
      })
    },
    [loadKeys, t]
  )

  if (byokUnavailable) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-6">
        <Alert
          variant="info"
          title={t("settings:providerKeys.unavailableTitle", "Provider key management is not available")}
        >
          {t("settings:providerKeys.unavailableDesc", "Set BYOK_ENCRYPTION_KEY in your server's .env file to enable user-managed provider keys. Docker users: this is auto-generated on first run.")}
        </Alert>
      </div>
    )
  }

  const sourceTag = (source: string) => {
    switch (source) {
      case "user":
        return <Tag color="blue">{t("settings:providerKeys.sourceUser", "User key")}</Tag>
      case "server_default":
        return <Tag>{t("settings:providerKeys.sourceServer", "Server default")}</Tag>
      case "team":
      case "org":
        return <Tag color="purple">{t("settings:providerKeys.sourceShared", "Shared key")}</Tag>
      default:
        return <Tag color="default">{t("settings:providerKeys.sourceNone", "Not configured")}</Tag>
    }
  }

  const columns = [
    {
      title: t("settings:providerKeys.colProvider", "Provider"),
      dataIndex: "provider",
      key: "provider",
      render: (provider: string) => (
        <span className="font-medium capitalize">{provider}</span>
      ),
    },
    {
      title: t("settings:providerKeys.colSource", "Source"),
      key: "source",
      render: (_: unknown, record: ProviderKey) => sourceTag(record.source),
    },
    {
      title: t("settings:providerKeys.colKeyHint", "Key Hint"),
      dataIndex: "key_hint",
      key: "hint",
      render: (hint: string | null) =>
        hint ? (
          <code className="text-xs text-text-muted">...{hint}</code>
        ) : (
          <span className="text-text-subtle">-</span>
        ),
    },
    {
      title: "",
      key: "actions",
      render: (_: unknown, record: ProviderKey) =>
        record.has_key && record.source === "user" ? (
          <Tooltip title={t("settings:providerKeys.deleteTooltip", "Remove your key (falls back to server default)")}>
            <Button
              type="text"
              danger
              size="small"
              icon={<Trash2 className="h-3.5 w-3.5" />}
              onClick={() => handleDelete(record.provider)}
            />
          </Tooltip>
        ) : null,
    },
  ]

  return (
    <div className="mx-auto max-w-3xl px-4 py-6">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-text">
          {t("settings:providerKeys.title", "LLM Provider Keys")}
        </h2>
        <p className="mt-1 text-sm text-text-muted">
          {t("settings:providerKeys.description", "Manage API keys for LLM providers. Server-configured keys from .env are used as defaults. Add your own key to override.")}
        </p>
      </div>

      {error && (
        <Alert
          variant="error"
          title={error}
          className="mb-4"
          dismissible
          onDismiss={() => setError(null)}
        />
      )}

      <div className="mb-4 flex items-center justify-between">
        <Button
          icon={<RefreshCw className="h-4 w-4" />}
          onClick={() => void loadKeys()}
          loading={loading}
          size="small"
        >
          {t("common:refresh", "Refresh")}
        </Button>
        <Button
          type="primary"
          icon={<Plus className="h-4 w-4" />}
          onClick={() => setShowAddForm(true)}
          disabled={showAddForm}
        >
          {t("settings:providerKeys.addButton", "Add Provider Key")}
        </Button>
      </div>

      {showAddForm && (
        <div className="mb-6 rounded-xl border border-border bg-surface/80 p-5">
          <h3 className="mb-4 text-sm font-semibold text-text">
            {t("settings:providerKeys.addButton", "Add Provider Key")}
          </h3>
          <div className="space-y-4">
            <div>
              <label className="mb-1 block text-xs font-medium text-text-muted">
                {t("settings:providerKeys.colProvider", "Provider")}
              </label>
              <Select
                value={addProvider || undefined}
                onChange={(v) => setAddProvider(v)}
                placeholder={t("settings:providerKeys.selectProvider", "Select provider...")}
                className="w-full"
                options={PROVIDER_OPTIONS.map((p) => ({
                  label: <span className="capitalize">{p}</span>,
                  value: p,
                }))}
              />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-text-muted">
                {t("settings:providerKeys.apiKeyLabel", "API Key")}
              </label>
              <Input.Password
                value={addApiKey}
                onChange={(e) => setAddApiKey(e.target.value)}
                placeholder="sk-..."
                autoComplete="new-password"
              />
            </div>

            <div className="flex items-center justify-end gap-2">
              <Button onClick={() => { setShowAddForm(false); setAddProvider(""); setAddApiKey("") }}>
                {t("common:cancel", "Cancel")}
              </Button>
              <Button
                type="primary"
                onClick={handleSave}
                disabled={!addProvider || !addApiKey}
                loading={saving}
                icon={<Key className="h-4 w-4" />}
              >
                {t("settings:providerKeys.saveButton", "Save Key")}
              </Button>
            </div>
          </div>
        </div>
      )}

      <Table
        dataSource={keys}
        columns={columns}
        rowKey="provider"
        loading={loading}
        pagination={false}
        size="small"
        locale={{
          emptyText: loading
            ? t("common:loading.title", "Loading...")
            : t("settings:providerKeys.emptyTable", "No provider keys configured. Add a key or configure providers in your server's .env file."),
        }}
      />
    </div>
  )
}

export default ProviderKeysSettings
