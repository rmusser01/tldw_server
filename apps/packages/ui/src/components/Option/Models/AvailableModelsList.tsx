import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { Button, Card, Skeleton, Tag } from 'antd'
import { useTranslation } from 'react-i18next'
import { tldwClient } from '@/services/tldw/TldwApiClient'
import { ProviderIcons } from '@/components/Common/ProviderIcon'
import { getProviderDisplayName } from '@/utils/provider-registry'
import { RecoveryCallout, buildCapabilityState } from '@/components/ui/state'

type ProviderModel = {
  id: string
  context_length?: number
  capabilities?: string[]
}

type ProviderMap = Record<string, ProviderModel[]>

const MODELS_METADATA_PATH = '/api/v1/llm/models/metadata'
const SECRET_VALUE_PATTERN =
  /\b(api[_-]?key|authorization|bearer|token|secret|password)(\s*[:=]\s*)([^\s,;]+)/gi
const BEARER_VALUE_PATTERN = /\b(Bearer\s+)([A-Za-z0-9._~+/-]+)/g

const isAbortLikeError = (error: unknown): boolean => {
  if (!error || typeof error !== "object") return false
  const candidate = error as {
    name?: unknown
    message?: unknown
    code?: unknown
  }
  const name = String(candidate.name || "")
  const message = String(candidate.message || "")
  const code = String(candidate.code || "")
  return (
    name === "AbortError" ||
    code === "REQUEST_ABORTED" ||
    /abort/i.test(message)
  )
}

const redactDiagnosticMessage = (value: string): string =>
  value
    .replace(BEARER_VALUE_PATTERN, '$1[redacted]')
    .replace(SECRET_VALUE_PATTERN, '$1$2[redacted]')

const getModelLoadErrorMessage = (error: unknown): string | null => {
  if (error instanceof Error && error.message.trim()) {
    return redactDiagnosticMessage(error.message)
  }
  if (typeof error === "string" && error.trim()) {
    return redactDiagnosticMessage(error)
  }
  if (error && typeof error === "object" && "message" in error) {
    const message = (error as { message?: unknown }).message
    if (typeof message === "string" && message.trim()) {
      return redactDiagnosticMessage(message)
    }
  }
  return null
}

const getModelLoadErrorStatus = (error: unknown): number | null => {
  if (!error || typeof error !== 'object') return null
  const candidate = error as {
    status?: unknown
    statusCode?: unknown
    response?: { status?: unknown }
  }
  const status = candidate.status ?? candidate.statusCode ?? candidate.response?.status
  return typeof status === 'number' && Number.isFinite(status) ? status : null
}

export const AvailableModelsList: React.FC = () => {
  const { t } = useTranslation(['settings', 'common'])
  const { data, status, error, refetch, isFetching } = useQuery({
    queryKey: ['tldw-providers-models'],
    queryFn: async () => {
      await tldwClient.initialize()
      let modelList: unknown[] | null = null
      try {
        // Accept either the legacy flat array or the current { models, total } envelope.
        const meta = await tldwClient.getModelsMetadata()
        modelList = Array.isArray(meta)
          ? meta
          : meta && typeof meta === "object" && Array.isArray((meta as { models?: unknown[] }).models)
            ? (meta as { models: unknown[] }).models
            : null
      } catch (requestError) {
        if (isAbortLikeError(requestError)) {
          return {}
        }
        throw requestError
      }
      if (!Array.isArray(modelList)) {
        throw new Error("Unexpected models metadata response")
      }
      const normalized: ProviderMap = {}
      for (const item of modelList) {
        const record =
          item && typeof item === 'object' ? (item as Record<string, unknown>) : {}
        const provider = String(record.provider || 'unknown')
        const id = String(record.id || record.model || record.name)
        const context_length =
          typeof record.context_length === 'number'
            ? record.context_length
            : typeof record.contextLength === 'number'
              ? record.contextLength
              : undefined
        const capabilities = Array.isArray(record.capabilities)
          ? record.capabilities.map(String)
          : Array.isArray(record.features)
            ? record.features.map(String)
            : undefined
        if (!normalized[provider]) normalized[provider] = []
        // Avoid duplicates
        if (!normalized[provider].some((m) => m.id === id)) {
          normalized[provider].push({ id, context_length, capabilities })
        }
      }
      // Sort each provider list and providers alphabetically
      for (const p of Object.keys(normalized)) {
        normalized[p] = normalized[p].sort((a, b) => a.id.localeCompare(b.id))
      }
      return normalized
    }
  })

  if (status === 'pending' && !data) {
    return <Skeleton paragraph={{ rows: 6 }} />
  }

  if (status === 'error') {
    const errorMessage = getModelLoadErrorMessage(error)
    const recoveryState = buildCapabilityState({
      featureName: 'Models',
      capabilityName: 'model metadata catalog',
      endpoint: MODELS_METADATA_PATH,
      method: 'GET',
      status: getModelLoadErrorStatus(error),
      rawMessage: errorMessage ?? 'Model metadata request failed',
      title: t('settings:models.loadErrorTitle', 'Unable to load models from server'),
      message: t(
        'settings:models.loadErrorBody',
        'The models endpoint returned an error. Check your server URL and API key, then try again.'
      )
    })
    return (
      <RecoveryCallout
        state={recoveryState.state}
        title={t('settings:models.loadErrorTitle', 'Unable to load models from server')}
        message={recoveryState.message}
        diagnostics={recoveryState.diagnostics}
        primaryAction={{
          label: t('common:retry', 'Retry'),
          onClick: () => {
            void refetch()
          },
          loading: isFetching
        }}
        data-testid="models-catalog-load-recovery"
      />
    )
  }

  const isEmpty = !data || Object.keys(data || {}).length === 0

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {Object.entries(data || {}).map(([provider, models]) => (
        <Card
          key={provider}
          title={
            <div className="flex items-center gap-2">
              <ProviderIcons provider={provider} className="h-4 w-4" />
              <span>{getProviderDisplayName(provider)}</span>
              <Tag>{models.length}</Tag>
            </div>
          }
        >
          <div className="flex flex-col gap-2">
            {models.map((m) => (
              <div key={m.id} className="flex items-center gap-2 text-xs flex-wrap">
                <Tag bordered>{m.id}</Tag>
                {typeof m.context_length === 'number' && (
                  <Tag color="blue" bordered>ctx {m.context_length}</Tag>
                )}
                {Array.isArray(m.capabilities) && (
                  <>
                    {m.capabilities.slice(0, 4).map((c: string) => (
                      <Tag key={c} color="green" bordered>{c}</Tag>
                    ))}
                    {m.capabilities.length > 4 && (
                      <Tag color="default" bordered>+{m.capabilities.length - 4}</Tag>
                    )}
                  </>
                )}
              </div>
            ))}
          </div>
        </Card>
      ))}
      {isEmpty && (
        <div className="text-sm text-text-muted">
          <div className="mb-1 font-medium">
            {t('settings:models.noProvidersTitle', 'No providers available.')}
          </div>
          <div className="text-xs">
            {t(
              'settings:models.noProvidersBody',
              'The extension could not load providers from your tldw_server. Check your server URL and API key in Settings, ensure the server is running, then use Retry (or Refresh) to try again.'
            )}
          </div>
          <Button
            size="small"
            className="mt-2"
            onClick={() => refetch()}
            loading={isFetching}>
            {t('common:retry', 'Retry')}
          </Button>
        </div>
      )}
    </div>
  )}

export default AvailableModelsList
