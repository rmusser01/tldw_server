import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { Alert, Button, Card, Skeleton, Tag } from 'antd'
import { useTranslation } from 'react-i18next'
import { tldwClient } from '@/services/tldw/TldwApiClient'
import { ProviderIcons } from '@/components/Common/ProviderIcon'
import { getProviderDisplayName } from '@/utils/provider-registry'

type ProviderMap = Record<string, Array<{ id: string, context_length?: number, capabilities?: string[] }>>

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
      for (const item of modelList as any[]) {
        const provider = String(item.provider || 'unknown')
        const id = String(item.id || item.model || item.name)
        const context_length = typeof item.context_length === 'number' ? item.context_length : (typeof item.contextLength === 'number' ? item.contextLength : undefined)
        const capabilities = Array.isArray(item.capabilities) ? item.capabilities : (Array.isArray(item.features) ? item.features : undefined)
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
    const errorMessage = error instanceof Error ? error.message : null
    return (
      <Alert
        type="error"
        showIcon
        title={t('settings:models.loadErrorTitle', 'Unable to load models from server')}
        description={
          <div className="flex flex-col gap-2 text-xs">
            <span>
              {t(
                'settings:models.loadErrorBody',
                'The models endpoint returned an error. Check your server URL and API key, then try again.'
              )}
            </span>
            {errorMessage ? (
              <details>
                <summary className="cursor-pointer font-medium">
                  {t('settings:models.requestDetails', 'Request details')}
                </summary>
                <code className="mt-1 block whitespace-pre-wrap break-words rounded border border-border bg-surface2 px-2 py-1">
                  {errorMessage}
                </code>
              </details>
            ) : null}
            <Button size="small" onClick={() => refetch()} loading={isFetching}>
              {t('common:retry', 'Retry')}
            </Button>
          </div>
        }
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
              <Tag>{(models as any[]).length}</Tag>
            </div>
          }
        >
          <div className="flex flex-col gap-2">
            {(models as any[]).map((m: any) => (
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
