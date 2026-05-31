import { useEffect, useRef, useState } from 'react'
import { Tag, Card, Space, Typography, Button, Tooltip, InputNumber } from 'antd'
import { browser } from 'wxt/browser'
import { Link, useNavigate } from 'react-router-dom'
import { tldwClient } from '@/services/tldw/TldwApiClient'
import { apiSend } from '@/services/api-send'
import type { AllowedPath } from '@/services/tldw/openapi-guard'
import { useTranslation } from 'react-i18next'
import type { TFunction } from 'i18next'
import { useAntdNotification } from '@/hooks/useAntdNotification'
import { getReturnTo, clearReturnTo } from "@/utils/return-to"
import { ServerOverviewHint } from "@/components/Common/ServerOverviewHint"
import {
  RecoveryCallout,
  SetupRequiredPanel,
  StatePanel
} from "@/components/ui/state"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import { LoadingState } from "@/components/ui/feedback"
import {
  useConnectionState,
  useConnectionUxState
} from "@/hooks/useConnectionState"
import { cleanUrl } from "@/libs/clean-url"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { getDesignSystemState } from "@/design-system"

type Check = {
  key: string
  label: string
  path: AllowedPath
  descriptionKey: string
  descriptionDefault: string
  docsUrlKey?: string
  docsUrlDefault?: string
}

const makeChecks = (t: TFunction): Check[] => {
  const base = 'settings:healthPage.checks'
  return [
    {
      key: 'core',
      label: t(`${base}.core`, 'Core API'),
      path: '/api/v1/health',
      descriptionKey: `${base}.coreDescription`,
      descriptionDefault:
        'Verifies the main chat API. If this is unhealthy, confirm your server URL, API key, and core logs.',
      docsUrlKey: `${base}.coreDocsUrl`,
      docsUrlDefault: 'https://github.com/rmusser01/tldw_browser_assistant'
    },
    {
      key: 'rag',
      label: t(`${base}.rag`, 'RAG'),
      path: '/api/v1/rag/health',
      descriptionKey: `${base}.ragDescription`,
      descriptionDefault:
        'Powers Knowledge search & retrieval. If this is unhealthy, ensure RAG is enabled and the knowledge index has been built on your server.',
      docsUrlKey: `${base}.ragDocsUrl`,
      docsUrlDefault: 'https://github.com/rmusser01/tldw_browser_assistant'
    },
    {
      key: 'audio',
      label: t(`${base}.audio`, 'Audio'),
      path: '/api/v1/audio/health',
      descriptionKey: `${base}.audioDescription`,
      descriptionDefault:
        'Covers text-to-speech and speech-to-text APIs. If this is unhealthy, confirm your audio endpoints are enabled and reachable.',
      docsUrlKey: `${base}.audioDocsUrl`,
      docsUrlDefault: 'https://github.com/rmusser01/tldw_browser_assistant'
    },
    {
      key: 'embeddings',
      label: t(`${base}.embeddings`, 'Embeddings'),
      path: '/api/v1/embeddings/health',
      descriptionKey: `${base}.embeddingsDescription`,
      descriptionDefault:
        'Checks the embeddings and indexing services used for retrieval. If this is unhealthy, rebuild your embedding index or restart the worker.',
      docsUrlKey: `${base}.embeddingsDocsUrl`,
      docsUrlDefault: 'https://github.com/rmusser01/tldw_browser_assistant'
    },
    {
      key: 'metrics',
      label: t(`${base}.metrics`, 'Metrics Health'),
      path: '/api/v1/metrics/health',
      descriptionKey: `${base}.metricsDescription`,
      descriptionDefault:
        'Verifies metrics and monitoring endpoints. If this is unhealthy, metrics dashboards or alerting may be unavailable.',
      docsUrlKey: `${base}.metricsDocsUrl`,
      docsUrlDefault: 'https://github.com/rmusser01/tldw_browser_assistant'
    },
    {
      key: 'chatMetrics',
      label: t(`${base}.chatMetrics`, 'Chat Metrics'),
      path: '/api/v1/metrics/chat',
      descriptionKey: `${base}.chatMetricsDescription`,
      descriptionDefault:
        'Tracks chat analytics such as volume and latency. If this is unhealthy, chat metrics collection or endpoints may be failing.',
      docsUrlKey: `${base}.chatMetricsDocsUrl`,
      docsUrlDefault: 'https://github.com/rmusser01/tldw_browser_assistant'
    },
    {
      key: 'mcp',
      label: t(`${base}.mcp`, 'MCP'),
      path: '/api/v1/mcp/health',
      descriptionKey: `${base}.mcpDescription`,
      descriptionDefault:
        'Checks Model Context Protocol (MCP) tools. If this is unhealthy, external tools and plugins may not be available in chat.',
      docsUrlKey: `${base}.mcpDocsUrl`,
      docsUrlDefault: 'https://github.com/rmusser01/tldw_browser_assistant'
    }
  ]
}

const READY_STATE_LABEL = getDesignSystemState("ready")?.label
const DEGRADED_STATE_LABEL = getDesignSystemState("degraded")?.label
const LOADING_STATE_LABEL = getDesignSystemState("loading")?.label

type Result = { status: 'unknown'|'healthy'|'unhealthy', detail?: any, statusCode?: number, durationMs?: number }

export default function HealthStatus() {
  const { t } = useTranslation(['settings', 'common', 'option'])
  const notification = useAntdNotification()
  const checks = makeChecks(t)
  const [results, setResults] = useState<Record<string, Result>>({})
  const [loading, setLoading] = useState(false)
  const [runningChecks, setRunningChecks] = useState<Set<string>>(new Set())
  const [serverUrl, setServerUrl] = useState<string>('')
  const [coreStatus, setCoreStatus] = useState<'unknown'|'connected'|'failed'>('unknown')
  const [autoRefresh, setAutoRefresh] = useState<boolean>(false)
  const [intervalSec, setIntervalSec] = useState<number>(30)
  const [recentHealthy, setRecentHealthy] = useState<Set<string>>(new Set())
  const [lastUpdatedAt, setLastUpdatedAt] = useState<number | null>(null)
  const [secondsSinceUpdate, setSecondsSinceUpdate] = useState<number | null>(null)
  const [queueStatus, setQueueStatus] = useState<any | null>(null)
  const [queueActivity, setQueueActivity] = useState<any | null>(null)
  const [queueLoading, setQueueLoading] = useState(false)
  const [queueError, setQueueError] = useState<string | null>(null)
  const isRunningRef = useRef(false)
  const initialLoadRef = useRef(false)
  const navigate = useNavigate()
  const MIN_INTERVAL_SEC = 5
  const SAFE_FLOOR_SEC = 15
  const {
    serverUrl: storeServerUrl,
    lastStatusCode,
    lastError
  } = useConnectionState()
  const { uxState, errorKind } = useConnectionUxState()
  const { capabilities } = useServerCapabilities()
  const storeHost = storeServerUrl ? cleanUrl(storeServerUrl) : null

  const runSingle = async (c: Check): Promise<boolean> => {
    setRunningChecks(prev => new Set(prev).add(c.key))
    const t0 = performance.now()
    try {
      const resp = await apiSend({ path: c.path, method: 'GET' })
      const t1 = performance.now()
      const statusCode =
        typeof resp?.status === 'number' && !Number.isNaN(resp.status)
          ? resp.status
          : undefined
      const detail = resp?.ok ? resp?.data : resp?.error || resp?.data
      setResults((prev) => ({
        ...prev,
        [c.key]: {
          status: resp?.ok ? 'healthy' : 'unhealthy',
          detail,
          statusCode,
          durationMs: Math.round(t1 - t0)
        }
      }))
      return !!resp?.ok
    } catch (e) {
      const t1 = performance.now()
      const message = (e as any)?.message || 'Network error'
      setResults((prev) => ({
        ...prev,
        [c.key]: {
          status: 'unhealthy',
          detail: message,
          statusCode: 0,
          durationMs: Math.round(t1 - t0)
        }
      }))
      return false
    } finally {
      setRunningChecks(prev => {
        const next = new Set(prev)
        next.delete(c.key)
        return next
      })
    }
  }

  const runChecks = async (userTriggered: boolean = false) => {
    if (isRunningRef.current) return
    isRunningRef.current = true
    setLoading(true)
    let allHealthy = true
    try {
      for (const c of checks) {
        // eslint-disable-next-line no-await-in-loop
        const prev = results[c.key]?.status
        const ok = await runSingle(c)
        if (userTriggered && ok && prev !== 'healthy') {
          // Mark as recently turned healthy for a subtle pulse
          setRecentHealthy(prevSet => {
            const next = new Set(prevSet)
            next.add(c.key)
            return next
          })
          setTimeout(() => {
            setRecentHealthy(prevSet => {
              const next = new Set(prevSet)
              next.delete(c.key)
              return next
            })
          }, 1200)
        }
        if (!ok) allHealthy = false
      }
      setLastUpdatedAt(Date.now())
      if (userTriggered && allHealthy) {
        notification.success({
          message: t('settings:tldw.connection.success', 'Server responded successfully. You can continue.'),
          placement: 'bottomRight',
          duration: 2
        })
      }
    } finally {
      setLoading(false)
      isRunningRef.current = false
    }
  }

  const recheckOne = async (c: Check) => {
    if (runningChecks.has(c.key)) return
    const prev = results[c.key]?.status
    const ok = await runSingle(c)
    // Show success feedback even if check was already healthy
    if (ok) {
      if (prev !== 'healthy') {
        setRecentHealthy(prevSet => {
          const next = new Set(prevSet)
          next.add(c.key)
          return next
        })
        setTimeout(() => {
          setRecentHealthy(prevSet => {
            const next = new Set(prevSet)
            next.delete(c.key)
            return next
          })
        }, 1200)
      }
      notification.success({
        message: t('settings:healthPage.recheckSuccess', 'Check passed'),
        placement: 'bottomRight',
        duration: 2
      })
    } else {
      notification.warning({
        message: t('settings:healthPage.recheckFailed', 'Check failed - see details below'),
        placement: 'bottomRight',
        duration: 3
      })
    }
  }

  const testCoreConnection = async () => {
    try {
      await tldwClient.initialize()
      const ok = await tldwClient.healthCheck()
      setCoreStatus(ok ? 'connected' : 'failed')
    } catch {
      setCoreStatus('failed')
    }
  }

  const loadQueueDiagnostics = async () => {
    if (!capabilities?.hasChatQueue) return
    setQueueLoading(true)
    setQueueError(null)
    try {
      await tldwClient.initialize().catch(() => null)
      const [status, activity] = await Promise.all([
        tldwClient.chatQueueStatus(),
        tldwClient.chatQueueActivity(25)
      ])
      setQueueStatus(status)
      setQueueActivity(activity)
    } catch (e: any) {
      setQueueError(e?.message || "Unable to load queue diagnostics.")
    } finally {
      setQueueLoading(false)
    }
  }

  useEffect(() => {
    if (initialLoadRef.current) return
    initialLoadRef.current = true
    ;(async () => {
      try {
        const cfg = await tldwClient.getConfig()
        setServerUrl(cfg?.serverUrl || '')
      } catch {}
      await testCoreConnection()
      await runChecks()
    })()
  }, [runChecks, testCoreConnection])

  useEffect(() => {
    if (!autoRefresh) return
    const id = setInterval(() => {
      void runChecks()
    }, Math.max(MIN_INTERVAL_SEC, intervalSec) * 1000)
    return () => clearInterval(id)
  }, [autoRefresh, intervalSec])

  useEffect(() => {
    if (!lastUpdatedAt) {
      setSecondsSinceUpdate(null)
      return
    }
    const update = () => {
      const diff = Math.max(0, Math.floor((Date.now() - lastUpdatedAt) / 1000))
      setSecondsSinceUpdate(diff)
    }
    update()
    const id = setInterval(update, 1000)
    return () => clearInterval(id)
  }, [lastUpdatedAt])

  useEffect(() => {
    if (capabilities?.hasChatQueue) {
      void loadQueueDiagnostics()
    }
  }, [capabilities?.hasChatQueue])

  const intervalSecClamped = Math.max(MIN_INTERVAL_SEC, intervalSec)
  const secondsUntilNext =
    autoRefresh && secondsSinceUpdate != null
      ? Math.max(0, intervalSecClamped - secondsSinceUpdate)
      : null
  const showIntervalWarning =
    autoRefresh && intervalSecClamped < SAFE_FLOOR_SEC

  const describeStatus = (status: Result['status']): string => {
    if (status === 'healthy') {
      return t('healthPage.statusReady', READY_STATE_LABEL)
    }
    if (status === 'unhealthy') {
      return t('healthPage.statusDegraded', DEGRADED_STATE_LABEL)
    }
    return t('healthPage.statusLoading', LOADING_STATE_LABEL)
  }

  const showAuthCallout =
    uxState === "error_auth" || errorKind === "auth"
  const showUnreachableCallout =
    uxState === "error_unreachable" || errorKind === "unreachable"
  const showDegradedCallout = uxState === "connected_degraded"

  // Calculate summary stats based on current checks
  const totalChecks = checks.length
  const healthyCount = checks.filter(c => results[c.key]?.status === 'healthy').length
  const unhealthyCount = checks.filter(c => results[c.key]?.status === 'unhealthy').length
  const unknownCount = totalChecks - healthyCount - unhealthyCount
  const allChecked = unknownCount === 0
  const allHealthy = allChecked && unhealthyCount === 0

  return (
    <Space orientation="vertical" size="large" className="w-full">
      {/* Summary banner */}
      {allChecked && (
        <StatePanel
          state={allHealthy ? "ready" : "degraded"}
          title={
            allHealthy
              ? t('settings:healthPage.summaryAllHealthy', {
                  defaultValue: 'All {{count}} checks passing',
                  count: totalChecks
                })
              : t('settings:healthPage.summaryPartial', {
                  defaultValue: '{{healthy}} of {{total}} checks passing',
                  healthy: healthyCount,
                  total: totalChecks
                })
          }
          message={
            !allHealthy
              ? t('settings:healthPage.summaryUnhealthyHint', {
                  defaultValue: 'Review the failing checks below for troubleshooting guidance.'
                })
              : undefined
          }
          primaryAction={{
            label: t('healthPage.recheckAll', 'Recheck All'),
            onClick: () => runChecks(true),
            loading
          }}
        />
      )}

      <div className="flex items-center justify-between">
        <div>
          <Typography.Title level={4} className="!mb-0">
            {t(
              "healthPage.title",
              "Health & diagnostics"
            )}
          </Typography.Title>
          <Typography.Paragraph type="secondary" className="!mb-0">
            {t(
              "healthPage.subtitle",
              "Quick overview of how your tldw server powers chat, Knowledge search, media ingest, and other tools."
            )}
          </Typography.Paragraph>
        </div>
        <Space>
          <Button
            type="primary"
            onClick={() => runChecks(true)}
            loading={loading}
          >
            {t('healthPage.recheckAll', 'Recheck All')}
          </Button>
          <Tooltip title={t('healthPage.copyDiagnosticsHelp', 'Copies JSON diagnostics to clipboard') as string}>
            <Button
              onClick={async () => {
                try {
                  const payload = {
                    serverUrl,
                    coreStatus,
                    timestamp: new Date().toISOString(),
                    results
                  }
                  const text = JSON.stringify(payload, null, 2)
                  await navigator.clipboard.writeText(text)
                  notification.success({
                    message: t(
                      'healthPage.copiedDiagnostics',
                      'Diagnostics copied'
                    ),
                    placement: 'bottomRight',
                    duration: 2
                  })
                } catch {
                  notification.error({
                    message: t(
                      'healthPage.copyFailed',
                      'Failed to copy diagnostics'
                    ),
                    placement: 'bottomRight',
                    duration: 2
                  })
                }
              }}>
              {t('healthPage.copyDiagnostics', 'Copy diagnostics')}
            </Button>
          </Tooltip>
          <Button
            onClick={() => {
              const target = getReturnTo()
              if (target) {
                clearReturnTo()
                navigate(target)
              } else {
                navigate(-1)
              }
            }}>
            ← {t('healthPage.backToChat', 'Back to chat')}
          </Button>
          <Link to="/settings/tldw">
            <Button>{t('healthPage.openSettings', 'Open tldw Settings')}</Button>
          </Link>
        </Space>
      </div>

      {(showAuthCallout || showUnreachableCallout || showDegradedCallout) && (
        <RecoveryCallout
          state={
            showAuthCallout
              ? "auth_required"
              : showUnreachableCallout
                ? "unavailable"
                : "degraded"
          }
          className="mt-3"
          title={
            showAuthCallout
              ? t(
                  "option:connectionCard.headlineErrorAuth",
                  "API key needs attention"
                )
              : showUnreachableCallout
                ? t(
                    "option:connectionCard.headlineError",
                    "Can’t reach your tldw server"
                  )
                : t(
                    "healthPage.degradedTitle",
                    "Chat is ready — some tools are offline"
                  )
          }
          message={
            <span className="space-y-1 text-sm">
              <span className="block">
                {showAuthCallout
                  ? t(
                      "healthSummary.issueAuthHint",
                      "Your server responded but the API key or login is invalid. Fix your credentials, then re-run checks."
                    )
                  : showUnreachableCallout
                    ? t(
                        "healthSummary.issueConnectivityHint",
                        "We couldn’t reach your tldw server. Check that it’s running, your browser has site access, and any proxies or firewalls allow the connection."
                      )
                    : t(
                        "healthPage.degradedBody",
                        "Core chat is connected, but some health checks are failing. You can continue using the assistant while you investigate."
                  )}
              </span>
              {(typeof lastStatusCode === "number" && lastStatusCode > 0) ||
              lastError ? (
                <span className="block text-[11px] text-text-muted">
                  {t(
                    "healthPage.lastErrorSummary",
                    "Most recent connection error: {{code}} {{message}}",
                    {
                      code:
                        typeof lastStatusCode === "number" &&
                        lastStatusCode > 0
                          ? `HTTP ${lastStatusCode}`
                          : t(
                              "healthPage.lastErrorNetwork",
                              "network/timeout"
                            ),
                      message: lastError || ""
                    }
                  )}
                </span>
              ) : null}
            </span>
          }
          primaryAction={{
            label: showAuthCallout
              ? t("healthPage.fixApiKeyCta", "Fix API key")
              : showUnreachableCallout
                ? t("healthPage.editUrlCta", "Edit server URL")
                : t("healthPage.backToAppCta", "Back to app"),
            onClick: () => {
              if (showAuthCallout || showUnreachableCallout) {
                navigate("/")
                return
              }
              const target = getReturnTo()
              if (target) {
                clearReturnTo()
                navigate(target)
              } else {
                navigate(-1)
              }
            }
          }}
        />
      )}

      {!serverUrl && (
        <DesignSystemAlert
          variant="info"
          className="mt-2"
          title={t(
            'healthPage.noServerBannerTitle',
            'Don’t have a server yet?'
          )}
        >
          {t(
            'healthPage.noServerBannerBody',
            'You can explore the UI first, then connect a tldw server later to enable chat history, media ingest, and Knowledge search.'
          )}
        </DesignSystemAlert>
      )}

      {!serverUrl || coreStatus === 'failed' ? (
        !serverUrl ? (
          <SetupRequiredPanel
            title={t('healthPage.serverNotConfigured', 'Server is not configured.')}
            message={t('healthPage.configureHint', 'Please configure a server URL under tldw settings.')}
            primaryAction={{
              label: t('healthPage.configureCta', 'Configure'),
              onClick: () => navigate("/settings/tldw")
            }}
          />
        ) : (
          <RecoveryCallout
            state="unavailable"
            title={t('healthPage.unableToReachCore', 'Unable to reach server core health endpoint.')}
            message={t('healthPage.triedGet', 'Tried GET {{url}}', { url: `${serverUrl.replace(/\/$/, '')}/api/v1/health` })}
            primaryAction={{
              label: t('healthPage.configureCta', 'Configure'),
              onClick: () => navigate("/settings/tldw")
            }}
          />
        )
      ) : (
        <DesignSystemAlert
          variant="success"
          title={t('healthPage.connectedTo', 'Connected to {{host}}', { host: serverUrl })}
        />
      )}

      {(!serverUrl || coreStatus === 'failed') && (
        <ServerOverviewHint />
      )}

      {autoRefresh && showIntervalWarning && (
        <DesignSystemAlert
          variant="warning"
          title={t(
            'healthPage.autoRefreshWarningTitle',
            'Auto-refresh is enabled with a short interval'
          )}
          className="mb-4"
        >
          {t(
            'healthPage.intervalWarning',
            'Short intervals can put load on your server. Consider using at least {{seconds}}s.',
            { seconds: SAFE_FLOOR_SEC }
          )}
        </DesignSystemAlert>
      )}

      <div className="flex items-center gap-4">
        <label className="text-sm flex items-center gap-2">
          <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} /> {t('healthPage.autoRefresh', 'Auto-refresh')}
        </label>
        <label className="text-sm flex items-center gap-2">
          {t('healthPage.intervalLabel', 'Interval:')}
          <InputNumber
            min={MIN_INTERVAL_SEC}
            className="w-32"
            value={intervalSec}
            onChange={(value) => {
              const next = typeof value === 'number' ? Math.max(MIN_INTERVAL_SEC, value) : MIN_INTERVAL_SEC
              setIntervalSec(next)
            }}
            suffix={t('healthPage.intervalSuffix', 'seconds')}
          />
        </label>
        {secondsSinceUpdate != null && (
          <span className="text-xs text-text-subtle">
            {secondsSinceUpdate <= 5
              ? t('healthPage.updatedJustNow', 'Updated just now')
              : t('healthPage.updatedSecondsAgo', 'Updated {{seconds}}s ago', { seconds: secondsSinceUpdate })}
          </span>
        )}
        {autoRefresh && secondsUntilNext != null && (
          <span className="text-xs text-text-subtle">
            {t('healthPage.nextRefreshIn', 'Next auto-refresh in {{seconds}}s', {
              seconds: secondsUntilNext
            })}
          </span>
        )}
      </div>

      <div aria-live="polite" aria-atomic="false">
        <div className="mt-4">
          <Typography.Title
            level={5}
            className="!mb-1 text-sm md:text-base"
          >
            {t(
              "healthPage.technicalSummaryTitle",
              "Technical details"
            )}
          </Typography.Title>
          <Typography.Paragraph
            type="secondary"
            className="!mb-2 text-xs md:text-sm"
          >
            {t(
              "healthPage.technicalSummaryBody",
              "Each check below hits a specific health endpoint and shows the raw JSON response so you can debug server-side issues."
            )}
          </Typography.Paragraph>
          {storeHost && (
            <Typography.Text className="text-[11px] text-text-subtle">
              {t(
                "healthPage.technicalSummaryHost",
                "Current server: {{host}}",
                { host: storeHost }
              )}
            </Typography.Text>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {checks.map(c => {
            const r = results[c.key] || { status: 'unknown' }
            const statusLabel = describeStatus(r.status)
            const description = t(c.descriptionKey, c.descriptionDefault)
            const docsUrl = c.docsUrlDefault
              ? t(c.docsUrlKey ?? '', c.docsUrlDefault)
              : ''
            const isUnhealthy = r.status === 'unhealthy'
            const detailText = (() => {
              if (!r.detail) return ""
              try {
                return JSON.stringify(r.detail, null, 2)
              } catch {
                return String(r.detail)
              }
            })()
            const MAX_DETAIL_CHARS = 5000
            const displayedDetail =
              detailText.length > MAX_DETAIL_CHARS
                ? `${detailText.slice(0, MAX_DETAIL_CHARS)}…`
                : detailText
            const isCheckRunning = runningChecks.has(c.key)
            return (
              <Card
                key={c.key}
                title={c.label}
                extra={
                  isCheckRunning ? (
                    <Space size="small">
                      <LoadingState
                        mode="inline"
                        size="sm"
                        data-testid={`health-check-${c.key}-loading`}
                      />
                      <span className="text-text-subtle">{t('healthPage.checking', 'Checking…')}</span>
                    </Space>
                  ) : (
                    <a onClick={() => recheckOne(c)}>{t('healthPage.recheck', 'Recheck')}</a>
                  )
                }
                role="group"
                aria-label={`${c.label}: ${statusLabel}`}
              >
                <Space size="middle" className="flex flex-wrap">
                  {r.status === 'healthy' ? (
                    <Tag color="green" className={recentHealthy.has(c.key) ? 'animate-pulse ring-2 ring-emerald-400' : undefined}>
                      {t('healthPage.ready', READY_STATE_LABEL)}
                    </Tag>
                  ) : r.status === 'unhealthy' ? (
                    <Tag color="red">{t('healthPage.degraded', DEGRADED_STATE_LABEL)}</Tag>
                  ) : (
                    <Tag>{t('healthPage.loadingState', LOADING_STATE_LABEL)}</Tag>
                  )}
                  <Typography.Text type="secondary">{c.path}</Typography.Text>
                  {typeof r.statusCode !== 'undefined' && (
                    <Tag>
                      {r.statusCode && r.statusCode > 0
                        ? t('healthPage.statusCodeTag', 'HTTP {{code}}', {
                            code: r.statusCode
                          })
                        : t('healthPage.statusCodeNetwork', 'Network/timeout')}
                    </Tag>
                  )}
                  {typeof r.durationMs !== 'undefined' && (
                    <Tag>{r.durationMs} ms</Tag>
                  )}
                </Space>
                {description && (
                  <Typography.Paragraph
                    type="secondary"
                    className="!mt-3 !mb-1 text-xs md:text-sm"
                  >
                    {description}
                  </Typography.Paragraph>
                )}
                {isUnhealthy && docsUrl && (
                  <Typography.Link
                    className="text-xs"
                    onClick={() => {
                      try {
                        browser.tabs.create({ url: docsUrl })
                      } catch {
                        window.open(docsUrl, '_blank')
                      }
                    }}
                  >
                    {t(
                      'healthPage.troubleshootLink',
                      'Troubleshooting tips'
                    )}
                  </Typography.Link>
                )}
                {r.detail && (
                  <>
                    <div className="mt-3 flex items-center justify-between">
                      <Typography.Text
                        type="secondary"
                        className="text-xs">
                        {t(
                          'healthPage.errorDetailsLabel',
                          'Technical details (for advanced users)'
                        )}
                      </Typography.Text>
                      <Button
                        size="small"
                        type="link"
                        onClick={async () => {
                          try {
                            const payload = {
                              check: c.key,
                              path: c.path,
                              status: r.status,
                              statusCode: r.statusCode,
                              durationMs: r.durationMs,
                              detail: r.detail
                            }
                            const text = JSON.stringify(payload, null, 2)
                            await navigator.clipboard.writeText(text)
                            notification.success({
                              message: t(
                                'healthPage.copiedError',
                                'Error details copied'
                              ),
                              placement: 'bottomRight',
                              duration: 2
                            })
                          } catch {
                            notification.error({
                              message: t(
                                'healthPage.copyFailed',
                                'Failed to copy diagnostics'
                              ),
                              placement: 'bottomRight',
                              duration: 2
                            })
                          }
                        }}>
                        {t('healthPage.copyError', 'Copy error')}
                      </Button>
                    </div>
                    <Typography.Text
                      type="secondary"
                      className="mt-1 block text-[10px] md:text-xs"
                    >
                      {t(
                        'healthPage.rawResponseFrom',
                        'Raw response from {{path}}',
                        { path: c.path }
                      )}
                    </Typography.Text>
                    <pre className="mt-1 p-2 bg-surface2 rounded text-xs overflow-auto max-h-40">
                      {displayedDetail}
                    </pre>
                  </>
                )}
              </Card>
            )
          })}
        </div>

        {capabilities?.hasChatQueue && (
          <div className="mt-6 space-y-3">
            <div className="flex items-center justify-between">
              <Typography.Title level={5} className="!mb-0 text-sm md:text-base">
                {t(
                  "healthPage.queueTitle",
                  "Chat queue diagnostics"
                )}
              </Typography.Title>
              <Button size="small" onClick={loadQueueDiagnostics} loading={queueLoading}>
                {t("common:refresh", "Refresh")}
              </Button>
            </div>
            {queueError && (
              <DesignSystemAlert
                variant="warning"
                title={t(
                  "healthPage.queueError",
                  "Queue diagnostics unavailable"
                )}
              >
                {queueError}
              </DesignSystemAlert>
            )}
            {queueStatus && (
              <Card
                title={t("healthPage.queueStatusTitle", "Queue status")}
              >
                <Space size="middle" className="flex flex-wrap">
                  <Tag color={queueStatus.enabled ? "green" : "default"}>
                    {queueStatus.enabled
                      ? t("healthPage.queueEnabled", "Enabled")
                      : t("healthPage.queueDisabled", "Disabled")}
                  </Tag>
                  {typeof queueStatus.queue_size !== "undefined" && (
                    <Tag>
                      {t("healthPage.queueDepth", "Queued")}:
                      {" "}
                      {queueStatus.queue_size}
                    </Tag>
                  )}
                  {typeof queueStatus.processing_count !== "undefined" && (
                    <Tag>
                      {t("healthPage.queueProcessing", "Processing")}:
                      {" "}
                      {queueStatus.processing_count}
                    </Tag>
                  )}
                  {typeof queueStatus.max_queue_size !== "undefined" && (
                    <Tag>
                      {t("healthPage.queueMax", "Max queue")}:
                      {" "}
                      {queueStatus.max_queue_size}
                    </Tag>
                  )}
                  {typeof queueStatus.max_concurrent !== "undefined" && (
                    <Tag>
                      {t("healthPage.queueMaxConcurrent", "Max concurrent")}:
                      {" "}
                      {queueStatus.max_concurrent}
                    </Tag>
                  )}
                  {typeof queueStatus.total_processed !== "undefined" && (
                    <Tag>
                      {t("healthPage.queueTotalProcessed", "Processed")}:
                      {" "}
                      {queueStatus.total_processed}
                    </Tag>
                  )}
                  {typeof queueStatus.total_rejected !== "undefined" && (
                    <Tag>
                      {t("healthPage.queueTotalRejected", "Rejected")}:
                      {" "}
                      {queueStatus.total_rejected}
                    </Tag>
                  )}
                </Space>
                {queueStatus.message && (
                  <Typography.Paragraph type="secondary" className="!mt-2 !mb-0 text-xs">
                    {queueStatus.message}
                  </Typography.Paragraph>
                )}
              </Card>
            )}
            {queueActivity && Array.isArray(queueActivity.activity) && (
              <Card title={t("healthPage.queueActivityTitle", "Recent activity")}>
                {queueActivity.activity.length === 0 ? (
                  <Typography.Paragraph type="secondary" className="!mb-0 text-xs">
                    {t("healthPage.queueActivityEmpty", "No recent activity.")}
                  </Typography.Paragraph>
                ) : (
                  <div className="space-y-2">
                    {queueActivity.activity.slice(0, 10).map((item: any, idx: number) => (
                      <div
                        key={`${item.request_id || idx}`}
                        className="rounded-md border border-border bg-surface2 p-2 text-xs"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-text">
                            {item.request_id || t("healthPage.queueItem", "Request")}
                          </span>
                          <Tag>{item.result || item.status || "ok"}</Tag>
                        </div>
                        <div className="text-text-subtle">
                          {t(
                            "healthPage.queueItemMeta",
                            "Priority {{priority}} · {{duration}}s",
                            {
                              priority: item.priority ?? "n/a",
                              duration:
                                typeof item.duration === "number"
                                  ? item.duration.toFixed(2)
                                  : "n/a"
                            }
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </Card>
            )}
          </div>
        )}
      </div>
    </Space>
  )
}
