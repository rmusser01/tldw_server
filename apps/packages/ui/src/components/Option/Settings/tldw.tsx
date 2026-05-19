import {
  Alert,
  Form,
  Modal,
  Spin,
  Button,
  Space
} from "antd"
import { Link, useNavigate } from "react-router-dom"
import React, { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { tldwClient, TldwConfig } from "@/services/tldw/TldwApiClient"
import { tldwAuth } from "@/services/tldw/TldwAuth"
import { SettingsSkeleton } from "@/components/Common/Settings/SettingsSkeleton"
import { DEFAULT_TLDW_API_KEY } from "@/services/tldw-server"
import { apiSend } from "@/services/api-send"
import type { PathOrUrl } from "@/services/tldw/openapi-guard"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import { useConnectionStore } from "@/store/connection"
import { mapMultiUserLoginErrorMessage } from "@/services/auth-errors"
import { emitSplashAfterSingleUserAuthSuccess } from "@/services/splash-auth"
import { ServerOverviewHint } from "@/components/Common/ServerOverviewHint"
import { requestOptionalHostPermission } from "@/utils/extension-permissions"
import type { CoreStatus, RagStatus } from "./tldw-connection-status"
import { TldwSettingsTabs } from "./tldw-settings-tabs"
import { probeServerHealth } from "./server-health-probe"
import { TldwConnectionSettings, type LoginMethod } from "./TldwConnectionSettings"
import {
  TldwTimeoutSettings,
  TIMEOUT_PRESETS,
  determinePreset,
  type TimeoutPresetKey
} from "./TldwTimeoutSettings"
import {
  TldwBillingSettings,
  type BillingPlan,
  type BillingSubscription,
  type BillingUsage,
  type BillingInvoice,
  type BillingInvoiceList
} from "./TldwBillingSettings"

const BILLING_BASE_URL = "https://vademhq.com"
const BILLING_SUCCESS_URL = `${BILLING_BASE_URL}/billing/success`
const BILLING_CANCEL_URL = `${BILLING_BASE_URL}/billing/cancel`
const BILLING_RETURN_URL = `${BILLING_BASE_URL}/billing`

export const TldwSettings = () => {
  const { t } = useTranslation(["settings", "common"])
  const message = useAntdMessage()
  const navigate = useNavigate()
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [initializing, setInitializing] = useState(true)
  const [initializingError, setInitializingError] = useState<string | null>(null)
  const [testingConnection, setTestingConnection] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'success' | 'error' | null>(null)
  const [connectionDetail, setConnectionDetail] = useState<string>("")
  const [coreStatus, setCoreStatus] = useState<CoreStatus>("unknown")
  const [ragStatus, setRagStatus] = useState<RagStatus>("unknown")
  const [authMode, setAuthMode] = useState<'single-user' | 'multi-user'>('single-user')
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [loginMethod, setLoginMethod] = useState<LoginMethod>('magic-link')
  const [magicEmail, setMagicEmail] = useState("")
  const [magicToken, setMagicToken] = useState("")
  const [magicSent, setMagicSent] = useState(false)
  const [magicSending, setMagicSending] = useState(false)
  const [serverUrl, setServerUrl] = useState("")
  const [requestTimeoutSec, setRequestTimeoutSec] = useState<number>(10)
  const [streamIdleTimeoutSec, setStreamIdleTimeoutSec] = useState<number>(15)
  const [chatRequestTimeoutSec, setChatRequestTimeoutSec] = useState<number>(10)
  const [chatStartupTimeoutSec, setChatStartupTimeoutSec] = useState<number>(10)
  const [chatStreamIdleTimeoutSec, setChatStreamIdleTimeoutSec] = useState<number>(15)
  const [ragRequestTimeoutSec, setRagRequestTimeoutSec] = useState<number>(10)
  const [mediaRequestTimeoutSec, setMediaRequestTimeoutSec] = useState<number>(60)
  const [uploadRequestTimeoutSec, setUploadRequestTimeoutSec] = useState<number>(60)
  const [timeoutPreset, setTimeoutPreset] = useState<TimeoutPresetKey | 'custom'>('balanced')
  const [showDefaultKeyWarning, setShowDefaultKeyWarning] = useState(false)
  const [billingLoading, setBillingLoading] = useState(false)
  const [billingError, setBillingError] = useState<string | null>(null)
  const [billingPlansError, setBillingPlansError] = useState<string | null>(null)
  const [billingStatusError, setBillingStatusError] = useState<string | null>(null)
  const [billingUsageError, setBillingUsageError] = useState<string | null>(null)
  const [billingPlans, setBillingPlans] = useState<BillingPlan[]>([])
  const [billingStatus, setBillingStatus] = useState<BillingSubscription | null>(null)
  const [billingUsage, setBillingUsage] = useState<BillingUsage | null>(null)
  const [billingInvoices, setBillingInvoices] = useState<BillingInvoice[]>([])
  const [billingInvoicesTotal, setBillingInvoicesTotal] = useState(0)
  const [billingInvoicesLoading, setBillingInvoicesLoading] = useState(false)
  const [billingInvoicesError, setBillingInvoicesError] = useState<string | null>(null)
  const [billingActionLoading, setBillingActionLoading] = useState(false)
  const [selectedPlan, setSelectedPlan] = useState<string | null>(null)
  const [billingCycle, setBillingCycle] = useState<'monthly' | 'yearly'>('monthly')

  // ── Config load ──────────────────────────────────────────────────

  useEffect(() => {
    loadConfig()
  }, [])

  const loadConfig = async () => {
    setLoading(true)
    setInitializingError(null)
    try {
      const config = await tldwClient.getConfig()
      if (config) {
        setAuthMode(config.authMode)
        setServerUrl(config.serverUrl)
        const nextTimeouts = { ...TIMEOUT_PRESETS.balanced }
        if (typeof (config as any).requestTimeoutMs === 'number') nextTimeouts.request = Math.round((config as any).requestTimeoutMs / 1000)
        if (typeof (config as any).streamIdleTimeoutMs === 'number') nextTimeouts.stream = Math.round((config as any).streamIdleTimeoutMs / 1000)
        if (typeof (config as any).chatRequestTimeoutMs === 'number') nextTimeouts.chatRequest = Math.round((config as any).chatRequestTimeoutMs / 1000)
        if (typeof (config as any).chatStartupTimeoutMs === 'number') nextTimeouts.chatStartup = Math.round((config as any).chatStartupTimeoutMs / 1000)
        if (typeof (config as any).chatStreamIdleTimeoutMs === 'number') nextTimeouts.chatStream = Math.round((config as any).chatStreamIdleTimeoutMs / 1000)
        if (typeof (config as any).ragRequestTimeoutMs === 'number') nextTimeouts.ragRequest = Math.round((config as any).ragRequestTimeoutMs / 1000)
        if (typeof (config as any).mediaRequestTimeoutMs === 'number') nextTimeouts.media = Math.round((config as any).mediaRequestTimeoutMs / 1000)
        if (typeof (config as any).uploadRequestTimeoutMs === 'number') nextTimeouts.upload = Math.round((config as any).uploadRequestTimeoutMs / 1000)

        setRequestTimeoutSec(nextTimeouts.request)
        setStreamIdleTimeoutSec(nextTimeouts.stream)
        setChatRequestTimeoutSec(nextTimeouts.chatRequest)
        setChatStartupTimeoutSec(nextTimeouts.chatStartup)
        setChatStreamIdleTimeoutSec(nextTimeouts.chatStream)
        setRagRequestTimeoutSec(nextTimeouts.ragRequest)
        setMediaRequestTimeoutSec(nextTimeouts.media)
        setUploadRequestTimeoutSec(nextTimeouts.upload)
        setTimeoutPreset(determinePreset(nextTimeouts))
        form.setFieldsValue({
          serverUrl: config.serverUrl,
          apiKey: config.apiKey,
          authMode: config.authMode
        })

        if (config.authMode === 'multi-user' && config.accessToken) {
          setIsLoggedIn(true)
        }
      } else {
        setTimeoutPreset('balanced')
      }
      setInitializingError(null)
    } catch (error) {
      console.error('Failed to load config:', error)
      setInitializingError(
        (error as Error)?.message ||
          t('settings:tldw.loadError', 'Unable to load tldw server settings. Check your connection and try again.')
      )
    } finally {
      setLoading(false)
      setInitializing(false)
    }
  }

  // ── Save handler ─────────────────────────────────────────────────

  const handleSave = async (values: any) => {
    setLoading(true)
    try {
      const config: Partial<TldwConfig & {
        requestTimeoutMs?: number
        streamIdleTimeoutMs?: number
        chatRequestTimeoutMs?: number
        chatStartupTimeoutMs?: number
        chatStreamIdleTimeoutMs?: number
        ragRequestTimeoutMs?: number
        mediaRequestTimeoutMs?: number
        uploadRequestTimeoutMs?: number
      }> = {
        serverUrl: values.serverUrl,
        authMode: values.authMode,
        requestTimeoutMs: Math.min(2147483000, Math.max(1, Math.round(Number(requestTimeoutSec) || 10)) * 1000),
        streamIdleTimeoutMs: Math.min(2147483000, Math.max(1, Math.round(Number(streamIdleTimeoutSec) || 15)) * 1000),
        chatRequestTimeoutMs: Math.min(2147483000, Math.max(1, Math.round(Number(chatRequestTimeoutSec) || requestTimeoutSec || 10)) * 1000),
        chatStartupTimeoutMs: Math.min(2147483000, Math.max(1, Math.round(Number(chatStartupTimeoutSec) || TIMEOUT_PRESETS.balanced.chatStartup)) * 1000),
        chatStreamIdleTimeoutMs: Math.min(2147483000, Math.max(1, Math.round(Number(chatStreamIdleTimeoutSec) || streamIdleTimeoutSec || 15)) * 1000),
        ragRequestTimeoutMs: Math.min(2147483000, Math.max(1, Math.round(Number(ragRequestTimeoutSec) || requestTimeoutSec || 10)) * 1000),
        mediaRequestTimeoutMs: Math.min(2147483000, Math.max(1, Math.round(Number(mediaRequestTimeoutSec) || requestTimeoutSec || 10)) * 1000),
        uploadRequestTimeoutMs: Math.min(2147483000, Math.max(1, Math.round(Number(uploadRequestTimeoutSec) || mediaRequestTimeoutSec || 60)) * 1000)
      }

      if (values.authMode === 'single-user') {
        config.apiKey = values.apiKey
        config.accessToken = undefined
        config.refreshToken = undefined
      }

      requestOptionalHostPermission(
        values.serverUrl,
        (granted, origin) => {
          if (!granted) {
            console.warn("Permission not granted for origin:", origin)
          }
        },
        (error) => {
          console.warn("Could not request optional host permission:", error)
        }
      )

      await tldwClient.updateConfig(config)
      message.success(t("settings:savedSuccessfully"))

      await testConnection({
        triggerSplashOnSuccess: values.authMode === "single-user"
      })
    } catch (error) {
      message.error(t("settings:saveFailed"))
      console.error('Failed to save config:', error)
    } finally {
      setLoading(false)
    }
  }

  // ── Billing loaders ──────────────────────────────────────────────

  const loadBilling = async () => {
    if (authMode !== 'multi-user' || !isLoggedIn) return
    setBillingLoading(true)
    setBillingError(null)
    setBillingPlansError(null)
    setBillingStatusError(null)
    setBillingUsageError(null)
    try {
      const [plansResp, subResp, usageResp] = await Promise.all([
        apiSend<{ plans?: BillingPlan[] }>({
          path: "/api/v1/billing/plans" as PathOrUrl,
          method: "GET"
        }),
        apiSend<BillingSubscription>({
          path: "/api/v1/billing/subscription" as PathOrUrl,
          method: "GET"
        }),
        apiSend<BillingUsage>({
          path: "/api/v1/billing/usage" as PathOrUrl,
          method: "GET"
        })
      ])
      const plans = plansResp?.data?.plans || []
      const subscription = subResp?.data || null
      const usage = usageResp?.data || null

      setBillingPlans(plans)
      setBillingStatus(subscription)
      setBillingUsage(usage)

      const plansError = !plansResp?.ok
        ? plansResp?.error || 'Failed to load plans'
        : null
      const statusError = !subResp?.ok
        ? subResp?.error || 'Failed to load subscription'
        : null
      const usageError = !usageResp?.ok
        ? usageResp?.error || 'Failed to load usage'
        : null

      setBillingPlansError(plansError)
      setBillingStatusError(statusError)
      setBillingUsageError(usageError)

      if (plansError || statusError) {
        setBillingError(plansError || statusError || 'Failed to load billing details')
      }
      if (plansError) {
        setBillingPlans([])
        setSelectedPlan(null)
      }
      if (statusError) {
        setBillingStatus(null)
      }
      if (usageError) {
        setBillingUsage(null)
      }

      if (!selectedPlan) {
        if (subscription?.plan_name) {
          setSelectedPlan(subscription.plan_name)
        } else if (plans.length > 0) {
          setSelectedPlan(plans[0].name)
        }
      }
      if (subscription?.billing_cycle) {
        setBillingCycle(subscription.billing_cycle === 'yearly' ? 'yearly' : 'monthly')
      }
    } catch (err: any) {
      const errorMsg = err?.message || 'Failed to load billing details'
      setBillingError(errorMsg)
      setBillingPlansError(errorMsg)
      setBillingStatusError(errorMsg)
      setBillingUsageError(errorMsg)
      setBillingPlans([])
      setBillingStatus(null)
      setBillingUsage(null)
      setSelectedPlan(null)
    } finally {
      setBillingLoading(false)
    }
  }

  const loadInvoices = async () => {
    if (authMode !== 'multi-user' || !isLoggedIn) return
    setBillingInvoicesLoading(true)
    setBillingInvoicesError(null)
    try {
      const resp = await apiSend<BillingInvoiceList>({
        path: "/api/v1/billing/invoices?limit=20" as PathOrUrl,
        method: "GET"
      })
      if (!resp.ok) {
        setBillingInvoicesError(resp.error || 'Failed to load invoices')
        setBillingInvoices([])
        setBillingInvoicesTotal(0)
        return
      }
      setBillingInvoices(resp.data?.items || [])
      setBillingInvoicesTotal(resp.data?.total || 0)
    } catch (err: any) {
      setBillingInvoicesError(err?.message || 'Failed to load invoices')
      setBillingInvoices([])
      setBillingInvoicesTotal(0)
    } finally {
      setBillingInvoicesLoading(false)
    }
  }

  useEffect(() => {
    if (authMode === 'multi-user' && isLoggedIn) {
      void loadBilling()
      void loadInvoices()
    }
  }, [authMode, isLoggedIn])

  // ── Connection test ──────────────────────────────────────────────

  const testConnection = async (options?: { triggerSplashOnSuccess?: boolean }) => {
    setTestingConnection(true)
    setConnectionStatus(null)
    setConnectionDetail("")
    setCoreStatus("checking")
    setRagStatus("unknown")

    let values: any = {}
    try {
      values = form.getFieldsValue()
      const errors = await form.validateFields(["serverUrl"]).catch(e => e)
      if (errors?.errorFields?.length) {
        return
      }
      let success = false

      const baseUrl = String(values.serverUrl || '').replace(/\/$/, '')
      const singleUser = values.authMode === "single-user"
      const hasApiKey =
        singleUser && typeof values.apiKey === "string" && values.apiKey.trim().length > 0

      const resp = baseUrl
        ? await probeServerHealth({
            serverUrl: baseUrl,
            authMode: values.authMode,
            apiKey: hasApiKey ? String(values.apiKey).trim() : undefined
          })
        : await apiSend({
            path: "/api/v1/health" as PathOrUrl,
            method: "GET",
            headers: hasApiKey ? { "X-API-KEY": String(values.apiKey).trim() } : undefined,
            noAuth: hasApiKey
          })

      success = !!resp?.ok
      setCoreStatus(success ? "connected" : "failed")

      if (!success) {
        if (resp?.status === 0) {
          requestOptionalHostPermission(values.serverUrl, (granted, origin) => {
            if (!granted) {
              message.warning(
                t(
                  "settings:siteAccessDenied",
                  "Permission not granted for {{origin}}",
                  { origin }
                )
              )
            }
          })
        }
        const code = resp?.status
        const detail = resp?.error || ""

        if (code === 401 || code === 403) {
          const hint =
            code === 401
              ? t(
                  "settings:tldw.errors.invalidApiKey",
                  "Invalid API key"
                )
              : t(
                  "settings:tldw.errors.forbidden",
                  "Forbidden (check permissions)"
                )
          const healthHint = t(
            "settings:tldw.errors.seeHealth",
            "Open Health & diagnostics for more details."
          )
          const suffix = code ? ` — HTTP ${code}` : ""
          const extra = detail ? ` (${detail})` : ""
          setConnectionDetail(`${hint}${suffix} — ${healthHint}${extra}`)
        } else {
          const base = t(
            "settings:tldw.errors.serverUnreachableDetailed",
            "Server not reachable. Check that your tldw_server is running and that your browser can reach it, then try again. Health & diagnostics can help debug connectivity issues."
          )
          const suffix = code ? ` — HTTP ${code}` : ""
          const extra = detail ? ` (${detail})` : ""
          setConnectionDetail(`${base}${suffix}${extra}`)
        }
      }

      setConnectionStatus(success ? 'success' : 'error')
      try {
        setRagStatus("checking")
        await tldwClient.initialize()
        const rag = await tldwClient.ragHealth()
        setRagStatus('healthy')
      } catch (e) {
        setRagStatus('unhealthy')
      }

      if (success) {
        message.success(t('settings:tldw.connection.success', 'Connection successful!'))
        if (options?.triggerSplashOnSuccess) {
          emitSplashAfterSingleUserAuthSuccess(values.authMode, true)
        }
        if (
          values.authMode === 'single-user' &&
          typeof values.apiKey === 'string' &&
          values.apiKey.trim() === DEFAULT_TLDW_API_KEY
        ) {
          setShowDefaultKeyWarning(true)
        } else {
          setShowDefaultKeyWarning(false)
        }
        await tldwClient.initialize()
        try {
          await useConnectionStore.getState().checkOnce()
        } catch {
          // Best-effort only; ignore failures here.
        }
      } else {
        message.error(t('settings:tldw.connection.failed', 'Connection failed. Please check your settings.'))
        setShowDefaultKeyWarning(false)
      }
    } catch (error) {
      setConnectionStatus('error')
      setCoreStatus("failed")
      const raw = (error as any)?.message || ''
      if (raw && /network|timeout|failed to fetch/i.test(raw)) {
        requestOptionalHostPermission(values.serverUrl, (granted, origin) => {
          if (!granted) {
            message.warning(
              t(
                "settings:siteAccessDenied",
                "Permission not granted for {{origin}}",
                { origin }
              )
            )
          }
        })
      }
      const friendly =
        raw && /network|timeout|failed to fetch/i.test(raw)
          ? t(
              'settings:tldw.errors.serverUnreachableDetailed',
              'Server not reachable. Check that your tldw_server is running and that your browser can reach it, then try again. Health & diagnostics can help debug connectivity issues.'
            )
          : raw ||
            t(
              'settings:tldw.errors.connectionFailedDetailed',
              'Connection failed. Please check your server URL and API key, then open Health & diagnostics for more details.'
            )
      setConnectionDetail(friendly)
      message.error(friendly)
      console.error('Connection test failed:', error)
    } finally {
      setTestingConnection(false)
    }
  }

  // ── Site access ──────────────────────────────────────────────────

  const grantSiteAccess = async () => {
    try {
      const values = form.getFieldsValue()
      const urlStr = String(values?.serverUrl || serverUrl || '')
      if (!urlStr) {
        message.warning(t('settings:enterServerUrlFirst', 'Enter a server URL first'))
        return
      }
      const origin = new URL(urlStr).origin
      const chromePermissions = (
        globalThis as {
          chrome?: {
            permissions?: {
              request?: (
                options: { origins: string[] },
                callback: (granted: boolean) => void
              ) => void
            }
          }
        }
      ).chrome?.permissions
      if (!chromePermissions?.request) {
        message.info(t('settings:siteAccessChromiumOnly', 'Site access is only needed on Chrome/Edge'))
        return
      }
      chromePermissions.request({ origins: [origin + '/*'] }, (granted) => {
        if (granted) message.success(t('settings:siteAccessGranted', 'Host permission granted for {{origin}}', { origin }))
        else message.warning(t('settings:siteAccessDenied', 'Permission not granted for {{origin}}', { origin }))
      })
    } catch (e: any) {
      message.error(t('settings:siteAccessFailed', 'Failed to request site access: {{msg}}', { msg: e?.message || String(e) }))
    }
  }

  // ── Auth handlers ────────────────────────────────────────────────

  const handleLogin = async () => {
    try {
      const values = await form.validateFields(['username', 'password'])
      setLoading(true)

      await tldwAuth.login({
        username: values.username,
        password: values.password
      })

      setIsLoggedIn(true)
      message.success(t('settings:tldw.login.success', 'Login successful!'))

      form.setFieldValue('password', '')

      await testConnection()
    } catch (error: any) {
      const friendly = mapMultiUserLoginErrorMessage(
        t,
        error,
        'settings'
      )
      message.error(friendly)
      console.error('Login failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSendMagicLink = async () => {
    if (!magicEmail.trim()) {
      message.warning(
        t('settings:tldw.magicLink.missingEmail', 'Enter your email to receive a magic link.')
      )
      return
    }
    setMagicSending(true)
    try {
      await tldwAuth.requestMagicLink(magicEmail.trim())
      setMagicSent(true)
      message.success(
        t('settings:tldw.magicLink.sent', 'Magic link sent. Check your inbox.')
      )
    } catch (error: any) {
      const friendly = mapMultiUserLoginErrorMessage(t, error, 'settings')
      message.error(friendly)
      console.error('Magic link request failed:', error)
    } finally {
      setMagicSending(false)
    }
  }

  const handleVerifyMagicLink = async () => {
    if (!magicToken.trim()) {
      message.warning(
        t('settings:tldw.magicLink.missingToken', 'Paste the magic link token to continue.')
      )
      return
    }
    setLoading(true)
    try {
      await tldwAuth.verifyMagicLink(magicToken.trim())
      setIsLoggedIn(true)
      message.success(t('settings:tldw.login.success', 'Login successful!'))
      setMagicToken('')
      await testConnection()
    } catch (error: any) {
      const friendly = mapMultiUserLoginErrorMessage(t, error, 'settings')
      message.error(friendly)
      console.error('Magic link login failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleLogout = async () => {
    try {
      setLoading(true)
      await tldwAuth.logout()
      setIsLoggedIn(false)
      message.success(t('settings:tldw.logout.success', 'Logged out successfully'))
    } catch (error) {
      message.error(t('settings:tldw.logout.failed', 'Logout failed'))
      console.error('Logout failed:', error)
    } finally {
      setLoading(false)
    }
  }

  // ── Billing action handlers ──────────────────────────────────────

  const handleCheckout = async () => {
    if (!selectedPlan) {
      message.warning(t('settings:tldw.billing.missingPlan', 'Select a plan to continue.'))
      return
    }
    setBillingLoading(true)
    try {
      const resp = await apiSend<{ url?: string }>({
        path: "/api/v1/billing/checkout" as PathOrUrl,
        method: "POST",
        body: {
          plan_name: selectedPlan,
          billing_cycle: billingCycle,
          success_url: BILLING_SUCCESS_URL,
          cancel_url: BILLING_CANCEL_URL
        }
      })
      if (resp?.data?.url) {
        window.open(resp.data.url, "_blank", "noopener,noreferrer")
      } else if (!resp?.ok) {
        message.error(resp?.error || t('settings:tldw.billing.checkoutFailed', 'Checkout failed.'))
      } else {
        message.error(t('settings:tldw.billing.checkoutFailed', 'Checkout failed.'))
      }
    } catch (error: any) {
      message.error(error?.message || t('settings:tldw.billing.checkoutFailed', 'Checkout failed.'))
    } finally {
      setBillingLoading(false)
    }
  }

  const handleBillingPortal = async () => {
    setBillingLoading(true)
    try {
      const resp = await apiSend<{ url?: string }>({
        path: "/api/v1/billing/portal" as PathOrUrl,
        method: "POST",
        body: { return_url: BILLING_RETURN_URL }
      })
      if (resp?.data?.url) {
        window.open(resp.data.url, "_blank", "noopener,noreferrer")
      } else if (!resp?.ok) {
        message.error(resp?.error || t('settings:tldw.billing.portalFailed', 'Unable to open billing portal.'))
      } else {
        message.error(t('settings:tldw.billing.portalFailed', 'Unable to open billing portal.'))
      }
    } catch (error: any) {
      message.error(error?.message || t('settings:tldw.billing.portalFailed', 'Unable to open billing portal.'))
    } finally {
      setBillingLoading(false)
    }
  }

  const handleCancelSubscription = async (atPeriodEnd: boolean) => {
    setBillingActionLoading(true)
    try {
      const resp = await apiSend<{ canceled?: boolean; current_period_end?: string }>({
        path: "/api/v1/billing/subscription/cancel" as PathOrUrl,
        method: "POST",
        body: { at_period_end: atPeriodEnd }
      })
      if (!resp.ok) {
        message.error(resp.error || t('settings:tldw.billing.cancelFailed', 'Unable to cancel subscription.'))
        return
      }
      message.success(
        atPeriodEnd
          ? t('settings:tldw.billing.cancelScheduled', 'Subscription will cancel at period end.')
          : t('settings:tldw.billing.cancelled', 'Subscription cancelled.')
      )
      await loadBilling()
      await loadInvoices()
    } catch (error: any) {
      message.error(error?.message || t('settings:tldw.billing.cancelFailed', 'Unable to cancel subscription.'))
    } finally {
      setBillingActionLoading(false)
    }
  }

  const handleResumeSubscription = async () => {
    setBillingActionLoading(true)
    try {
      const resp = await apiSend<{ resumed?: boolean }>({
        path: "/api/v1/billing/subscription/resume" as PathOrUrl,
        method: "POST"
      })
      if (!resp.ok) {
        message.error(resp.error || t('settings:tldw.billing.resumeFailed', 'Unable to resume subscription.'))
        return
      }
      message.success(t('settings:tldw.billing.resumed', 'Subscription resumed.'))
      await loadBilling()
      await loadInvoices()
    } catch (error: any) {
      message.error(error?.message || t('settings:tldw.billing.resumeFailed', 'Unable to resume subscription.'))
    } finally {
      setBillingActionLoading(false)
    }
  }

  const confirmCancelSubscription = () => {
    Modal.confirm({
      title: t('settings:tldw.billing.cancelTitle', 'Cancel subscription?'),
      content: t(
        'settings:tldw.billing.cancelBody',
        'This will cancel your subscription at the end of the current billing period.'
      ),
      okText: t('settings:tldw.billing.cancelConfirm', 'Cancel at period end'),
      okType: 'danger',
      cancelText: t('common:keep', 'Keep subscription'),
      centered: true,
      onOk: () => handleCancelSubscription(true)
    })
  }

  const confirmResumeSubscription = () => {
    Modal.confirm({
      title: t('settings:tldw.billing.resumeTitle', 'Resume subscription?'),
      content: t(
        'settings:tldw.billing.resumeBody',
        'This will keep your subscription active and remove the scheduled cancellation.'
      ),
      okText: t('settings:tldw.billing.resumeConfirm', 'Resume subscription'),
      cancelText: t('common:cancel', 'Cancel'),
      centered: true,
      onOk: () => handleResumeSubscription()
    })
  }

  const openHealthDiagnostics = () => {
    navigate("/settings/health")
  }

  // ── Render ───────────────────────────────────────────────────────

  if (initializing) {
    return (
      <div className="max-w-2xl">
        <SettingsSkeleton sections={3} />
      </div>
    )
  }

  return (
    <Spin
      spinning={loading}
      tip={loading ? t('common:saving', 'Saving...') : undefined}>
      <div className="max-w-2xl">
        {initializingError && (
          <Alert
            type="error"
            showIcon
            closable
            className="mb-4"
            title={t('settings:tldw.loadError', 'Unable to load tldw settings')}
            description={initializingError}
            onClose={() => setInitializingError(null)}
          />
        )}
        {showDefaultKeyWarning && authMode === 'single-user' && (
          <Alert
            type="warning"
            showIcon
            closable
            className="mb-4"
            title={t(
              'settings:tldw.defaultKeyWarning.title',
              'Default demo API key in use'
            )}
            description={t(
              'settings:tldw.defaultKeyWarning.body',
              'You are using the default demo API key for tldw_server. For production or shared deployments, rotate the key on your server and update it here. Continue at your own risk.'
            )}
            onClose={() => setShowDefaultKeyWarning(false)}
          />
        )}
        <div className="mb-4 rounded-lg bg-surface2 p-4">
          <h3 className="mb-2 font-semibold">
            {t(
              "settings:tldw.about.title",
              "About tldw server integration"
            )}
          </h3>
          <p className="text-sm text-text-muted">
            {t(
              "settings:tldw.about.description",
              "tldw server turns this extension into a full workspace for chat, knowledge search, and media."
            )}
          </p>
          <ServerOverviewHint />
        </div>
        <div className="mb-4 p-2 rounded border border-transparent bg-transparent flex items-center justify-between transition-colors duration-150 hover:border-border hover:bg-surface2">
          <div className="text-sm text-text">
            <span className="mr-2 font-medium">{t('settings:tldw.serverLabel', 'Server:')}</span>
            <span className="text-text-muted break-all">{serverUrl || t('settings:tldw.notConfigured', 'Not configured')}</span>
          </div>
          <Space>
            <Link to="/settings/health">
              <Button>{t('settings:tldw.buttons.health', 'Health')}</Button>
            </Link>
            <Button type="primary" onClick={() => { void testConnection() }} loading={testingConnection}>{t('settings:tldw.buttons.recheck', 'Recheck')}</Button>
          </Space>
        </div>
        <TldwSettingsTabs authMode={authMode} isLoggedIn={isLoggedIn} />
        <h2
          id="tldw-settings-connection"
          className="mb-4 scroll-mt-24 text-base font-semibold text-text">
          {t('settings:tldw.serverConfigTitle', 'tldw Server Configuration')}
        </h2>
        <Form
          form={form}
          onFinish={handleSave}
          layout="vertical"
          initialValues={{
            authMode: 'single-user',
            apiKey: ''
          }}
        >
          <TldwConnectionSettings
            t={t}
            form={form}
            authMode={authMode}
            setAuthMode={setAuthMode}
            isLoggedIn={isLoggedIn}
            setIsLoggedIn={setIsLoggedIn}
            loginMethod={loginMethod}
            setLoginMethod={setLoginMethod}
            magicEmail={magicEmail}
            setMagicEmail={setMagicEmail}
            magicToken={magicToken}
            setMagicToken={setMagicToken}
            magicSent={magicSent}
            setMagicSent={setMagicSent}
            magicSending={magicSending}
            testingConnection={testingConnection}
            connectionStatus={connectionStatus}
            connectionDetail={connectionDetail}
            coreStatus={coreStatus}
            ragStatus={ragStatus}
            onTestConnection={() => { void testConnection() }}
            onLogin={handleLogin}
            onSendMagicLink={handleSendMagicLink}
            onVerifyMagicLink={handleVerifyMagicLink}
            onLogout={handleLogout}
            onGrantSiteAccess={grantSiteAccess}
            onOpenHealthDiagnostics={openHealthDiagnostics}
          />
          <TldwTimeoutSettings
            t={t}
            message={message}
            requestTimeoutSec={requestTimeoutSec}
            setRequestTimeoutSec={setRequestTimeoutSec}
            streamIdleTimeoutSec={streamIdleTimeoutSec}
            setStreamIdleTimeoutSec={setStreamIdleTimeoutSec}
            chatRequestTimeoutSec={chatRequestTimeoutSec}
            setChatRequestTimeoutSec={setChatRequestTimeoutSec}
            chatStartupTimeoutSec={chatStartupTimeoutSec}
            setChatStartupTimeoutSec={setChatStartupTimeoutSec}
            chatStreamIdleTimeoutSec={chatStreamIdleTimeoutSec}
            setChatStreamIdleTimeoutSec={setChatStreamIdleTimeoutSec}
            ragRequestTimeoutSec={ragRequestTimeoutSec}
            setRagRequestTimeoutSec={setRagRequestTimeoutSec}
            mediaRequestTimeoutSec={mediaRequestTimeoutSec}
            setMediaRequestTimeoutSec={setMediaRequestTimeoutSec}
            uploadRequestTimeoutSec={uploadRequestTimeoutSec}
            setUploadRequestTimeoutSec={setUploadRequestTimeoutSec}
            timeoutPreset={timeoutPreset}
            setTimeoutPreset={setTimeoutPreset}
          />
        </Form>

        {authMode === 'multi-user' && isLoggedIn && (
          <TldwBillingSettings
            t={t}
            billingLoading={billingLoading}
            billingError={billingError}
            billingPlansError={billingPlansError}
            billingStatusError={billingStatusError}
            billingUsageError={billingUsageError}
            billingPlans={billingPlans}
            billingStatus={billingStatus}
            billingUsage={billingUsage}
            billingInvoices={billingInvoices}
            billingInvoicesTotal={billingInvoicesTotal}
            billingInvoicesLoading={billingInvoicesLoading}
            billingInvoicesError={billingInvoicesError}
            billingActionLoading={billingActionLoading}
            selectedPlan={selectedPlan}
            setSelectedPlan={setSelectedPlan}
            billingCycle={billingCycle}
            setBillingCycle={setBillingCycle}
            onLoadBilling={() => { void loadBilling() }}
            onLoadInvoices={() => { void loadInvoices() }}
            onCheckout={handleCheckout}
            onBillingPortal={handleBillingPortal}
            onCancelSubscription={confirmCancelSubscription}
            onResumeSubscription={confirmResumeSubscription}
          />
        )}
      </div>
    </Spin>
  )
}
