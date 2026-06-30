import {
  Segmented,
  Space,
  Button,
  Select,
  Tag
} from "antd"
import React from "react"
import type { TFunction } from "i18next"
import { Alert as DsAlert } from "@/components/ui/primitives"

export type BillingPlan = {
  name: string
  display_name: string
  description?: string
  price_usd_monthly?: number
  price_usd_yearly?: number
  limits?: Record<string, any>
}

export type BillingSubscription = {
  plan_name: string
  plan_display_name: string
  status: string
  billing_cycle?: string | null
  current_period_end?: string | null
  trial_end?: string | null
  cancel_at_period_end?: boolean
  limits?: Record<string, any>
}

export type BillingUsage = {
  plan_name?: string
  limits?: Record<string, any>
  usage?: Record<string, number>
  limit_checks?: Record<string, {
    limit?: number | null
    current?: number
    exceeded?: boolean
    warning?: boolean
    unlimited?: boolean
    percent_used?: number
  }>
  has_warnings?: boolean
  has_exceeded?: boolean
}

export type BillingInvoice = {
  id: number
  amount_cents: number
  currency?: string
  status?: string
  description?: string | null
  invoice_pdf_url?: string | null
  created_at?: string | null
  amount_display?: string
}

export type BillingInvoiceList = {
  items: BillingInvoice[]
  total: number
}

export type TldwBillingSettingsProps = {
  t: TFunction
  billingLoading: boolean
  billingError: string | null
  billingPlansError: string | null
  billingStatusError: string | null
  billingUsageError: string | null
  billingPlans: BillingPlan[]
  billingStatus: BillingSubscription | null
  billingUsage: BillingUsage | null
  billingInvoices: BillingInvoice[]
  billingInvoicesTotal: number
  billingInvoicesLoading: boolean
  billingInvoicesError: string | null
  billingActionLoading: boolean
  selectedPlan: string | null
  setSelectedPlan: (plan: string | null) => void
  billingCycle: "monthly" | "yearly"
  setBillingCycle: (cycle: "monthly" | "yearly") => void
  onLoadBilling: () => void
  onLoadInvoices: () => void
  onCheckout: () => void
  onBillingPortal: () => void
  onCancelSubscription: () => void
  onResumeSubscription: () => void
}

const formatNumber = (t: TFunction, value?: number | null) => {
  if (value === null || value === undefined) {
    return t('settings:tldw.billing.unknown', '—')
  }
  if (Number.isFinite(value)) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 2 })
  }
  return String(value)
}

const formatLimitValue = (t: TFunction, value?: number | null, unlimited?: boolean) => {
  if (unlimited) {
    return t('settings:tldw.billing.unlimited', 'Unlimited')
  }
  if (value === null || value === undefined) {
    return t('settings:tldw.billing.unknown', '—')
  }
  if (value === -1) {
    return t('settings:tldw.billing.unlimited', 'Unlimited')
  }
  return formatNumber(t, value)
}

const formatUsageLabel = (t: TFunction, key: string) => {
  const map: Record<string, string> = {
    api_calls_day: t('settings:tldw.billing.usage.apiCallsDay', 'API calls / day'),
    llm_tokens_month: t('settings:tldw.billing.usage.llmTokensMonth', 'LLM tokens / month'),
    storage_mb: t('settings:tldw.billing.usage.storageMb', 'Storage (MB)'),
    team_members: t('settings:tldw.billing.usage.teamMembers', 'Team members'),
    transcription_minutes_month: t('settings:tldw.billing.usage.transcriptionMinutes', 'Transcription minutes / month'),
    rag_queries_day: t('settings:tldw.billing.usage.ragQueriesDay', 'RAG queries / day'),
    concurrent_jobs: t('settings:tldw.billing.usage.concurrentJobs', 'Concurrent jobs')
  }
  return map[key] || key.replace(/_/g, ' ')
}

const formatPlanPrice = (t: TFunction, plan: BillingPlan, cycle: 'monthly' | 'yearly') => {
  const price =
    cycle === 'yearly'
      ? plan.price_usd_yearly
      : plan.price_usd_monthly
  if (typeof price === 'number' && !Number.isNaN(price)) {
    const suffix = cycle === 'yearly' ? 'yr' : 'mo'
    return `$${price.toLocaleString()}/${suffix}`
  }
  return t('settings:tldw.billing.customPrice', 'Custom')
}

const formatDate = (t: TFunction, value?: string | null) => {
  if (!value) return t('settings:tldw.billing.unknown', '—')
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    return value
  }
  return parsed.toLocaleDateString()
}

const billingStatusColor = (status?: string | null) => {
  switch (status) {
    case "active":
      return "green"
    case "trialing":
      return "blue"
    case "past_due":
      return "orange"
    case "canceling":
      return "gold"
    case "canceled":
      return "red"
    default:
      return "default"
  }
}

const billingStatusLabel = (t: TFunction, status?: string | null) => {
  switch (status) {
    case "active":
      return t('settings:tldw.billing.status.active', 'Active')
    case "trialing":
      return t('settings:tldw.billing.status.trialing', 'Trialing')
    case "past_due":
      return t('settings:tldw.billing.status.pastDue', 'Past due')
    case "canceling":
      return t('settings:tldw.billing.status.canceling', 'Canceling')
    case "canceled":
      return t('settings:tldw.billing.status.canceled', 'Canceled')
    default:
      return t('settings:tldw.billing.status.unknown', 'Unknown')
  }
}

const invoiceStatusColor = (status?: string | null) => {
  switch (status) {
    case "succeeded":
      return "green"
    case "failed":
      return "red"
    case "pending":
      return "orange"
    default:
      return "default"
  }
}

const invoiceStatusLabel = (t: TFunction, status?: string | null) => {
  switch (status) {
    case "succeeded":
      return t('settings:tldw.billing.invoice.status.paid', 'Paid')
    case "failed":
      return t('settings:tldw.billing.invoice.status.failed', 'Failed')
    case "pending":
      return t('settings:tldw.billing.invoice.status.pending', 'Pending')
    default:
      return t('settings:tldw.billing.invoice.status.unknown', 'Unknown')
  }
}

const formatInvoiceAmount = (invoice: BillingInvoice) => {
  if (invoice.amount_display) {
    return invoice.amount_display
  }
  const currency = (invoice.currency || "USD").toUpperCase()
  const amount =
    typeof invoice.amount_cents === "number" ? invoice.amount_cents / 100 : 0

  try {
    return new Intl.NumberFormat(undefined, {
      style: "currency",
      currency
    }).format(amount)
  } catch {
    return `${amount.toFixed(2)} ${currency}`
  }
}

const USAGE_ORDER = [
  "api_calls_day",
  "llm_tokens_month",
  "storage_mb",
  "team_members",
  "transcription_minutes_month",
  "rag_queries_day",
  "concurrent_jobs"
]

export const TldwBillingSettings = ({
  t,
  billingLoading,
  billingError,
  billingPlansError,
  billingStatusError,
  billingUsageError,
  billingPlans,
  billingStatus,
  billingUsage,
  billingInvoices,
  billingInvoicesTotal,
  billingInvoicesLoading,
  billingInvoicesError,
  billingActionLoading,
  selectedPlan,
  setSelectedPlan,
  billingCycle,
  setBillingCycle,
  onLoadBilling,
  onLoadInvoices,
  onCheckout,
  onBillingPortal,
  onCancelSubscription,
  onResumeSubscription
}: TldwBillingSettingsProps) => {
  const selectedPlanDetails = React.useMemo(
    () => billingPlans.find((plan) => plan.name === selectedPlan) || null,
    [billingPlans, selectedPlan]
  )

  const planOptions = React.useMemo(
    () =>
      billingPlans.map((plan) => ({
        value: plan.name,
        label: (
          <div className="flex items-center justify-between gap-2">
            <span>{plan.display_name}</span>
            <span className="text-xs text-text-muted">
              {formatPlanPrice(t, plan, billingCycle)}
            </span>
          </div>
        )
      })),
    [billingCycle, billingPlans, t]
  )

  const usageChecks = billingUsage?.limit_checks ?? {}
  const sortedUsageEntries = React.useMemo(() => {
    const usageEntries = Object.entries(billingUsage?.usage ?? {})

    return usageEntries.sort((a, b) => {
      const aIndex = USAGE_ORDER.indexOf(a[0])
      const bIndex = USAGE_ORDER.indexOf(b[0])
      const aRank = aIndex === -1 ? USAGE_ORDER.length + 1 : aIndex
      const bRank = bIndex === -1 ? USAGE_ORDER.length + 1 : bIndex
      if (aRank !== bRank) return aRank - bRank
      return a[0].localeCompare(b[0])
    })
  }, [billingUsage?.usage])

  const isSamePlan = !!billingStatus?.plan_name && selectedPlan === billingStatus?.plan_name
  const isSameCycle = !!billingStatus?.billing_cycle && billingStatus?.billing_cycle === billingCycle

  return (
    <div
      id="tldw-settings-billing"
      className="mt-6 scroll-mt-24 rounded-lg border border-border bg-surface2 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold text-text">
            {t('settings:tldw.billing.title', 'Billing & usage')}
          </h3>
          <p className="text-xs text-text-muted">
            {t(
              'settings:tldw.billing.subtitle',
              'Manage your plan, billing cycle, and usage limits.'
            )}
          </p>
        </div>
        <Space>
          <Button
            onClick={() => {
              onLoadBilling()
              onLoadInvoices()
            }}
            loading={billingLoading || billingInvoicesLoading}
          >
            {t('settings:tldw.billing.refresh', 'Refresh')}
          </Button>
          <Button onClick={onBillingPortal} loading={billingLoading}>
            {t('settings:tldw.billing.portal', 'Billing portal')}
          </Button>
        </Space>
      </div>

      {billingError && (
        <DsAlert
          variant="error"
          className="mt-4"
          title={t('settings:tldw.billing.errorTitle', 'Billing unavailable')}
        >
          {billingError}
        </DsAlert>
      )}

      <div className="mt-4 space-y-4">
        {billingStatusError ? (
          <DsAlert
            variant="error"
            title={t('settings:tldw.billing.subscriptionError', 'Unable to load subscription')}
          >
            {billingStatusError}
          </DsAlert>
        ) : billingStatus ? (
          <div className="rounded border border-border bg-surface p-3">
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <span className="font-medium">
                {t('settings:tldw.billing.currentPlan', 'Current plan')}
              </span>
              <Tag color={billingStatusColor(billingStatus.status)}>
                {billingStatusLabel(t, billingStatus.status)}
              </Tag>
              <span className="text-text">
                {billingStatus.plan_display_name || billingStatus.plan_name}
              </span>
              {billingStatus.billing_cycle && (
                <Tag>
                  {billingStatus.billing_cycle === 'yearly'
                    ? t('settings:tldw.billing.cycle.yearly', 'Yearly')
                    : t('settings:tldw.billing.cycle.monthly', 'Monthly')}
                </Tag>
              )}
            </div>
            <div className="mt-2 text-xs text-text-muted flex flex-wrap gap-4">
              <span>
                {t('settings:tldw.billing.renewal', 'Renews')}:{" "}
                {formatDate(t, billingStatus.current_period_end)}
              </span>
              {billingStatus.trial_end && (
                <span>
                  {t('settings:tldw.billing.trialEnds', 'Trial ends')}:{" "}
                  {formatDate(t, billingStatus.trial_end)}
                </span>
              )}
            </div>
            {billingStatus.cancel_at_period_end && (
              <DsAlert
                variant="warning"
                className="mt-3"
                title={t(
                  'settings:tldw.billing.cancelAtPeriodEnd',
                  'Subscription will cancel at period end.'
                )}
              />
            )}
            <div className="mt-3 flex flex-wrap gap-2">
              {!billingStatus.cancel_at_period_end &&
                billingStatus.status !== 'canceled' && (
                  <Button
                    danger
                    onClick={onCancelSubscription}
                    loading={billingActionLoading}
                  >
                    {t(
                      'settings:tldw.billing.cancelAction',
                      'Cancel at period end'
                    )}
                  </Button>
                )}
              {billingStatus.cancel_at_period_end &&
                billingStatus.status !== 'canceled' && (
                  <Button
                    onClick={onResumeSubscription}
                    loading={billingActionLoading}
                  >
                    {t(
                      'settings:tldw.billing.resumeAction',
                      'Resume subscription'
                    )}
                  </Button>
                )}
            </div>
          </div>
        ) : !billingLoading ? (
          <div className="rounded border border-border bg-surface p-3">
            <div className="text-sm font-medium mb-1">
              {t('settings:tldw.billing.currentPlan', 'Current plan')}
            </div>
            <div className="text-xs text-text-muted">
              {t(
                'settings:tldw.billing.subscriptionEmpty',
                'No active subscription yet. Choose a plan to get started.'
              )}
            </div>
          </div>
        ) : null}

        {billingUsage?.has_exceeded && (
          <DsAlert
            variant="error"
            title={t(
              'settings:tldw.billing.limitExceeded',
              'Usage has exceeded one or more plan limits.'
            )}
          >
            <div className="mt-1">
              {billingUsage.limit_checks && (
                <div className="text-xs mb-2">
                  {Object.entries(billingUsage.limit_checks)
                    .filter(([, v]) => v.exceeded)
                    .map(([k]) => k.replace(/_/g, ' '))
                    .join(', ') || 'plan limits'}
                </div>
              )}
              <button
                type="button"
                className="text-xs underline"
                onClick={() => document.getElementById('tldw-billing-plans')?.scrollIntoView({ behavior: 'smooth' })}
              >
                {t('settings:tldw.billing.viewPlans', 'View upgrade options')}
              </button>
            </div>
          </DsAlert>
        )}
        {!billingUsage?.has_exceeded && billingUsage?.has_warnings && (
          <DsAlert
            variant="warning"
            title={t(
              'settings:tldw.billing.limitWarning',
              'Approaching plan limits for some resources.'
            )}
          >
            {billingUsage.limit_checks ? (
              <div className="text-xs mt-1">
                {Object.entries(billingUsage.limit_checks)
                  .filter(([, v]) => v.warning && !v.exceeded)
                  .map(([k]) => k.replace(/_/g, ' '))
                  .join(', ')} {t('settings:tldw.billing.nearingLimit', 'usage is nearing the limit.')}
              </div>
            ) : null}
          </DsAlert>
        )}

        <div id="tldw-billing-plans" className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="rounded border border-border bg-surface p-3">
            <div className="text-sm font-medium mb-2">
              {t('settings:tldw.billing.selectPlan', 'Select a plan')}
            </div>
            {billingPlansError && (
              <DsAlert
                variant="error"
                title={t('settings:tldw.billing.plansError', 'Unable to load plans')}
              >
                {billingPlansError}
              </DsAlert>
            )}
            {!billingPlansError && (
              <>
                <Select
                  className="w-full"
                  placeholder={t('settings:tldw.billing.choosePlan', 'Choose a plan')}
                  options={planOptions}
                  value={selectedPlan || undefined}
                  onChange={(value) => setSelectedPlan(value)}
                  disabled={billingPlans.length === 0}
                />
                <div className="mt-3">
                  <span className="text-xs text-text-muted">
                    {t('settings:tldw.billing.billingCycle', 'Billing cycle')}
                  </span>
                  <div className="mt-2">
                    <Segmented
                      options={[
                        { label: t('settings:tldw.billing.cycle.monthly', 'Monthly'), value: 'monthly' },
                        { label: t('settings:tldw.billing.cycle.yearly', 'Yearly'), value: 'yearly' }
                      ]}
                      value={billingCycle}
                      onChange={(value) => {
                        if (value === 'monthly' || value === 'yearly') {
                          setBillingCycle(value)
                        }
                      }}
                    />
                  </div>
                </div>
                {selectedPlanDetails && (
                  <div className="mt-3 text-xs text-text-muted space-y-1">
                    <div className="font-medium text-text">
                      {selectedPlanDetails.display_name}
                    </div>
                    {selectedPlanDetails.description && (
                      <div>{selectedPlanDetails.description}</div>
                    )}
                    <div>
                      {t('settings:tldw.billing.price', 'Price')}:{" "}
                      {formatPlanPrice(t, selectedPlanDetails, billingCycle)}
                    </div>
                  </div>
                )}
                <div className="mt-4 flex flex-wrap gap-2">
                  <Button
                    type="primary"
                    onClick={onCheckout}
                    loading={billingLoading}
                    disabled={!selectedPlan || (isSamePlan && isSameCycle)}
                  >
                    {isSamePlan && isSameCycle
                      ? t('settings:tldw.billing.currentPlanCta', 'Current plan')
                      : t('settings:tldw.billing.checkout', 'Continue to checkout')}
                  </Button>
                  {billingPlans.length === 0 && !billingLoading && (
                    <span className="text-xs text-text-subtle">
                      {t('settings:tldw.billing.noPlans', 'No plans available yet.')}
                    </span>
                  )}
                </div>
              </>
            )}
          </div>

          <div className="rounded border border-border bg-surface p-3">
            <div className="text-sm font-medium mb-2">
              {t('settings:tldw.billing.usageTitle', 'Usage')}
            </div>
            {billingUsageError && (
              <DsAlert
                variant="error"
                title={t('settings:tldw.billing.usageError', 'Unable to load usage data')}
              >
                {billingUsageError}
              </DsAlert>
            )}
            {!billingUsageError && !billingLoading && sortedUsageEntries.length === 0 && (
              <span className="text-xs text-text-muted">
                {t('settings:tldw.billing.usageEmpty', 'Usage data will appear after activity.')}
              </span>
            )}
            {!billingUsageError && sortedUsageEntries.length > 0 && (
              <div className="space-y-2">
                {sortedUsageEntries.map(([key, value]) => {
                  const check = usageChecks[key] || {}
                  const limit = typeof check.limit !== 'undefined'
                    ? check.limit
                    : billingUsage?.limits?.[key]
                  const statusColor = check.exceeded
                    ? 'red'
                    : check.warning
                      ? 'orange'
                      : 'green'
                  return (
                    <div key={key} className="flex flex-wrap items-center justify-between gap-2 text-xs">
                      <span className="text-text">{formatUsageLabel(t, key)}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-text-muted">
                          {formatNumber(t, value)} / {formatLimitValue(t, limit, check.unlimited)}
                        </span>
                        {typeof check.percent_used === 'number' && !check.unlimited && (
                          <Tag color={statusColor}>
                            {Math.round(check.percent_used)}%
                          </Tag>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>

        <div className="rounded border border-border bg-surface p-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="text-sm font-medium">
              {t('settings:tldw.billing.invoices.title', 'Invoice history')}
            </div>
            {billingInvoicesTotal > 0 && (
              <span className="text-xs text-text-muted">
                {t(
                  'settings:tldw.billing.invoices.total',
                  'Total: {{count}}',
                  { count: billingInvoicesTotal }
                )}
              </span>
            )}
          </div>
          {billingInvoicesLoading && (
            <div className="mt-2 text-xs text-text-muted">
              {t('settings:tldw.billing.invoices.loading', 'Loading invoices\u2026')}
            </div>
          )}
          {billingInvoicesError && (
            <DsAlert
              variant="error"
              className="mt-3"
              title={t('settings:tldw.billing.invoices.error', 'Unable to load invoices')}
            >
              {billingInvoicesError}
            </DsAlert>
          )}
          {!billingInvoicesLoading && !billingInvoicesError && billingInvoices.length === 0 && (
            <div className="mt-2 text-xs text-text-muted">
              {t('settings:tldw.billing.invoices.empty', 'No invoices yet.')}
            </div>
          )}
          {!billingInvoicesError && billingInvoices.length > 0 && (
            <div className="mt-3 space-y-2">
              {billingInvoices.map((invoice) => (
                <div
                  key={invoice.id}
                  className="flex flex-wrap items-center justify-between gap-2 border-b border-border/50 pb-2 text-xs"
                >
                  <div className="space-y-1">
                    <div className="font-medium text-text">
                      {formatInvoiceAmount(invoice)}
                    </div>
                    <div className="text-text-muted">
                      {formatDate(t, invoice.created_at)}{" "}
                      {invoice.description ? `· ${invoice.description}` : `· #${invoice.id}`}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Tag color={invoiceStatusColor(invoice.status)}>
                      {invoiceStatusLabel(t, invoice.status)}
                    </Tag>
                    {invoice.invoice_pdf_url && (
                      <a
                        href={invoice.invoice_pdf_url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-primary hover:text-primaryStrong underline"
                      >
                        {t('settings:tldw.billing.invoices.pdf', 'PDF')}
                      </a>
                    )}
                  </div>
                </div>
              ))}
              {billingInvoicesTotal > billingInvoices.length && (
                <div className="text-xs text-text-muted">
                  {t(
                    'settings:tldw.billing.invoices.showing',
                    'Showing {{count}} of {{total}}',
                    { count: billingInvoices.length, total: billingInvoicesTotal }
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
