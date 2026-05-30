// @vitest-environment jsdom

import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import type { TFunction } from "i18next"
import {
  TldwBillingSettings,
  type TldwBillingSettingsProps
} from "../TldwBillingSettings"

vi.mock("antd", () => ({
  Alert: ({
    title,
    message,
    description
  }: {
    title?: React.ReactNode
    message?: React.ReactNode
    description?: React.ReactNode
  }) => (
    <div>
      {title}
      {message}
      {description}
    </div>
  ),
  Button: ({
    children,
    onClick,
    disabled
  }: {
    children?: React.ReactNode
    onClick?: () => void
    disabled?: boolean
  }) => (
    <button type="button" disabled={disabled} onClick={onClick}>
      {children}
    </button>
  ),
  Segmented: ({
    options,
    value,
    onChange
  }: {
    options: Array<{ label: React.ReactNode; value: string }>
    value?: string
    onChange?: (value: string) => void
  }) => (
    <select
      aria-label="billing cycle"
      value={value}
      onChange={(event) => onChange?.(event.currentTarget.value)}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {typeof option.label === "string" ? option.label : option.value}
        </option>
      ))}
    </select>
  ),
  Select: ({
    options,
    value,
    onChange,
    placeholder,
    disabled
  }: {
    options?: Array<{ value: string; label: React.ReactNode }>
    value?: string
    onChange?: (value: string) => void
    placeholder?: string
    disabled?: boolean
  }) => (
    <select
      aria-label={placeholder ?? "select"}
      value={value}
      disabled={disabled}
      onChange={(event) => onChange?.(event.currentTarget.value)}
    >
      {options?.map((option) => (
        <option key={option.value} value={option.value}>
          {typeof option.label === "string" ? option.label : option.value}
        </option>
      ))}
    </select>
  ),
  Space: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
  Tag: ({ children }: { children?: React.ReactNode }) => <span>{children}</span>
}))

const t = ((key: string, fallbackOrOptions?: string | { defaultValue?: string }) => {
  if (typeof fallbackOrOptions === "string") return fallbackOrOptions
  if (fallbackOrOptions?.defaultValue) return fallbackOrOptions.defaultValue
  return key
}) as TFunction

const createBillingProps = (
  overrides: Partial<TldwBillingSettingsProps> = {}
): TldwBillingSettingsProps => ({
  t,
  billingLoading: false,
  billingError: null,
  billingPlansError: null,
  billingStatusError: null,
  billingUsageError: null,
  billingPlans: [
    {
      name: "pro",
      display_name: "Pro",
      price_usd_monthly: 19,
      price_usd_yearly: 190
    }
  ],
  billingStatus: null,
  billingUsage: null,
  billingInvoices: [],
  billingInvoicesTotal: 0,
  billingInvoicesLoading: false,
  billingInvoicesError: null,
  billingActionLoading: false,
  selectedPlan: "pro",
  setSelectedPlan: vi.fn(),
  billingCycle: "monthly",
  setBillingCycle: vi.fn(),
  onLoadBilling: vi.fn(),
  onLoadInvoices: vi.fn(),
  onCheckout: vi.fn(),
  onBillingPortal: vi.fn(),
  onCancelSubscription: vi.fn(),
  onResumeSubscription: vi.fn(),
  ...overrides
})

const expectDesignSystemAlert = (text: string | RegExp) => {
  const node =
    typeof text === "string"
      ? screen.getByText(text, { exact: false })
      : screen.getByText(text)
  expect(node.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
}

describe("TldwBillingSettings design-system alerts", () => {
  it("renders billing load errors through design-system alerts", () => {
    render(
      <TldwBillingSettings
        {...createBillingProps({
          billingError: "Billing service is offline",
          billingStatusError: "Subscription request failed",
          billingPlansError: "Plan request failed",
          billingUsageError: "Usage request failed",
          billingInvoicesError: "Invoice request failed"
        })}
      />
    )

    expectDesignSystemAlert("Billing unavailable")
    expectDesignSystemAlert("Billing service is offline")
    expectDesignSystemAlert("Unable to load subscription")
    expectDesignSystemAlert("Subscription request failed")
    expectDesignSystemAlert("Unable to load plans")
    expectDesignSystemAlert("Plan request failed")
    expectDesignSystemAlert("Unable to load usage data")
    expectDesignSystemAlert("Usage request failed")
    expectDesignSystemAlert("Unable to load invoices")
    expectDesignSystemAlert("Invoice request failed")
  })

  it("renders subscription cancellation warnings through design-system alerts", () => {
    render(
      <TldwBillingSettings
        {...createBillingProps({
          billingStatus: {
            plan_name: "pro",
            plan_display_name: "Pro",
            status: "active",
            billing_cycle: "monthly",
            current_period_end: "2026-06-30T00:00:00Z",
            cancel_at_period_end: true
          }
        })}
      />
    )

    expectDesignSystemAlert("Subscription will cancel at period end.")
  })

  it("renders usage limit states through design-system alerts", () => {
    const { rerender } = render(
      <TldwBillingSettings
        {...createBillingProps({
          billingUsage: {
            has_exceeded: true,
            limit_checks: {
              api_calls_day: {
                current: 120,
                limit: 100,
                exceeded: true
              }
            }
          }
        })}
      />
    )

    expectDesignSystemAlert("Usage has exceeded one or more plan limits.")
    expectDesignSystemAlert("View upgrade options")

    rerender(
      <TldwBillingSettings
        {...createBillingProps({
          billingUsage: {
            has_warnings: true,
            limit_checks: {
              llm_tokens_month: {
                current: 900,
                limit: 1000,
                warning: true
              }
            }
          }
        })}
      />
    )

    expectDesignSystemAlert("Approaching plan limits for some resources.")
    expectDesignSystemAlert(/usage is nearing the limit/)
  })
})
