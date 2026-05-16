// @vitest-environment jsdom

import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { TFunction } from "i18next"

const {
  formItemSpy,
  modalConfirmMock
} = vi.hoisted(() => ({
  formItemSpy: vi.fn(),
  modalConfirmMock: vi.fn()
}))

vi.mock("@heroicons/react/24/outline", () => ({
  CheckIcon: () => <svg aria-hidden="true" />,
  XMarkIcon: () => <svg aria-hidden="true" />
}))

vi.mock("antd", () => {
  const Input = ({
    onChange,
    value,
    placeholder,
    type = "text"
  }: {
    onChange?: (event: { target: { value: string } }) => void
    value?: string | number
    placeholder?: string
    type?: string
  }) => (
    <input
      type={type}
      value={value}
      placeholder={placeholder}
      onChange={(event) =>
        onChange?.({ target: { value: event.currentTarget.value } })
      }
    />
  )

  Input.Password = ({
    onChange,
    value,
    placeholder
  }: {
    onChange?: (event: { target: { value: string } }) => void
    value?: string | number
    placeholder?: string
  }) => (
    <input
      type="password"
      value={value}
      placeholder={placeholder}
      onChange={(event) =>
        onChange?.({ target: { value: event.currentTarget.value } })
      }
    />
  )

  return {
    Alert: ({
      title,
      description
    }: {
      title?: React.ReactNode
      description?: React.ReactNode
    }) => (
      <div>
        {title}
        {description}
      </div>
    ),
    Button: ({
      children,
      onClick,
      disabled,
      htmlType
    }: {
      children?: React.ReactNode
      onClick?: () => void
      disabled?: boolean
      htmlType?: "button" | "submit" | "reset"
    }) => (
      <button type={htmlType ?? "button"} disabled={disabled} onClick={onClick}>
        {children}
      </button>
    ),
    Collapse: ({
      items
    }: {
      items?: Array<{ key: string; label: React.ReactNode; children: React.ReactNode }>
    }) => <div>{items?.map((item) => <div key={item.key}>{item.children}</div>)}</div>,
    Form: {
      Item: ({
        children,
        label,
        name,
        rules
      }: {
        children?: React.ReactNode
        label?: React.ReactNode
        name?: string
        rules?: unknown[]
      }) => {
        formItemSpy({ label, name, rules })
        const testId =
          typeof label === "string"
            ? `form-item-${label.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`
            : "form-item"
        return <div data-testid={testId}>{children}</div>
      }
    },
    Input,
    Modal: {
      confirm: modalConfirmMock
    },
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
        aria-label="segmented"
        value={value}
        onChange={(event) => onChange?.(event.currentTarget.value)}>
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
      placeholder
    }: {
      options?: Array<{ value: string; label: React.ReactNode }>
      value?: string
      onChange?: (value: string) => void
      placeholder?: string
    }) => (
      <select
        aria-label={placeholder ?? "select"}
        value={value}
        onChange={(event) => onChange?.(event.currentTarget.value)}>
        {options?.map((option) => (
          <option key={option.value} value={option.value}>
            {typeof option.label === "string" ? option.label : option.value}
          </option>
        ))}
      </select>
    ),
    Space: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
    Tag: ({ children }: { children?: React.ReactNode }) => <span>{children}</span>
  }
})

vi.mock("@/config/platform", () => ({
  isFirefoxTarget: () => false
}))

import { TldwBillingSettings } from "../TldwBillingSettings"
import { TldwConnectionSettings } from "../TldwConnectionSettings"
import { parseSeconds } from "../TldwTimeoutSettings"

const t = ((key: string, fallbackOrOptions?: string | { defaultValue?: string }) => {
  if (typeof fallbackOrOptions === "string") return fallbackOrOptions
  if (fallbackOrOptions?.defaultValue) return fallbackOrOptions.defaultValue
  return key
}) as TFunction

const createConnectionProps = (
  overrides: Partial<React.ComponentProps<typeof TldwConnectionSettings>> = {}
): React.ComponentProps<typeof TldwConnectionSettings> => ({
  t,
  form: {
    setFieldValue: vi.fn()
  } as any,
  authMode: "single-user",
  setAuthMode: vi.fn(),
  isLoggedIn: false,
  setIsLoggedIn: vi.fn(),
  loginMethod: "magic-link",
  setLoginMethod: vi.fn(),
  magicEmail: "persisted@example.com",
  setMagicEmail: vi.fn(),
  magicToken: "persisted-token",
  setMagicToken: vi.fn(),
  magicSent: true,
  setMagicSent: vi.fn(),
  magicSending: false,
  testingConnection: false,
  connectionStatus: null,
  connectionDetail: "",
  coreStatus: "unknown",
  ragStatus: "unknown",
  onTestConnection: vi.fn(),
  onLogin: vi.fn(),
  onSendMagicLink: vi.fn(),
  onVerifyMagicLink: vi.fn(),
  onLogout: vi.fn(),
  onGrantSiteAccess: vi.fn(),
  onOpenHealthDiagnostics: vi.fn(),
  ...overrides
})

const createBillingProps = (
  overrides: Partial<React.ComponentProps<typeof TldwBillingSettings>> = {}
): React.ComponentProps<typeof TldwBillingSettings> => ({
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

describe("settings PR review fixes", () => {
  beforeEach(() => {
    formItemSpy.mockClear()
    modalConfirmMock.mockReset()
    modalConfirmMock.mockImplementation(
      ({ onOk }: { onOk?: () => void }) => onOk?.()
    )
  })

  afterEach(() => {
    cleanup()
  })

  it("clears both password and magic-link credentials when auth mode changes", () => {
    const props = createConnectionProps()

    render(<TldwConnectionSettings {...props} />)

    fireEvent.change(screen.getByRole("combobox", { name: "segmented" }), {
      target: { value: "multi-user" }
    })

    expect(props.setAuthMode).toHaveBeenCalledWith("multi-user")
    expect(props.form.setFieldValue).toHaveBeenCalledWith("apiKey", "")
    expect(props.form.setFieldValue).toHaveBeenCalledWith("username", "")
    expect(props.form.setFieldValue).toHaveBeenCalledWith("password", "")
    expect(props.form.setFieldValue).toHaveBeenCalledWith("magicEmail", "")
    expect(props.form.setFieldValue).toHaveBeenCalledWith("magicToken", "")
    expect(props.setMagicEmail).toHaveBeenCalledWith("")
    expect(props.setMagicToken).toHaveBeenCalledWith("")
    expect(props.setMagicSent).toHaveBeenCalledWith(false)
    expect(props.setIsLoggedIn).toHaveBeenCalledWith(false)
  })

  it("does not register magic-link inputs as named Form fields", () => {
    render(
      <TldwConnectionSettings
        {...createConnectionProps({
          authMode: "multi-user",
          loginMethod: "magic-link"
        })}
      />
    )

    const emailField = formItemSpy.mock.calls
      .map(([props]) => props)
      .find((props) => props.label === "Email")
    const tokenField = formItemSpy.mock.calls
      .map(([props]) => props)
      .find((props) => props.label === "Magic link token")

    expect(emailField).toMatchObject({
      name: undefined,
      rules: undefined
    })
    expect(tokenField).toMatchObject({
      name: undefined,
      rules: undefined
    })
  })

  it("renders the login-required notice through the design-system alert primitive", () => {
    render(
      <TldwConnectionSettings
        {...createConnectionProps({
          authMode: "multi-user",
          isLoggedIn: false
        })}
      />
    )

    const loginRequiredTitle = screen.getByText(/Login Required/)
    expect(
      loginRequiredTitle.closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
  })

  it("renders the logged-in notice through the design-system alert primitive and preserves logout", () => {
    const onLogout = vi.fn()

    render(
      <TldwConnectionSettings
        {...createConnectionProps({
          authMode: "multi-user",
          isLoggedIn: true,
          onLogout
        })}
      />
    )

    const loggedInTitle = screen.getByText(/Logged In/)
    expect(
      loggedInTitle.closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Logout" }))
    expect(onLogout).toHaveBeenCalledTimes(1)
  })

  it("formats invoice amounts using the invoice currency instead of a hardcoded dollar prefix", () => {
    render(
      <TldwBillingSettings
        {...createBillingProps({
          billingInvoices: [
            {
              id: 42,
              amount_cents: 1250,
              currency: "eur",
              status: "succeeded",
              created_at: "2026-04-04T00:00:00Z"
            }
          ],
          billingInvoicesTotal: 1
        })}
      />
    )

    expect(
      screen.getByText(
        new Intl.NumberFormat(undefined, {
          style: "currency",
          currency: "EUR"
        }).format(12.5)
      )
    ).toBeInTheDocument()
  })

  it("clamps zero and negative timeout inputs to the minimum supported value", () => {
    expect(parseSeconds("0", 10)).toBe(1)
    expect(parseSeconds("-5", 10)).toBe(1)
    expect(parseSeconds("15", 10)).toBe(15)
  })
})
