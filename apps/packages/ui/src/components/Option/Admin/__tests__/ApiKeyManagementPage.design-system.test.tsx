// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import ApiKeyManagementPage from "../ApiKeyManagementPage"

const apiMock = vi.hoisted(() => ({
  listAdminUsers: vi.fn(),
  listUserApiKeys: vi.fn(),
  createUserApiKey: vi.fn(),
  revokeUserApiKey: vi.fn(),
  rotateUserApiKey: vi.fn()
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")

  const Select = ({
    options = [],
    value,
    onChange,
    placeholder,
    loading
  }: {
    options?: Array<{ value: number; label: string }>
    value?: number | null
    onChange?: (value: number) => void
    placeholder?: string
    loading?: boolean
  }) => (
    <select
      aria-label="Select User"
      disabled={loading}
      value={value ?? ""}
      onChange={(event) => onChange?.(Number(event.currentTarget.value))}
    >
      <option value="">{placeholder ?? "Select"}</option>
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  )

  const Modal = ({
    children,
    confirmLoading,
    onCancel,
    onOk,
    open,
    title
  }: {
    children?: React.ReactNode
    confirmLoading?: boolean
    onCancel?: () => void
    onOk?: () => void
    open?: boolean
    title?: string
  }) => {
    if (!open) return null

    return (
      <div role="dialog" aria-label={title}>
        {children}
        <button type="button" disabled={confirmLoading} onClick={onOk}>
          OK
        </button>
        <button type="button" onClick={onCancel}>
          Cancel
        </button>
      </div>
    )
  }

  return {
    ...actual,
    Modal,
    Select
  }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock
}))

const expectDesignSystemAlertForText = async (text: string) => {
  const title = await screen.findByText(text)
  const alert = title.closest('[data-ds-component="Alert"]')

  expect(alert).not.toBeNull()
  return alert as HTMLElement
}

const renderWithUser = async () => {
  render(<ApiKeyManagementPage />)

  await screen.findByText("API Key Management")
  fireEvent.change(await screen.findByLabelText("Select User"), {
    target: { value: "11" }
  })
}

describe("ApiKeyManagementPage design-system states", () => {
  beforeEach(() => {
    vi.clearAllMocks()

    if (!window.matchMedia) {
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: false,
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn()
        }))
      })
    }

    if (!(window as any).ResizeObserver) {
      ;(window as any).ResizeObserver = class {
        observe() {}
        unobserve() {}
        disconnect() {}
      }
    }

    apiMock.listAdminUsers.mockResolvedValue({
      users: [{ id: 11, username: "admin", email: "admin@example.com" }]
    })
    apiMock.listUserApiKeys.mockResolvedValue([])
    apiMock.createUserApiKey.mockResolvedValue({ key: "sk-test-secret" })
    apiMock.revokeUserApiKey.mockResolvedValue({})
    apiMock.rotateUserApiKey.mockResolvedValue({})
  })

  it("renders forbidden guard feedback through the design-system Alert primitive", async () => {
    apiMock.listAdminUsers.mockRejectedValueOnce({ status: 403 })

    render(<ApiKeyManagementPage />)

    const alert = await expectDesignSystemAlertForText("Access Denied")
    expect(alert).toHaveAttribute("role", "alert")
    expect(alert).toHaveTextContent("You don't have permission to manage API keys.")
  })

  it("renders missing-endpoint guard feedback through the design-system Alert primitive", async () => {
    apiMock.listAdminUsers.mockRejectedValueOnce({ status: 404 })

    render(<ApiKeyManagementPage />)

    const alert = await expectDesignSystemAlertForText("Not Available")
    expect(alert).toHaveAttribute("role", "alert")
    expect(alert).toHaveTextContent("API key management is not available on this server.")
  })

  it("renders key-load failures through the design-system Alert primitive", async () => {
    apiMock.listUserApiKeys.mockRejectedValueOnce(new Error("key API failed"))

    await renderWithUser()

    const alert = await expectDesignSystemAlertForText("key API failed")
    expect(alert).toHaveAttribute("role", "alert")
  })

  it("renders created-key success feedback through the design-system Alert primitive", async () => {
    await renderWithUser()

    fireEvent.click(await screen.findByRole("button", { name: "Create Key" }))
    fireEvent.change(screen.getByPlaceholderText("e.g. Production Key"), {
      target: { value: "Production Key" }
    })
    fireEvent.click(screen.getByRole("button", { name: "OK" }))

    await waitFor(() => {
      expect(apiMock.createUserApiKey).toHaveBeenCalledWith(11, {
        name: "Production Key",
        rate_limit: undefined
      })
    })

    const alert = await expectDesignSystemAlertForText("New API Key Created")
    expect(alert).toHaveAttribute("role", "status")
    expect(alert).toHaveTextContent("Copy this key now -- it will not be shown again:")
    expect(alert).toHaveTextContent("sk-test-secret")
  })
})
