// @vitest-environment jsdom
//
// Stage B-S3: the API-key user selector must remote-search instead of
// preloading 100 accounts (WatchlistsOversightPage pattern: showSearch +
// filterOption={false} + 300ms debounce + out-of-order guard).

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

  // Simplified Select mirroring the design-system test's mock, extended with
  // an onSearch input so the remote-search wiring is exercisable.
  const Select = ({
    options = [],
    value,
    onChange,
    onSearch,
    placeholder,
    loading
  }: {
    options?: Array<{ value: number; label: string }>
    value?: number | null
    onChange?: (value: number) => void
    onSearch?: (term: string) => void
    placeholder?: string
    loading?: boolean
  }) => (
    <div>
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
      <input
        aria-label="Search users input"
        placeholder="Type to search"
        onChange={(event) => onSearch?.(event.target.value)}
      />
    </div>
  )

  const Modal = ({
    children,
    confirmLoading,
    okText,
    onCancel,
    onOk,
    open,
    title
  }: {
    children?: React.ReactNode
    confirmLoading?: boolean
    okText?: React.ReactNode
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
          {okText ?? "OK"}
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

const firstTwentyUsers = () => ({
  users: Array.from({ length: 20 }, (_, i) => ({
    id: i + 1,
    username: `user_${i + 1}`,
    email: `user_${i + 1}@example.com`
  }))
})

describe("ApiKeyManagementPage remote user search (admin perf B-S3)", () => {
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

    apiMock.listAdminUsers.mockImplementation(async (params?: { search?: string }) => {
      if (params?.search) {
        return {
          users: [{ id: 99, username: params.search, email: `${params.search}@example.com` }]
        }
      }
      return firstTwentyUsers()
    })
    apiMock.listUserApiKeys.mockResolvedValue([])
    apiMock.createUserApiKey.mockResolvedValue({ key: "sk-test" })
    apiMock.revokeUserApiKey.mockResolvedValue({})
    apiMock.rotateUserApiKey.mockResolvedValue({})
  })

  const advanceDebounce = () => new Promise((resolve) => setTimeout(resolve, 350))

  it("test_no_initial_100_user_preload: mount loads the first 20 users, not a 100-user preload", async () => {
    render(<ApiKeyManagementPage />)

    await screen.findByText("API Key Management")

    await waitFor(() => {
      expect(apiMock.listAdminUsers).toHaveBeenCalled()
    })
    expect(apiMock.listAdminUsers).toHaveBeenCalledWith({ limit: 20 })
    apiMock.listAdminUsers.mock.calls.forEach((call) => {
      expect(call[0]?.limit).not.toBe(100)
    })
  })

  it("test_apikey_user_select_remote_search: debounced search hits listAdminUsers({search, limit:20}), replaces options, and the seq guard drops stale responses", async () => {
    let resolveSlowSearch: (value: { users: Array<{ id: number; username: string; email: string }> }) => void = () => {}
    const slowSearch = new Promise<{ users: Array<{ id: number; username: string; email: string }> }>((resolve) => {
      resolveSlowSearch = resolve
    })

    apiMock.listAdminUsers.mockImplementation(async (params?: { search?: string }) => {
      if (params?.search === "ad") {
        // First keystroke burst: hangs until manually resolved.
        return slowSearch
      }
      if (params?.search === "ada") {
        return { users: [{ id: 99, username: "ada", email: "ada@example.com" }] }
      }
      return firstTwentyUsers()
    })

    render(<ApiKeyManagementPage />)

    // Initial options come from the limit-20 preload.
    expect(await screen.findByText("user_1 (user_1@example.com)")).toBeInTheDocument()

    // Typing triggers the debounced remote search.
    fireEvent.change(screen.getByLabelText("Search users input"), {
      target: { value: "ad" }
    })
    await advanceDebounce()
    await waitFor(() => {
      expect(apiMock.listAdminUsers).toHaveBeenCalledWith({
        search: "ad",
        limit: 20
      })
    })

    // Second burst resolves fast while the first is still in flight.
    fireEvent.change(screen.getByLabelText("Search users input"), {
      target: { value: "ada" }
    })
    await advanceDebounce()

    expect(await screen.findByText("ada (ada@example.com)")).toBeInTheDocument()
    expect(apiMock.listAdminUsers).toHaveBeenLastCalledWith({
      search: "ada",
      limit: 20
    })

    // The slow "ad" response lands late and must NOT overwrite ada's options.
    resolveSlowSearch({
      users: [{ id: 42, username: "stale_ad", email: "stale@example.com" }]
    })
    await new Promise((resolve) => setTimeout(resolve, 50))

    expect(screen.queryByText("stale_ad (stale@example.com)")).not.toBeInTheDocument()
    expect(screen.getByText("ada (ada@example.com)")).toBeInTheDocument()
  })
})
