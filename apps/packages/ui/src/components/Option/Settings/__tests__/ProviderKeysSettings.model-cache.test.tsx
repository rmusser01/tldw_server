import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { Modal } from "antd"
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest"

import { ProviderKeysSettings } from "../ProviderKeysSettings"

const mocks = vi.hoisted(() => ({
  listUserProviderKeys: vi.fn(),
  upsertUserProviderKey: vi.fn(),
  deleteUserProviderKey: vi.fn(),
  clearCache: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? key
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    listUserProviderKeys: mocks.listUserProviderKeys,
    upsertUserProviderKey: mocks.upsertUserProviderKey,
    deleteUserProviderKey: mocks.deleteUserProviderKey
  }
}))

vi.mock("@/services/tldw/TldwModels", () => ({
  tldwModels: { clearCache: mocks.clearCache }
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  return {
    ...actual,
    // A native select, so the test picks a provider without antd's virtual dropdown.
    Select: ({
      value,
      onChange,
      options
    }: {
      value?: string
      onChange: (value: string) => void
      options: Array<{ value: string }>
    }) => (
      <select aria-label="provider" value={value ?? ""} onChange={(event) => onChange(event.target.value)}>
        <option value="" />
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.value}
          </option>
        ))}
      </select>
    ),
    Modal: { ...actual.Modal, confirm: vi.fn() }
  }
})

describe("ProviderKeysSettings refreshes the model pickers (TASK-15583)", () => {
  beforeAll(() => {
    if (typeof window.matchMedia !== "function") {
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
    if (!(globalThis as any).ResizeObserver) {
      ;(globalThis as any).ResizeObserver = class ResizeObserver {
        observe() {}
        unobserve() {}
        disconnect() {}
      }
    }
  })

  beforeEach(() => {
    vi.clearAllMocks()
    mocks.listUserProviderKeys.mockResolvedValue({
      items: [
        {
          provider: "anthropic",
          has_key: true,
          source: "user",
          key_hint: "1234",
          auth_source: "api_key",
          last_used_at: null
        }
      ]
    })
    mocks.upsertUserProviderKey.mockResolvedValue({
      provider: "openai",
      status: "stored",
      key_hint: "abcd",
      updated_at: "2026-11-21T00:00:00Z"
    })
    mocks.deleteUserProviderKey.mockResolvedValue(undefined)
  })

  it("clears the model catalog after a key is saved, so the pickers offer its provider at once", async () => {
    render(<ProviderKeysSettings />)
    fireEvent.click(await screen.findByRole("button", { name: /Add Provider Key/ }))
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "openai" } })
    fireEvent.change(screen.getByPlaceholderText("sk-..."), { target: { value: "sk-users-own" } })
    fireEvent.click(screen.getByRole("button", { name: /Save Key/ }))

    await waitFor(() => expect(mocks.clearCache).toHaveBeenCalledTimes(1))
    expect(mocks.upsertUserProviderKey).toHaveBeenCalledWith("openai", "sk-users-own")
  })

  it("clears the model catalog after a key is deleted", async () => {
    const { container } = render(<ProviderKeysSettings />)
    await screen.findByText("anthropic")
    fireEvent.click(container.querySelector("button.ant-btn-dangerous") as HTMLButtonElement)
    const { onOk } = vi.mocked(Modal.confirm).mock.calls[0][0] as { onOk: () => Promise<void> }
    await act(async () => {
      await onOk()
    })

    expect([mocks.deleteUserProviderKey.mock.calls, mocks.clearCache.mock.calls.length]).toEqual([[["anthropic"]], 1])
  })
})
