import { render, screen } from "@testing-library/react"
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest"

import { expectInsideDesignSystemAlertAsync } from "@/test-utils/designSystemAlert"
import { ProviderKeysSettings } from "../ProviderKeysSettings"

const { listUserProviderKeysMock, translate } = vi.hoisted(() => ({
  listUserProviderKeysMock: vi.fn(),
  translate: (
    key: string,
    fallbackOrOptions?: string | { defaultValue?: string }
  ) => {
    if (typeof fallbackOrOptions === "string") return fallbackOrOptions
    return fallbackOrOptions?.defaultValue ?? key
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: translate
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    listUserProviderKeys: listUserProviderKeysMock,
    upsertUserProviderKey: vi.fn(),
    deleteUserProviderKey: vi.fn()
  }
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  return {
    ...actual,
    Modal: {
      ...actual.Modal,
      confirm: vi.fn()
    }
  }
})

describe("ProviderKeysSettings design-system alerts", () => {
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
  })

  it("renders BYOK unavailable guidance through the design-system Alert primitive", async () => {
    listUserProviderKeysMock.mockRejectedValue({ status: 403 })

    render(<ProviderKeysSettings />)

    const alert = await expectInsideDesignSystemAlertAsync(
      "Provider key management is not available"
    )
    expect(alert).toHaveAttribute("role", "status")
    expect(
      screen.getByText(
        "Set BYOK_ENCRYPTION_KEY in your server's .env file to enable user-managed provider keys. Docker users: this is auto-generated on first run."
      )
    ).toBeInTheDocument()
  })

  it("renders provider-key load failures through the design-system Alert primitive", async () => {
    listUserProviderKeysMock.mockRejectedValue(new Error("network unavailable"))

    render(<ProviderKeysSettings />)

    const alert = await expectInsideDesignSystemAlertAsync(
      "Failed to load provider keys"
    )
    expect(alert).toHaveAttribute("role", "alert")
  })
})
