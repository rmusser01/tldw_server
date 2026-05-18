import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { DataManagementSettings } from "../system-settings"

const mocks = vi.hoisted(() => ({
  clearDB: vi.fn(),
  clearChat: vi.fn(),
  invalidateQueries: vi.fn(),
  mutate: vi.fn(),
  notificationDestroy: vi.fn(),
  notificationError: vi.fn(),
  notificationInfo: vi.fn(),
  notificationSuccess: vi.fn(),
  storageClear: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      return fallbackOrOptions?.defaultValue ?? key
    }
  })
}))

vi.mock("@tanstack/react-query", () => ({
  useMutation: () => ({
    isPending: false,
    mutate: mocks.mutate
  }),
  useQueryClient: () => ({
    invalidateQueries: mocks.invalidateQueries
  })
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    clearChat: mocks.clearChat
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    destroy: mocks.notificationDestroy,
    error: mocks.notificationError,
    info: mocks.notificationInfo,
    success: mocks.notificationSuccess
  })
}))

vi.mock("@/db/dexie/chat", () => ({
  PageAssistDatabase: vi.fn().mockImplementation(function () {
    return {
      clearDB: mocks.clearDB
    }
  })
}))

vi.mock("@/utils/is-private-mode", () => ({
  isFireFox: false,
  isFireFoxPrivateMode: false
}))

describe("DataManagementSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.clearDB.mockResolvedValue(undefined)
    mocks.storageClear.mockResolvedValue(undefined)
    vi.stubGlobal("browser", {
      storage: {
        local: { clear: mocks.storageClear },
        session: { clear: mocks.storageClear },
        sync: { clear: mocks.storageClear }
      }
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it("keeps import, export, and typed-reset actions on the data surface", () => {
    render(<DataManagementSettings />)

    expect(
      screen.getByRole("heading", { name: /data management/i })
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /export data/i })).toBeInTheDocument()
    expect(screen.getByText(/import data/i)).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /reset all/i })).toBeInTheDocument()
  })

  it("opens the hidden import file input from a keyboard-focusable button", () => {
    const clickSpy = vi.spyOn(HTMLInputElement.prototype, "click")

    render(<DataManagementSettings />)

    fireEvent.click(screen.getByRole("button", { name: /import data/i }))

    expect(clickSpy).toHaveBeenCalled()
  })

  it("clears the scheduled reset reload when the data surface unmounts", async () => {
    vi.useFakeTimers()
    const setTimeoutSpy = vi.spyOn(globalThis, "setTimeout")
    const clearTimeoutSpy = vi.spyOn(globalThis, "clearTimeout")

    const { unmount } = render(<DataManagementSettings />)

    fireEvent.click(screen.getByRole("button", { name: /reset all/i }))
    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "RESET" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Reset" }))

    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()

    expect(mocks.clearDB).toHaveBeenCalledTimes(1)

    const reloadTimerCallIndex = setTimeoutSpy.mock.calls.findIndex(
      ([, delay]) => delay === 1500
    )
    expect(reloadTimerCallIndex).toBeGreaterThanOrEqual(0)
    const reloadTimer =
      setTimeoutSpy.mock.results[reloadTimerCallIndex]?.value

    unmount()

    expect(clearTimeoutSpy).toHaveBeenCalledWith(reloadTimer)
  })
})
