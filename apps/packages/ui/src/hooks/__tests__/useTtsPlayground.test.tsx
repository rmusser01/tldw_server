// @vitest-environment jsdom

import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const { notificationErrorMock, resolveTtsProviderContextMock } = vi.hoisted(() => ({
  notificationErrorMock: vi.fn(),
  resolveTtsProviderContextMock: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback: string) => fallback
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    error: notificationErrorMock,
    info: vi.fn(),
    warning: vi.fn()
  })
}))

vi.mock("@/services/tts-provider", () => ({
  resolveTtsProviderContext: resolveTtsProviderContextMock
}))

vi.mock("@/utils/tts", () => ({
  splitMessageContent: (text: string) => [text]
}))

import { useTtsPlayground } from "../useTtsPlayground"

describe("useTtsPlayground", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resolveTtsProviderContextMock.mockResolvedValue({
      provider: "tldw",
      utterance: "hello",
      supported: true,
      playbackSpeed: 1,
      synthesize: vi.fn(async () => {
        throw new Error("Request failed for sk_secret_inline")
      })
    })
  })

  it("classifies generation errors without exposing raw provider details", async () => {
    const { result } = renderHook(() => useTtsPlayground())

    await act(async () => {
      await result.current.generateSegments("hello")
    })

    expect(notificationErrorMock).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "Credentials need attention",
        description: expect.stringContaining("Settings -> Speech")
      })
    )
    expect(JSON.stringify(notificationErrorMock.mock.calls)).not.toContain(
      "sk_secret_inline"
    )
  })
})
