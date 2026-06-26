// @vitest-environment jsdom

import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import type { TldwTtsProvidersInfo } from "@/services/tldw/audio-providers"
import { VoiceCloningManager } from "../VoiceCloningManager"

const mocks = vi.hoisted(() => ({
  listCustomVoices: vi.fn(),
  uploadCustomVoice: vi.fn(),
  encodeCustomVoice: vi.fn(),
  deleteCustomVoice: vi.fn(),
  synthesizeSpeech: vi.fn(),
  notificationError: vi.fn(),
  notificationSuccess: vi.fn()
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  return {
    ...actual,
    notification: {
      ...actual.notification,
      error: (...args: unknown[]) => mocks.notificationError(...args),
      success: (...args: unknown[]) => mocks.notificationSuccess(...args)
    }
  }
})

vi.mock("@/services/tldw/voice-cloning", async () => {
  const actual = await vi.importActual<typeof import("@/services/tldw/voice-cloning")>(
    "@/services/tldw/voice-cloning"
  )
  return {
    ...actual,
    listCustomVoices: (...args: unknown[]) => mocks.listCustomVoices(...args),
    uploadCustomVoice: (...args: unknown[]) => mocks.uploadCustomVoice(...args),
    encodeCustomVoice: (...args: unknown[]) => mocks.encodeCustomVoice(...args),
    deleteCustomVoice: (...args: unknown[]) => mocks.deleteCustomVoice(...args)
  }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    synthesizeSpeech: (...args: unknown[]) => mocks.synthesizeSpeech(...args)
  }
}))

const providersInfo: TldwTtsProvidersInfo = {
  providers: {
    chatterbox: {
      provider_name: "Chatterbox",
      supports_voice_cloning: true
    }
  },
  voices: {}
}

const customVoice = {
  voice_id: "voice-1",
  name: "Lab voice",
  provider: "chatterbox",
  duration: 2.4,
  format: "wav",
  size_bytes: 2048
}

const rawServerError = (method: string, path: string, action: string) =>
  new Error(
    `Request failed: 500 (${method} ${path}) token=sk_secret_${action} /Users/alice/private/${action}.json`
  )

const renderVoiceCloningManager = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false
      },
      mutations: {
        retry: false
      }
    }
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <VoiceCloningManager providersInfo={providersInfo} />
    </QueryClientProvider>
  )
}

const expectSanitizedDescription = () => {
  const payload = mocks.notificationError.mock.calls.at(-1)?.[0] as {
    description?: string
  }
  expect(payload.description).toContain("[server-endpoint]")
  expect(payload.description).toContain("[redacted-path]")
  expect(payload.description).toContain("[redacted-secret]")
  expect(payload.description).not.toContain("/api/v1")
  expect(payload.description).not.toContain("/Users/alice")
  expect(payload.description).not.toContain("sk_secret")
}

describe("VoiceCloningManager", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.listCustomVoices.mockResolvedValue([customVoice])
    mocks.uploadCustomVoice.mockResolvedValue({
      voice_id: "new-voice",
      name: "New voice"
    })
    mocks.encodeCustomVoice.mockResolvedValue({
      voice_id: customVoice.voice_id,
      provider: customVoice.provider
    })
    mocks.deleteCustomVoice.mockResolvedValue(undefined)
    mocks.synthesizeSpeech.mockResolvedValue(new ArrayBuffer(8))
  })

  it("sanitizes upload failure notifications", async () => {
    const user = userEvent.setup()
    const { container } = renderVoiceCloningManager()
    mocks.uploadCustomVoice.mockRejectedValueOnce(
      rawServerError("POST", "/api/v1/audio/voices/upload", "upload")
    )

    await user.type(screen.getByPlaceholderText("Researcher voice"), "Research voice")
    const fileInput = container.querySelector('input[type="file"]')
    expect(fileInput).toBeInstanceOf(HTMLInputElement)
    fireEvent.change(fileInput as HTMLInputElement, {
      target: {
        files: [new File(["voice"], "voice.wav", { type: "audio/wav" })]
      }
    })

    await user.click(screen.getByRole("button", { name: /upload voice/i }))

    await waitFor(() => {
      expect(mocks.notificationError).toHaveBeenCalledWith(
        expect.objectContaining({
          message: "Voice upload failed"
        })
      )
    })
    expectSanitizedDescription()
  })

  it("sanitizes existing voice action failure notifications", async () => {
    const user = userEvent.setup()
    const { container } = renderVoiceCloningManager()

    await screen.findByText("Lab voice")

    mocks.encodeCustomVoice.mockRejectedValueOnce(
      rawServerError("POST", "/api/v1/audio/voices/encode", "encode")
    )
    await user.click(screen.getByRole("button", { name: /encode/i }))
    await waitFor(() => {
      expect(mocks.notificationError).toHaveBeenCalledWith(
        expect.objectContaining({
          message: "Voice encoding failed"
        })
      )
    })
    expectSanitizedDescription()

    mocks.notificationError.mockClear()
    mocks.synthesizeSpeech.mockRejectedValueOnce(
      rawServerError("POST", "/api/v1/audio/speech", "preview")
    )
    await user.click(screen.getByRole("button", { name: /preview/i }))
    await waitFor(() => {
      expect(mocks.notificationError).toHaveBeenCalledWith(
        expect.objectContaining({
          message: "Preview failed"
        })
      )
    })
    expectSanitizedDescription()

    mocks.notificationError.mockClear()
    mocks.deleteCustomVoice.mockRejectedValueOnce(
      rawServerError("DELETE", "/api/v1/audio/voices/voice-1", "delete")
    )
    const deleteButton = container.querySelector(".ant-list-item-action .ant-btn-dangerous")
    expect(deleteButton).toBeInstanceOf(HTMLElement)
    await user.click(deleteButton as HTMLElement)
    await user.click(await screen.findByRole("button", { name: "OK" }))
    await waitFor(() => {
      expect(mocks.notificationError).toHaveBeenCalledWith(
        expect.objectContaining({
          message: "Delete failed"
        })
      )
    })
    expectSanitizedDescription()
  })
})
