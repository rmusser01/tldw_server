import { fireEvent, render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { SetupRecoverySettings } from "../setup-recovery-settings"

const mocks = vi.hoisted(() => ({
  modalError: vi.fn(),
  modalConfirm: vi.fn(),
  navigate: vi.fn(),
  restartOnboarding: vi.fn(async () => undefined)
}))

vi.mock("antd", () => ({
  Modal: {
    error: mocks.modalError,
    confirm: mocks.modalConfirm
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? _key
  })
}))

vi.mock("react-router-dom", async () => {
  const actual =
    await vi.importActual<typeof import("react-router-dom")>(
      "react-router-dom"
    )

  return {
    ...actual,
    useNavigate: () => mocks.navigate
  }
})

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionActions: () => ({ restartOnboarding: mocks.restartOnboarding }),
  useConnectionState: () => ({
    isConnected: false,
    knowledgeStatus: "unknown",
    serverUrl: "http://127.0.0.1:8000"
  }),
  useConnectionUxState: () => ({
    errorKind: "auth",
    isChecking: false,
    uxState: "error_auth"
  })
}))

vi.mock("@/hooks/chat/useSelectedModel", () => ({
  useSelectedModel: () => ({
    selectedModel: "openai/gpt-4o-mini",
    selectedModelIsLoading: false
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, fallback: unknown) =>
    key === "defaultEmbeddingModel"
      ? ["openai/text-embedding-3-small", vi.fn()]
      : [fallback, vi.fn()]
}))

describe("SetupRecoverySettings", () => {
  beforeEach(() => {
    mocks.modalError.mockClear()
    mocks.modalConfirm.mockClear()
    mocks.navigate.mockClear()
    mocks.restartOnboarding.mockClear()
    mocks.restartOnboarding.mockResolvedValue(undefined)
  })

  it("shows recovery rows with specialist links and no raw diagnostics payload", () => {
    render(
      <MemoryRouter>
        <SetupRecoverySettings />
      </MemoryRouter>
    )

    expect(
      screen.getByRole("heading", { level: 2, name: "Setup & Recovery" })
    ).toBeInTheDocument()
    expect(screen.getByRole("link", { name: /fix auth/i })).toHaveAttribute(
      "href",
      "/settings/tldw"
    )
    expect(
      screen.getByRole("link", { name: /model settings/i })
    ).toHaveAttribute("href", "/settings/model")
    expect(
      screen.getByRole("link", { name: /embedding defaults/i })
    ).toHaveAttribute("href", "/settings/rag")
    expect(screen.queryByText(/\{/)).not.toBeInTheDocument()
  })

  it("confirms and redirects when restarting onboarding", async () => {
    render(
      <MemoryRouter>
        <SetupRecoverySettings />
      </MemoryRouter>
    )

    fireEvent.click(
      screen.getByRole("button", { name: /restart onboarding/i })
    )

    expect(mocks.modalConfirm).toHaveBeenCalledTimes(1)

    const confirmOptions = mocks.modalConfirm.mock.calls[0]?.[0] as {
      onOk?: () => Promise<void> | void
      title?: string
    }

    expect(confirmOptions.title).toBe("Restart onboarding?")
    await confirmOptions.onOk?.()

    expect(mocks.restartOnboarding).toHaveBeenCalledTimes(1)
    expect(mocks.navigate).toHaveBeenCalledWith("/")
  })

  it("shows an error and stays put when restarting onboarding fails", async () => {
    mocks.restartOnboarding.mockRejectedValueOnce(new Error("failed"))

    render(
      <MemoryRouter>
        <SetupRecoverySettings />
      </MemoryRouter>
    )

    fireEvent.click(
      screen.getByRole("button", { name: /restart onboarding/i })
    )

    const confirmOptions = mocks.modalConfirm.mock.calls[0]?.[0] as {
      onOk?: () => Promise<void> | void
    }

    await confirmOptions.onOk?.()

    expect(mocks.navigate).not.toHaveBeenCalled()
    expect(mocks.modalError).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Restart failed"
      })
    )
  })
})
