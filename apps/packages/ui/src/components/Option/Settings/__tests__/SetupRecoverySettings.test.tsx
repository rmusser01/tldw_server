import { render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"

import { SetupRecoverySettings } from "../setup-recovery-settings"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? _key
  })
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionActions: () => ({ restartOnboarding: vi.fn() }),
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
  it("shows recovery rows with specialist links and no raw diagnostics payload", () => {
    render(
      <MemoryRouter>
        <SetupRecoverySettings />
      </MemoryRouter>
    )

    expect(
      screen.getByRole("heading", { name: "Setup & Recovery" })
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
})
