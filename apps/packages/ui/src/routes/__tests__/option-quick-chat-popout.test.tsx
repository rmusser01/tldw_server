import React from "react"
import { render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"

import QuickChatPopout from "../option-quick-chat-popout"

vi.mock("antd", () => ({
  Select: ({
    "aria-label": ariaLabel,
    options = [],
  }: {
    "aria-label"?: string
    options?: Array<{ value: string; label: string }>
  }) => (
    <select aria-label={ariaLabel}>
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  ),
  Segmented: ({
    options = [],
    value,
    onChange,
  }: {
    options?: Array<{ value: string; label: string }>
    value?: string
    onChange?: (value: string) => void
  }) => (
    <div role="group" aria-label="Quick chat mode">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={value === option.value}
          onClick={() => onChange?.(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  ),
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: ({ select }: { select?: (data: Array<{ model: string }>) => unknown }) => {
    const models = [{ model: "local:test-model" }]
    return {
      data: select ? select(models) : models,
      isLoading: false,
      isError: false,
    }
  },
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key,
  }),
}))

vi.mock("@/hooks/useQuickChat", () => ({
  useQuickChat: () => ({
    messages: [],
    sendMessage: vi.fn(),
    cancelStream: vi.fn(),
    isStreaming: false,
    hasModel: true,
    activeModel: "local:test-model",
    currentModel: "local:test-model",
    modelOverride: null,
    setModelOverride: vi.fn(),
  }),
}))

vi.mock("@/store/quick-chat", () => {
  const store = {
    assistantMode: "chat",
    setAssistantMode: vi.fn(),
  }
  const useQuickChatStore = () => store
  useQuickChatStore.getState = () => ({
    restoreFromState: vi.fn(),
    clearMessages: vi.fn(),
  })
  return { useQuickChatStore }
})

vi.mock("@/services/tldw-server", () => ({
  fetchChatModels: vi.fn(),
}))

vi.mock("@/hooks/useChatModelsSelect", () => ({
  useChatModelsSelect: () => ({
    allowClear: true,
    modelOptions: [{ value: "local:test-model", label: "Local test model" }],
    modelPlaceholder: "Select model",
    handleModelChange: vi.fn(),
  }),
}))

vi.mock("@/components/Common/QuickChatHelper/QuickChatInput", () => ({
  QuickChatInput: () => <textarea aria-label="Quick chat input" />,
}))

vi.mock("@/components/Common/QuickChatHelper/QuickChatMessage", () => ({
  QuickChatMessage: () => <div data-testid="quick-chat-message" />,
}))

vi.mock("@/components/Common/QuickChatHelper/QuickChatGuidesPanel", () => ({
  QuickChatGuidesPanel: () => <div data-testid="quick-chat-guides" />,
}))

describe("QuickChatPopout route identity", () => {
  it("presents quick chat as a helper surface", () => {
    render(
      <MemoryRouter initialEntries={["/quick-chat-popout"]}>
        <QuickChatPopout />
      </MemoryRouter>
    )

    expect(
      screen.getByRole("heading", { name: "Quick Chat Helper" })
    ).toBeInTheDocument()
    expect(screen.getByLabelText("Model")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Chat" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Docs Q&A" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Browse Guides" })).toBeInTheDocument()
    expect(
      screen.getByText("Start a quick side chat to keep your main thread clean.")
    ).toBeInTheDocument()
  })
})
