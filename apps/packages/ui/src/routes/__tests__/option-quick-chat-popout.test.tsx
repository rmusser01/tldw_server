import React from "react"
import fs from "node:fs"
import path from "node:path"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import QuickChatPopout from "../option-quick-chat-popout"
import { fetchChatModels } from "@/services/tldw-server"

vi.mock("antd", () => ({
  Select: ({
    "aria-label": ariaLabel,
    options = [],
    disabled,
  }: {
    "aria-label"?: string
    disabled?: boolean
    options?: Array<{ value: string; label: string }>
  }) => (
    <select aria-label={ariaLabel} disabled={disabled}>
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

vi.mock("@/components/Common/QuickChatHelper/QuickChatInput", () => ({
  QuickChatInput: () => <textarea aria-label="Quick chat input" />,
}))

vi.mock("@/components/Common/QuickChatHelper/QuickChatMessage", () => ({
  QuickChatMessage: () => <div data-testid="quick-chat-message" />,
}))

vi.mock("@/components/Common/QuickChatHelper/QuickChatGuidesPanel", () => ({
  QuickChatGuidesPanel: () => <div data-testid="quick-chat-guides" />,
}))

const renderPopout = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}><MemoryRouter initialEntries={["/quick-chat-popout"]}><QuickChatPopout /></MemoryRouter></QueryClientProvider>);
}

describe("QuickChatPopout route identity", () => {
  beforeEach(() => { vi.mocked(fetchChatModels).mockReset().mockResolvedValue([{ model: "local:test-model", provider: "llama", nickname: "Local test model" }]) });
  it("recovers a failed mounted catalog through Retry", async () => {
    vi.mocked(fetchChatModels).mockRejectedValueOnce(new Error("Metadata unavailable"));
    renderPopout();
    await screen.findByText("Check your server connection, then try again.");
    expect(screen.getByLabelText("Model")).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    await screen.findByRole("option", { name: /Local test model/ });
    expect(screen.getByLabelText("Model")).toBeEnabled();
    expect(fetchChatModels).toHaveBeenCalledTimes(2);
  })
  it("presents quick chat as a helper surface", async () => {
    renderPopout()

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
    await waitFor(() => expect(fetchChatModels).toHaveBeenCalledWith({ requireFresh: true }))
  })

  it("keeps the packaged extension on the shared route and model service", () => {
    const extensionEntry = fs.readFileSync(
      path.resolve(
        __dirname,
        "../../../../../extension/entrypoints/options/main.tsx"
      ),
      "utf8"
    )
    const legacyExtensionRoute = fs.readFileSync(
      path.resolve(
        __dirname,
        "../../../../../tldw-frontend/extension/routes/option-quick-chat-popout.tsx"
      ),
      "utf8"
    )

    expect(extensionEntry).toContain('import "@tldw/ui/entries/options/main"')
    expect(legacyExtensionRoute).toContain(
      'import { fetchChatModels } from "@/services/tldw-server"'
    )
    expect(legacyExtensionRoute).not.toContain("getModels(")
  })
})
