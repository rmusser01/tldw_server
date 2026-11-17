// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { usePlaygroundRawPreview } from "../hooks/usePlaygroundRawPreview"
import {
  captureChatRequestDebugSnapshot,
  getLastChatRequestDebugSnapshot,
  setChatRequestDebugEnabled
} from "@/services/tldw/chat-request-debug"

vi.mock("@/utils/resolve-api-provider", () => ({
  resolveApiProviderForModel: vi.fn(async () => "openai")
}))

const buildDeps = () => ({
  composerModels: [{ id: "tool-model", capabilities: ["tools"] }],
  selectedModel: "tool-model",
  compareModeActive: false,
  compareSelectedModels: [],
  compareMaxModels: 4,
  currentChatModelSettings: { apiProvider: "openai" },
  history: [],
  systemPrompt: undefined,
  hasMcp: false,
  mcpHealthState: "healthy",
  mcpTools: [],
  toolChoice: "auto",
  temporaryChat: true,
  serverChatId: null,
  serverChatState: null,
  serverChatSource: null,
  selectedCharacter: null,
  messageSteeringMode: "none",
  messageSteeringForceNarrate: false,
  ragMediaIds: null,
  selectedKnowledge: null,
  ragPinnedResults: [],
  fileRetrievalEnabled: false,
  contextFiles: [],
  documentContext: [],
  selectedDocuments: [],
  imageBackendDefaultTrimmed: "",
  resolveSubmissionIntent: (message: string) => ({
    message,
    isImageCommand: false
  }),
  formImage: "",
  formMessage: "hello",
  researchContext: undefined,
  notificationApi: { error: vi.fn(), success: vi.fn() },
  t: (_key: string, defaultValueOrOptions?: unknown) =>
    typeof defaultValueOrOptions === "string" ? defaultValueOrOptions : _key,
  setToolsPopoverOpen: vi.fn()
})

afterEach(() => {
  setChatRequestDebugEnabled(false)
})

describe("usePlaygroundRawPreview debug gate lifecycle (WP-22 round 1)", () => {
  it("enables the debug pipeline while the modal is open and disables it on close", async () => {
    setChatRequestDebugEnabled(false)
    const { result } = renderHook(() => usePlaygroundRawPreview(buildDeps()))

    // Before opening: capture is a no-op.
    captureChatRequestDebugSnapshot({
      endpoint: "/api/v1/chat/completions",
      method: "POST",
      mode: "non-stream",
      body: { model: "x" }
    })
    expect(getLastChatRequestDebugSnapshot()).toBeNull()

    // Opening the modal opts in; its own capture lands.
    act(() => {
      result.current.openRawRequestModal()
    })
    await waitFor(() => {
      expect(result.current.rawRequestSnapshot).not.toBeNull()
    })
    expect(getLastChatRequestDebugSnapshot()).not.toBeNull()

    // Closing the modal withdraws the opt-in: subsequent captures no-op again.
    act(() => {
      result.current.closeRawRequestModal()
    })
    expect(result.current.rawRequestModalOpen).toBe(false)
    captureChatRequestDebugSnapshot({
      endpoint: "/api/v1/chat/completions",
      method: "POST",
      mode: "non-stream",
      body: { model: "x" }
    })
    expect(getLastChatRequestDebugSnapshot()).toBeNull()
  })
})
