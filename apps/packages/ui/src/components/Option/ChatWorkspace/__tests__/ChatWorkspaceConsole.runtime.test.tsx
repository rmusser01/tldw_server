import { render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ChatWorkspaceConsole } from "../ChatWorkspaceConsole"
import type { StagedWorkspaceSource } from "../types"

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    messages: [],
    history: [],
    streaming: false,
    isLoading: false,
    isProcessing: false,
    selectedModel: "global-model",
    selectedAssistant: null,
    selectedAssistantSource: "none",
    serverChatId: null,
    serverChatAssistantKind: null,
    serverChatAssistantId: null,
    serverChatLoadState: "idle",
    serverChatLoadError: null,
    onSubmit: vi.fn(),
    setMessages: vi.fn(),
    stopStreamingRequest: vi.fn()
  })
}))
vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: () => ({ apiProvider: "openai" })
}))
vi.mock("@/hooks/useSetting", () => ({
  useSetting: (setting: { defaultValue: unknown }) => [setting.defaultValue]
}))
vi.mock("../MacroRunDetailDrawer", () => ({ MacroRunDetailDrawer: () => null }))
vi.mock("../../ResearchWorkspace/SourcesPane/WorkspaceSourcePreview", () => ({
  WorkspaceSourcePreview: () => null
}))

const props = {
  workspaceId: "workspace-1",
  workspaceReady: true,
  workspaceName: "Workspace",
  sources: [],
  browsedSourceId: null,
  stagedSources: [],
  selectedModelLabel: "global-model",
  hasModelSelected: true,
  selectedPersonaLabel: null,
  assistantSource: "none" as const,
  backendAvailable: true,
  chatBackendAvailable: true,
  streaming: false,
  onBrowseSource: vi.fn(),
  onStageSources: vi.fn(),
  onUnstageSource: vi.fn(),
  onClearStagedSources: vi.fn(),
  onRuntimeStateChange: vi.fn()
}

describe("ChatWorkspaceConsole runtime wiring", () => {
  it.each([
    [{ connectionMode: "demo" }, "Demo mode - not live"],
    [{ connectionMode: "bypass" }, "Offline bypass - not verified"],
    [{ historyLoading: true }, "Loading chat history"],
    [{ historyLoadError: "History unavailable" }, "Chat history unavailable"],
    [{ sending: true }, "Sending"]
  ] as const)("forwards %j to both runtime surfaces", (runtime, label) => {
    render(<ChatWorkspaceConsole {...props} {...runtime} />)
    expect(within(screen.getByLabelText("Chat workspace status")).getByRole("status")).toHaveTextContent(label)
    expect(within(screen.getByRole("complementary", { name: "Chat workspace inspector" })).getByText(label)).toBeInTheDocument()
    expect(screen.queryByText("Ready")).not.toBeInTheDocument()
  })

  it("keeps Send disabled until hydration completes even with a workspace identity", () => {
    const stagedSources: StagedWorkspaceSource[] = [{
      sourceId: "source-1", mediaId: 1, title: "Notes", type: "document",
      scopeLabel: "Workspace", availability: "ready" as const
    }]
    const view = render(<ChatWorkspaceConsole {...props} stagedSources={stagedSources} workspaceReady={false} />)
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled()
    view.rerender(<ChatWorkspaceConsole {...props} stagedSources={stagedSources} />)
    expect(screen.getByRole("button", { name: "Send message" })).toBeEnabled()
  })
})
