import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { useMessageOption as useMessageOptionHook } from "@/hooks/useMessageOption"
import { WorkspaceChatPanel } from "../WorkspaceChatPanel"
import type { StagedWorkspaceSource } from "../types"
import type { EffectiveWorkspaceAssistantDefault } from "@/types/workspace"

type UseMessageOptionHook = typeof useMessageOptionHook
type UseMessageOptionState = ReturnType<UseMessageOptionHook>
type SubmitPayload = Parameters<UseMessageOptionState["onSubmit"]>[0]

const chatHookState = vi.hoisted(() => {
  const onSubmit = vi.fn<UseMessageOptionState["onSubmit"]>(
    async (): Promise<any> => ({ status: "submitted" })
  )
  const stopStreamingRequest = vi.fn()
  const value = {
    messages: [],
    onSubmit,
    streaming: false,
    isLoading: false,
    isProcessing: false,
    stopStreamingRequest,
    selectedModel: "gpt-test",
    selectedAssistant: { kind: "persona", id: "p1", name: "Analyst" },
    serverChatAssistantKind: null,
    serverChatAssistantId: null,
    serverChatMetaLoaded: false
  } as unknown as UseMessageOptionState
  const useMessageOption = vi.fn<UseMessageOptionHook>(() => value)

  return { onSubmit, stopStreamingRequest, useMessageOption, value }
})

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: (...args: Parameters<UseMessageOptionHook>) =>
    chatHookState.useMessageOption(...args)
}))

vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: (props: { conversationInstanceId: string; message: string }) => (
    <article
      data-testid="workspace-panel-message"
      data-conversation-instance-id={props.conversationInstanceId}
    >
      {props.message}
    </article>
  )
}))

const staged: StagedWorkspaceSource[] = [
  {
    sourceId: "source-1",
    mediaId: 101,
    title: "Operator Notes",
    type: "document",
    scopeLabel: "Default workspace",
    availability: "ready"
  }
]

const stagedWithoutReadyMedia: StagedWorkspaceSource[] = [
  {
    sourceId: "source-processing",
    mediaId: 202,
    title: "Indexing Notes",
    type: "document",
    scopeLabel: "Default workspace",
    availability: "processing"
  }
]

const stagedWithMixedAvailability: StagedWorkspaceSource[] = [
  staged[0],
  {
    sourceId: "source-processing",
    mediaId: 202,
    title: "Indexing Notes",
    type: "document",
    scopeLabel: "Default workspace",
    availability: "processing"
  }
]

const getSubmitPayload = (): SubmitPayload => {
  const payload = chatHookState.onSubmit.mock.calls[0]?.[0]
  if (!payload) {
    throw new Error("Expected workspace chat submit payload")
  }
  return payload
}

const availableWorkspaceDefault: EffectiveWorkspaceAssistantDefault = {
  status: "available",
  source: "workspace",
  assistantKind: "persona",
  assistantId: "workspace-persona",
  label: "Workspace Analyst",
  personaMemoryMode: "read_write",
  degradedReason: null
}

const unavailableWorkspaceDefault: EffectiveWorkspaceAssistantDefault = {
  status: "unavailable",
  source: "workspace",
  assistantKind: "persona",
  assistantId: "workspace-persona",
  label: "Workspace Analyst",
  personaMemoryMode: "read_write",
  degradedReason: "persona_deleted"
}

describe("WorkspaceChatPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    chatHookState.value.messages = []
    chatHookState.value.streaming = false
    chatHookState.value.isLoading = false
    chatHookState.value.isProcessing = false
    chatHookState.value.selectedAssistant = {
      kind: "persona",
      id: "p1",
      name: "Analyst"
    }
    chatHookState.value.serverChatAssistantKind = null
    chatHookState.value.serverChatAssistantId = null
    chatHookState.value.serverChatMetaLoaded = false
    chatHookState.onSubmit.mockResolvedValue({ status: "submitted" })
  })

  it("inserts staged source summary into the composer without sending and clears structured staging", () => {
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Insert context summary" }))

    expect(screen.getByRole("textbox", { name: "Chat workspace message" })).toHaveValue(
      "Context sources:\n1. Operator Notes [document, scope: Default workspace]\n\n"
    )
    expect(chatHookState.onSubmit).not.toHaveBeenCalled()
    expect(onClearStagedSources).toHaveBeenCalledTimes(1)
  })

  it("clears inserted draft context when the workspace changes", () => {
    const onClearStagedSources = vi.fn()

    const { rerender } = render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Insert context summary" }))
    expect(
      (screen.getByRole("textbox", {
        name: "Chat workspace message"
      }) as HTMLTextAreaElement).value
    ).toContain("Operator Notes")

    rerender(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-2"
      />
    )

    expect(screen.getByRole("textbox", { name: "Chat workspace message" }))
      .toHaveValue("")
  })

  it("sends with staged context through the shared chat path", async () => {
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Summarize this" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(getSubmitPayload()).toMatchObject({
      message: expect.stringContaining("Summarize this"),
      image: "",
      requestOverrides: expect.objectContaining({
        ragMediaIds: [101],
        fileRetrievalEnabled: true
      })
    })
    expect(onClearStagedSources).toHaveBeenCalledTimes(1)
  })

  it("sends the composer draft with Ctrl+Enter", async () => {
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    const composer = screen.getByRole("textbox", { name: "Chat workspace message" })
    fireEvent.change(composer, { target: { value: "Keyboard send" } })
    fireEvent.keyDown(composer, { key: "Enter", ctrlKey: true })

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(getSubmitPayload()).toMatchObject({
      message: "Keyboard send",
      requestOverrides: expect.objectContaining({
        ragMediaIds: [],
        fileRetrievalEnabled: false,
        chatMode: "normal"
      })
    })
  })

  it("includes staged source summary in the submitted message when no ready media ids can carry it", async () => {
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={stagedWithoutReadyMedia}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Draft instruction" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(getSubmitPayload()).toMatchObject({
      message: expect.stringContaining("Draft instruction"),
      requestOverrides: expect.objectContaining({
        ragMediaIds: [],
        fileRetrievalEnabled: false,
        chatMode: "normal"
      })
    })
    expect(getSubmitPayload().message).toEqual(
      expect.stringContaining("Indexing Notes")
    )
    expect(onClearStagedSources).toHaveBeenCalledTimes(1)
  })

  it("includes staged source summary when mixed staged sources cannot all be sent as ready media ids", async () => {
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={stagedWithMixedAvailability}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Draft instruction" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(getSubmitPayload()).toMatchObject({
      requestOverrides: expect.objectContaining({
        ragMediaIds: [101],
        fileRetrievalEnabled: true,
        chatMode: "rag"
      })
    })
    expect(getSubmitPayload().message).toEqual(
      expect.stringContaining("Draft instruction")
    )
    expect(getSubmitPayload().message).toEqual(
      expect.stringContaining("Indexing Notes")
    )
    expect(onClearStagedSources).toHaveBeenCalledTimes(1)
  })

  it("sends staged-only context when the composer is empty", async () => {
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(getSubmitPayload()).toMatchObject({
      message: expect.stringContaining("Operator Notes"),
      requestOverrides: expect.objectContaining({
        ragMediaIds: [101],
        fileRetrievalEnabled: true,
        chatMode: "rag"
      })
    })
    expect(onClearStagedSources).toHaveBeenCalledTimes(1)
  })

  it("does not submit while a stream is active", () => {
    chatHookState.value.streaming = true
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Do not send yet" }
    })

    const sendButton = screen.getByRole("button", { name: "Send message" })
    expect(sendButton).toBeDisabled()
    fireEvent.click(sendButton)
    expect(chatHookState.onSubmit).not.toHaveBeenCalled()
  })

  it("does not submit while the backend is unavailable", () => {
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={onClearStagedSources}
        backendAvailable={false}
        workspaceId="workspace-1"
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Do not send offline" }
    })

    const sendButton = screen.getByRole("button", { name: "Send message" })
    expect(sendButton).toBeDisabled()
    fireEvent.click(sendButton)

    expect(chatHookState.onSubmit).not.toHaveBeenCalled()
    expect(onClearStagedSources).not.toHaveBeenCalled()
  })

  it("treats an empty workspace id as loading and does not submit under a global scope", () => {
    chatHookState.value.messages = [
      {
        id: "message-1",
        isBot: true,
        name: "Analyst",
        role: "assistant",
        message: "Hydrating",
        sources: []
      }
    ]
    const onClearStagedSources = vi.fn()
    const onRuntimeStateChange = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId=""
        onRuntimeStateChange={onRuntimeStateChange}
      />
    )

    expect(chatHookState.useMessageOption).toHaveBeenCalledWith({
      scope: { type: "global" }
    })
    expect(screen.getByTestId("workspace-panel-message")).toHaveAttribute(
      "data-conversation-instance-id",
      "workspace-chat"
    )

    const composer = screen.getByRole("textbox", { name: "Chat workspace message" })
    fireEvent.change(composer, { target: { value: "Do not send during hydration" } })

    const sendButton = screen.getByRole("button", { name: "Send message" })
    expect(sendButton).toBeDisabled()
    fireEvent.click(sendButton)
    expect(screen.getByRole("button", { name: "Send with staged context" }))
      .toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))
    expect(chatHookState.onSubmit).not.toHaveBeenCalled()
    expect(onClearStagedSources).not.toHaveBeenCalled()
    expect(onRuntimeStateChange).toHaveBeenCalledWith(
      expect.objectContaining({ backendAvailable: false })
    )
    expect(screen.getByText("Loading workspace context")).toBeInTheDocument()
  })

  it("preserves draft and staged context when submit returns a failed result", async () => {
    chatHookState.onSubmit.mockResolvedValueOnce({
      status: "failed",
      errorMessage: "network"
    })
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Keep this draft" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))

    await screen.findByText("network")
    expect(screen.getByRole("textbox", { name: "Chat workspace message" })).toHaveValue("Keep this draft")
    expect(onClearStagedSources).not.toHaveBeenCalled()
  })

  it("preserves draft and staged context without an error when submit is skipped", async () => {
    chatHookState.onSubmit.mockResolvedValueOnce({
      status: "skipped",
      reason: "empty"
    })
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Keep this draft" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: "Chat workspace message" })).toHaveValue("Keep this draft")
    expect(onClearStagedSources).not.toHaveBeenCalled()
  })

  it("also preserves draft and staged context when submit rejects unexpectedly", async () => {
    chatHookState.onSubmit.mockRejectedValueOnce(new Error("network"))
    const onClearStagedSources = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={onClearStagedSources}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Keep this draft" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send with staged context" }))

    await screen.findByText("Send failed")
    expect(screen.getByRole("textbox", { name: "Chat workspace message" })).toHaveValue("Keep this draft")
    expect(onClearStagedSources).not.toHaveBeenCalled()
  })

  it("shows loading state and wires stop streaming to the shared abort handler", () => {
    chatHookState.value.streaming = true
    chatHookState.value.isProcessing = true

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    expect(screen.getByText("Streaming")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Stop generating" }))
    expect(chatHookState.stopStreamingRequest).toHaveBeenCalledTimes(1)
    expect(chatHookState.stopStreamingRequest).toHaveBeenCalledWith()
  })

  it("uses workspace chat scope and reports runtime state", () => {
    const onRuntimeStateChange = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={staged}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
        onRuntimeStateChange={onRuntimeStateChange}
      />
    )

    expect(chatHookState.useMessageOption).toHaveBeenCalledWith(
      expect.objectContaining({
        scope: { type: "workspace", workspaceId: "workspace-1" }
      })
    )
    expect(onRuntimeStateChange).toHaveBeenCalledWith(
      expect.objectContaining({
        streaming: false,
        selectedModelLabel: "gpt-test",
        selectedPersonaLabel: "Analyst",
        assistantSource: "explicit"
      })
    )
  })

  it("inherits the available workspace persona default on first submit", async () => {
    chatHookState.value.selectedAssistant = null
    const onRuntimeStateChange = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
        effectiveAssistantDefault={availableWorkspaceDefault}
        onRuntimeStateChange={onRuntimeStateChange}
      />
    )

    expect(chatHookState.useMessageOption).toHaveBeenCalledWith(
      expect.objectContaining({
        scope: { type: "workspace", workspaceId: "workspace-1" },
        inheritedAssistant: expect.objectContaining({
          kind: "persona",
          id: "workspace-persona",
          name: "Workspace Analyst",
          metadata: expect.objectContaining({
            selectionMode: "tracked",
            source: "workspace",
            personaMemoryMode: "read_write"
          })
        })
      })
    )
    expect(onRuntimeStateChange).toHaveBeenCalledWith(
      expect.objectContaining({
        selectedPersonaLabel: "Workspace Analyst",
        assistantSource: "workspace"
      })
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Use the default persona" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(getSubmitPayload()).toMatchObject({
      requestOverrides: expect.objectContaining({
        assistant_kind: "persona",
        assistant_id: "workspace-persona",
        persona_memory_mode: "read_write"
      })
    })
  })

  it("keeps an explicit selected persona ahead of the workspace default", async () => {
    chatHookState.value.selectedAssistant = {
      kind: "persona",
      id: "explicit-persona",
      name: "Explicit Analyst",
      metadata: {
        selectionMode: "tracked"
      }
    }

    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
        effectiveAssistantDefault={availableWorkspaceDefault}
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Use explicit persona" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(getSubmitPayload().requestOverrides).not.toMatchObject({
      assistant_id: "workspace-persona"
    })
  })

  it("does not inherit an unavailable workspace default", async () => {
    chatHookState.value.selectedAssistant = null
    const onRuntimeStateChange = vi.fn()

    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
        effectiveAssistantDefault={unavailableWorkspaceDefault}
        onRuntimeStateChange={onRuntimeStateChange}
      />
    )

    expect(onRuntimeStateChange).toHaveBeenCalledWith(
      expect.objectContaining({
        selectedPersonaLabel: null,
        assistantSource: "unavailable",
        workspaceAssistantDegradedReason: "persona_deleted"
      })
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "No default persona" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(getSubmitPayload().requestOverrides).not.toMatchObject({
      assistant_kind: "persona",
      assistant_id: "workspace-persona"
    })
  })

  it("does not mutate existing chat assistant metadata when the workspace default changes", async () => {
    chatHookState.value.selectedAssistant = null
    chatHookState.value.serverChatAssistantKind = "persona"
    chatHookState.value.serverChatAssistantId = "session-persona"
    chatHookState.value.serverChatMetaLoaded = true

    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
        effectiveAssistantDefault={availableWorkspaceDefault}
      />
    )

    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), {
      target: { value: "Continue existing chat" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))

    await waitFor(() => expect(chatHookState.onSubmit).toHaveBeenCalledTimes(1))
    expect(getSubmitPayload().requestOverrides).not.toMatchObject({
      assistant_id: "workspace-persona"
    })
  })
})
