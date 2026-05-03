import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { WorkspaceChatPanel } from "../WorkspaceChatPanel"
import type { StagedWorkspaceSource } from "../types"

const chatHookState = vi.hoisted(() => {
  const onSubmit = vi.fn(async (): Promise<any> => ({ status: "submitted" }))
  const stopStreamingRequest = vi.fn()
  const value: any = {
    messages: [],
    onSubmit,
    streaming: false,
    isLoading: false,
    isProcessing: false,
    stopStreamingRequest,
    selectedModel: "gpt-test",
    selectedAssistant: { kind: "persona", id: "p1", name: "Analyst" }
  }
  const useMessageOption = vi.fn(() => value)

  return { onSubmit, stopStreamingRequest, useMessageOption, value }
})

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: (...args: unknown[]) => chatHookState.useMessageOption(...args)
}))

vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: (props: { message: string }) => <article>{props.message}</article>
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

describe("WorkspaceChatPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    chatHookState.value.streaming = false
    chatHookState.value.isLoading = false
    chatHookState.value.isProcessing = false
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
    expect(chatHookState.onSubmit.mock.calls[0][0]).toMatchObject({
      message: expect.stringContaining("Summarize this"),
      image: "",
      requestOverrides: expect.objectContaining({
        ragMediaIds: [101],
        fileRetrievalEnabled: true
      })
    })
    expect(onClearStagedSources).toHaveBeenCalledTimes(1)
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

    await screen.findByText("Send failed")
    expect(screen.getByRole("textbox", { name: "Chat workspace message" })).toHaveValue("Keep this draft")
    expect(onClearStagedSources).not.toHaveBeenCalled()
  })

  it.each([
    { label: "skipped", result: { status: "skipped", reason: "empty" } },
    { label: "undefined", result: undefined }
  ])("preserves draft and staged context when submit returns $label", async ({ result }) => {
    chatHookState.onSubmit.mockResolvedValueOnce(result)
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

    expect(chatHookState.useMessageOption).toHaveBeenCalledWith({
      scope: { type: "workspace", workspaceId: "workspace-1" }
    })
    expect(onRuntimeStateChange).toHaveBeenCalledWith(
      expect.objectContaining({
        streaming: false,
        selectedModelLabel: "gpt-test",
        selectedPersonaLabel: "Analyst"
      })
    )
  })
})
