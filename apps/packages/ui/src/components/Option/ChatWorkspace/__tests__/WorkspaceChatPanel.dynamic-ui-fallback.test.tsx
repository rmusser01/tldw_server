import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { WorkspaceChatPanel } from "../WorkspaceChatPanel"

const chatHookState = vi.hoisted(() => {
  const value: any = {
    messages: [],
    onSubmit: vi.fn(async () => ({ status: "submitted" })),
    streaming: false,
    isLoading: false,
    isProcessing: false,
    stopStreamingRequest: vi.fn(),
    selectedModel: "gpt-test",
    selectedAssistant: null
  }
  return {
    value,
    useMessageOption: vi.fn(() => value)
  }
})

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => chatHookState.useMessageOption()
}))

vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: (props: {
    dynamicUISurface?: string
    metadataExtra?: Record<string, any>
    message: string
  }) => {
    const envelope = props.metadataExtra?.dynamic_ui
    return (
      <article data-testid="workspace-panel-message">
        {envelope ? (
          <pre
            data-testid="dynamic-ui-source-fallback"
            data-dynamic-ui-surface={props.dynamicUISurface}
          >
            {envelope.source}
          </pre>
        ) : (
          props.message
        )}
      </article>
    )
  }
}))

describe("WorkspaceChatPanel Dynamic UI fallback", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    chatHookState.value.messages = []
  })

  it("renders OpenUI metadata as source fallback on the workspace surface", () => {
    chatHookState.value.messages = [
      {
        id: "assistant-openui",
        role: "assistant",
        isBot: true,
        message: "Here is the requested UI.",
        metadataExtra: {
          dynamic_ui: {
            renderer: "openui",
            version: "v1",
            source: "root = <Card><Text>Workspace source</Text></Card>"
          }
        }
      }
    ]

    render(
      <WorkspaceChatPanel
        stagedSources={[]}
        onClearStagedSources={vi.fn()}
        backendAvailable
        workspaceId="workspace-1"
      />
    )

    expect(screen.getByTestId("dynamic-ui-source-fallback")).toHaveTextContent(
      "root = <Card><Text>Workspace source</Text></Card>"
    )
    expect(screen.getByTestId("dynamic-ui-source-fallback")).not.toHaveTextContent(
      "Here is the requested UI."
    )
    expect(screen.getByTestId("dynamic-ui-source-fallback")).toHaveAttribute(
      "data-dynamic-ui-surface",
      "workspace"
    )
    expect(screen.queryByTestId("openui-runtime")).not.toBeInTheDocument()
  })
})
