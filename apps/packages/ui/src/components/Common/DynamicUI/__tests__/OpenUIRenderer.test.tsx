// @vitest-environment jsdom
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import OpenUIRenderer from "../renderers/OpenUIRenderer"

type MockOpenUIRendererProps = {
  response: string | null
  library?: unknown
  initialState?: Record<string, unknown>
  isStreaming?: boolean
  toolProvider?: unknown
  onAction?: (payload: unknown) => void
}

const openuiMocks = vi.hoisted(() => {
  const safeOpenUIChatLibrary = { name: "mock-safe-openui-chat-library" }
  const openuiChatLibrary = {
    root: "Card",
    components: {
      Card: { name: "Card" },
      TextContent: { name: "TextContent" },
      Form: { name: "Form" },
      Button: { name: "Button" },
      BarChart: { name: "BarChart" },
      Series: { name: "Series" },
      PieChart: { name: "PieChart" },
      Slice: { name: "Slice" }
    },
    componentGroups: [
      { name: "Content", components: ["TextContent"] },
      { name: "Forms", components: ["Form", "Button"] },
      { name: "Charts (2D)", components: ["BarChart", "Series"] },
      { name: "Charts (1D)", components: ["PieChart", "Slice"] }
    ]
  }
  const createLibrarySpy = vi.fn(() => safeOpenUIChatLibrary)
  const rendererSpy = vi.fn(
    ({ response, onAction }: MockOpenUIRendererProps) => (
      <button
        type="button"
        data-testid="openui-runtime"
        onClick={() =>
          onAction?.({
            type: "Submit",
            params: { actionId: "survey" },
            humanFriendlyMessage: "Submit survey",
            formName: "surveyForm",
            formState: {
              surveyForm: {
                answer: { value: "yes", componentType: "RadioGroup" }
              }
            }
          })
        }>
        {response}
      </button>
    )
  )

  return { createLibrarySpy, openuiChatLibrary, rendererSpy, safeOpenUIChatLibrary }
})

vi.mock("@openuidev/react-lang", () => ({
  createLibrary: openuiMocks.createLibrarySpy,
  Renderer: openuiMocks.rendererSpy
}))

vi.mock("@openuidev/react-ui/genui-lib", () => ({
  openuiChatLibrary: openuiMocks.openuiChatLibrary
}))

describe("OpenUIRenderer", () => {
  it("passes source, state, allowlisted chat library, and actions to the OpenUI runtime", () => {
    const onAction = vi.fn()
    const state = {
      surveyForm: {
        answer: { value: "yes", componentType: "RadioGroup" }
      }
    }

    render(
      <OpenUIRenderer
        envelope={{
          renderer: "openui",
          version: "v1",
          source: "root = <Form name=\"surveyForm\" />",
          state
        }}
        source={'root = <Form name="surveyForm" />'}
        sourceMessageId="assistant-1"
        onAction={onAction}
      />
    )

    const shell = screen.getByTestId("dynamic-ui-openui-shell")
    expect(shell).toHaveClass("dynamic-ui-openui")
    expect(shell).toHaveStyle({
      "--openui-background": "rgb(var(--color-surface))",
      "--openui-text-neutral-primary": "rgb(var(--color-text))",
      "--openui-border-default": "rgb(var(--color-border))"
    })
    expect(screen.getByTestId("openui-runtime")).toHaveTextContent(
      "root = <Form name=\"surveyForm\" />"
    )

    expect(openuiMocks.createLibrarySpy).toHaveBeenCalledWith({
      root: "Card",
      components: [
        openuiMocks.openuiChatLibrary.components.Card,
        openuiMocks.openuiChatLibrary.components.TextContent,
        openuiMocks.openuiChatLibrary.components.Form,
        openuiMocks.openuiChatLibrary.components.Button
      ],
      componentGroups: [
        { name: "Content", components: ["TextContent"] },
        { name: "Forms", components: ["Form", "Button"] }
      ]
    })

    const rendererProps = openuiMocks.rendererSpy.mock.calls[0]?.[0]
    expect(rendererProps).toMatchObject({
      response: "root = <Form name=\"surveyForm\" />",
      initialState: state,
      isStreaming: false
    })
    expect(rendererProps.library).toBe(openuiMocks.safeOpenUIChatLibrary)
    expect(rendererProps.toolProvider).toBeNull()

    screen.getByTestId("openui-runtime").click()
    expect(onAction).toHaveBeenCalledWith({
      actionId: "survey",
      actionType: "submit",
      values: { answer: "yes" }
    })
  })

  it("normalizes OpenUI actions with missing params without throwing", () => {
    const onAction = vi.fn()

    render(
      <OpenUIRenderer
        envelope={{
          renderer: "openui",
          version: "v1",
          source: "root = <Form name=\"surveyForm\" />"
        }}
        source={'root = <Form name="surveyForm" />'}
        sourceMessageId="assistant-1"
        onAction={onAction}
      />
    )

    const rendererProps = openuiMocks.rendererSpy.mock.calls.at(-1)?.[0]
    expect(() =>
      rendererProps?.onAction?.({
        type: "Submit",
        formName: "surveyForm"
      } as any)
    ).not.toThrow()
    expect(onAction).toHaveBeenCalledWith({
      actionId: "surveyForm",
      actionType: "submit",
      values: {}
    })
  })
})
