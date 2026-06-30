// @vitest-environment jsdom
import { render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { DynamicMessageRenderer } from "../DynamicMessageRenderer"

vi.mock("../registry", async () => {
  const actual = await vi.importActual<typeof import("../registry")>("../registry")
  return {
    ...actual,
    loadDynamicUIRenderer: vi.fn(async () => ({
      default: ({
        source,
        onAction
      }: {
        source: string
        onAction?: (payload: unknown) => void
      }) => {
        if (source.includes("throw")) {
          throw new Error("renderer crashed")
        }
        return (
          <button
            type="button"
            data-testid="openui-rendered"
            onClick={() =>
              onAction?.({
                actionId: "survey",
                actionType: "submit",
                values: { answer: "yes" }
              })
            }>
            {source}
          </button>
        )
      }
    }))
  }
})

describe("DynamicMessageRenderer", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => undefined)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("renders enabled OpenUI metadata on web chat", async () => {
    render(
      <DynamicMessageRenderer
        envelope={{ renderer: "openui", version: "v1", source: "root = <Card />" }}
        sourceMessageId="assistant-1"
        sourceText="root = <Card />"
        surface="web-chat"
      />
    )

    expect(await screen.findByTestId("openui-rendered")).toHaveTextContent("root = <Card />")
  })

  it("falls back to source when surface is disabled", () => {
    render(
      <DynamicMessageRenderer
        envelope={{ renderer: "openui", version: "v1", source: "root = <Card />" }}
        sourceMessageId="assistant-1"
        sourceText="root = <Card />"
        surface="extension-sidepanel"
      />
    )

    expect(screen.getByText(/OpenUI source/i)).toBeInTheDocument()
  })

  it("falls back to the canonical envelope source on disabled surfaces", () => {
    render(
      <DynamicMessageRenderer
        envelope={{
          renderer: "openui",
          version: "v1",
          source: "root = <Card><Text>Canonical source</Text></Card>"
        }}
        sourceMessageId="assistant-1"
        sourceText="assistant prose that differs from source"
        surface="workspace"
      />
    )

    const fallback = screen.getByTestId("dynamic-ui-source-fallback")
    expect(fallback).toHaveAttribute("data-dynamic-ui-surface", "workspace")
    expect(fallback).toHaveTextContent("root = <Card><Text>Canonical source</Text></Card>")
    expect(fallback).not.toHaveTextContent("assistant prose that differs from source")
  })

  it("falls back to source with an alert when the renderer component throws", async () => {
    render(
      <DynamicMessageRenderer
        envelope={{ renderer: "openui", version: "v1", source: "root = <Card /> // throw" }}
        sourceMessageId="assistant-1"
        sourceText="root = <Card /> // throw"
        surface="web-chat"
      />
    )

    expect(await screen.findByRole("alert")).toHaveTextContent("renderer crashed")
    expect(screen.getByText(/root = <Card \/> \/\/ throw/)).toBeInTheDocument()
  })

  it("attaches host-owned source message provenance to renderer actions", async () => {
    const onAction = vi.fn()
    render(
      <DynamicMessageRenderer
        envelope={{ renderer: "openui", version: "v1", source: "root = <Form />" }}
        sourceMessageId="assistant-1"
        sourceText="root = <Form />"
        surface="web-chat"
        onAction={onAction}
      />
    )

    ;(await screen.findByTestId("openui-rendered")).click()

    expect(onAction).toHaveBeenCalledWith({
      renderer: "openui",
      sourceMessageId: "assistant-1",
      actionId: "survey",
      actionType: "submit",
      values: { answer: "yes" }
    })
  })

  it("wraps non-object renderer actions before adding provenance", async () => {
    vi.mocked((await import("../registry")).loadDynamicUIRenderer).mockResolvedValueOnce({
      default: ({ onAction }) => (
        <button
          type="button"
          data-testid="openui-rendered-primitive"
          onClick={() => onAction?.("accepted")}>
          action
        </button>
      )
    })
    const onAction = vi.fn()

    render(
      <DynamicMessageRenderer
        envelope={{ renderer: "openui", version: "v1", source: "root = <Button />" }}
        sourceMessageId="assistant-primitive"
        sourceText="root = <Button />"
        surface="web-chat"
        onAction={onAction}
      />
    )

    ;(await screen.findByTestId("openui-rendered-primitive")).click()

    expect(onAction).toHaveBeenCalledWith({
      renderer: "openui",
      sourceMessageId: "assistant-primitive",
      values: "accepted"
    })
  })
})
