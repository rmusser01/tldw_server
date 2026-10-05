import React from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { IngestWizardProvider, useIngestWizard } from "../IngestWizardContext"
import { useQuickIngestSessionStore } from "@/store/quick-ingest-session"
import { AddContentStep } from "../AddContentStep"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, value: any) =>
      typeof value === "string" ? value : value?.defaultValue || key
  })
}))
vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({ capabilities: {} })
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {} }))

function QueueProbe() {
  const { state, skipToProcessing } = useIngestWizard()
  return (
    <>
      <output data-testid="urls">
        {JSON.stringify(state.queueItems.map((item) => item.url))}
      </output>
      <output data-testid="submitted">
        {state.processingState.perItemProgress.length}
      </output>
      <button onClick={skipToProcessing}>submit probe</button>
    </>
  )
}
const mount = () =>
  render(
    <IngestWizardProvider>
      <AddContentStep />
      <QueueProbe />
    </IngestWizardProvider>
  )
const add = (input: string) => {
  fireEvent.change(screen.getByRole("textbox", { name: "URL input area" }), {
    target: { value: input }
  })
  fireEvent.click(screen.getByRole("button", { name: "Add URLs to queue" }))
}
afterEach(() => vi.unstubAllGlobals())

describe("eligible source queue", () => {
  it.each([
    [
      "https://example.com/a, https://example.com/b",
      ["https://example.com/a", "https://example.com/b"]
    ],
    ["https://example.com/a,b", ["https://example.com/a,b"]],
    [
      "https://example.com/a\nhttps://example.com/b",
      ["https://example.com/a", "https://example.com/b"]
    ]
  ])("parses %s without changing a URL", (input, urls) => {
    mount()
    add(input)
    expect(JSON.parse(screen.getByTestId("urls").textContent!)).toEqual(urls)
  })
  it("excludes ambiguous combined and invalid inputs from processing", () => {
    mount()
    add("https://example.com/a,https://example.com/b\ninvalid")
    fireEvent.click(screen.getByText("submit probe"))
    expect(screen.getByTestId("submitted")).toHaveTextContent("0")
  })
  it("skips duplicates unless explicitly processed again", () => {
    mount()
    add("https://example.com/a\nhttps://example.com/a")
    fireEvent.click(screen.getByText("submit probe"))
    expect(screen.getByTestId("submitted")).toHaveTextContent("1")
    fireEvent.click(screen.getByRole("button", { name: "Process again" }))
    fireEvent.click(screen.getByText("submit probe"))
    expect(screen.getByTestId("submitted")).toHaveTextContent("2")
  })
  it("never queries tabs on WebUI", () => {
    const query = vi.fn()
    vi.stubGlobal("browser", { runtime: {}, tabs: { query } })
    mount()
    expect(
      screen.queryByRole("button", { name: "Capture current tab" })
    ).not.toBeInTheDocument()
    expect(query).not.toHaveBeenCalled()
  })
  it.each(["chrome://settings", undefined])(
    "explains inaccessible active tab %s",
    async (url) => {
      vi.stubGlobal("browser", {
        runtime: { id: "extension" },
        tabs: { query: vi.fn().mockResolvedValue([{ url }]) }
      })
      mount()
      fireEvent.click(
        screen.getByRole("button", { name: "Capture current tab" })
      )
      expect(await screen.findByRole("alert")).toHaveTextContent(/HTTP.*HTTPS/)
      expect(screen.getByTestId("urls")).toHaveTextContent("[]")
    }
  )
  it("captures an HTTP tab into the queue", async () => {
    vi.stubGlobal("browser", {
      runtime: { id: "extension" },
      tabs: {
        query: vi
          .fn()
          .mockResolvedValue([{ url: "https://example.com/article" }])
      }
    })
    mount()
    fireEvent.click(screen.getByRole("button", { name: "Capture current tab" }))
    await waitFor(() =>
      expect(screen.getByTestId("urls")).toHaveTextContent(
        "https://example.com/article"
      )
    )
  })
})

it("ignores a captured tab after the owner changes", async () => {
  let resolveTabs!: (tabs: { url: string }[]) => void
  const tabs = new Promise<{ url: string }[]>((resolve) => {
    resolveTabs = resolve
  })
  vi.stubGlobal("browser", {
    runtime: { id: "extension" },
    tabs: { query: vi.fn().mockReturnValue(tabs) }
  })
  mount()
  fireEvent.click(screen.getByRole("button", { name: "Capture current tab" }))
  act(() =>
    useQuickIngestSessionStore.getState().setAuthority("different-owner")
  )
  await act(async () => {
    resolveTabs([{ url: "https://example.com/private" }])
    await tabs
  })
  expect(screen.getByTestId("urls")).toHaveTextContent("[]")
  useQuickIngestSessionStore.getState().setAuthority(null)
})
