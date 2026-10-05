import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter, useLocation, useNavigate } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { MediaKnowledgeActions } from "../MediaKnowledgeActions"
import { consumeResearchWorkspacePrefill } from "@/utils/research-workspace-prefill"
const memory = vi.hoisted(() => ({
  values: new Map<string, unknown>(),
  wait: null as Promise<void> | null,
}))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    set: async (key: string, value: unknown) => {
      await memory.wait
      memory.values.set(key, value)
    },
    get: async (key: string) => memory.values.get(key),
    remove: async (key: string) => {
      memory.values.delete(key)
    },
  }),
  safeStorageSerde: { deserializer: (value: unknown) => value },
}))
vi.mock("@/hooks/useHomeMilestoneScope", () => ({
  useHomeMilestoneScope: () => "server:alice",
}))
vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({ error: vi.fn() }),
}))
function Destination() {
  const location = useLocation()
  return (
    <output aria-label="Destination">
      {location.pathname}
      {location.search}
    </output>
  )
}
const items = [
  { id: 3, title: "Alpha paper", type: "pdf" },
  { id: "7", title: "Beta transcript", type: "audio" },
]
function Actions({ sources }: { sources: typeof items }) {
  const navigate = useNavigate()
  return <MediaKnowledgeActions items={sources} navigate={navigate} selection />
}
function mount(sources = items) {
  return render(
    <MemoryRouter initialEntries={["/review"]}>
      <Actions sources={sources} />
      <Destination />
    </MemoryRouter>,
  )
}

beforeEach(() => {
  memory.values.clear()
  memory.wait = null
})
describe("review source continuation", () => {
  it("addresses precisely the reviewed source IDs", () => {
    mount()
    fireEvent.click(screen.getByRole("button", { name: "Ask selected items" }))
    expect(screen.getByLabelText("Destination")).toHaveTextContent(
      "/knowledge?media_ids=3%2C7",
    )
  })
  it("persists recognizable sources before entering Research Workspace", async () => {
    let finish!: () => void
    memory.wait = new Promise((resolve) => {
      finish = resolve
    })
    mount()
    fireEvent.click(
      screen.getByRole("button", { name: "Research with selected sources" }),
    )
    expect(screen.getByLabelText("Destination")).toHaveTextContent("/review")
    await act(async () => {
      finish()
      await Promise.resolve()
    })
    await waitFor(() =>
      expect(screen.getByLabelText("Destination")).toHaveTextContent(
        "/research-workspace",
      ),
    )
    const payload = await consumeResearchWorkspacePrefill()
    expect(
      payload?.sources.map((source) => ({
        mediaId: source.mediaId,
        title: source.title,
        type: source.type,
      })),
    ).toEqual([
      { mediaId: 3, title: "Alpha paper", type: "pdf" },
      { mediaId: 7, title: "Beta transcript", type: "audio" },
    ])
  })
  it("never continues a malformed selection as partial or whole-library scope", () => {
    mount([{ id: "invalid", title: "Unstored source", type: "pdf" }])
    expect(
      screen.getByRole("button", { name: "Ask selected items" }),
    ).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Ask selected items" }))
    expect(screen.getByLabelText("Destination")).toHaveTextContent("/review")
  })
  it("drops navigation when an account change interrupts source persistence", async () => {
    let finish!: () => void
    memory.wait = new Promise((resolve) => {
      finish = resolve
    })
    mount()
    fireEvent.click(
      screen.getByRole("button", { name: "Research with selected sources" }),
    )
    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:auth-principal-changed", {
          detail: { kind: "logout" },
        }),
      ),
    )
    await act(async () => {
      finish()
      await Promise.resolve()
    })
    expect(screen.getByLabelText("Destination")).toHaveTextContent("/review")
    expect(
      screen.getByRole("button", { name: "Ask selected items" }),
    ).toBeDisabled()
  })
})
