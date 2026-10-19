import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import { usePromptInventory } from "../usePromptInventory"

const state = vi.hoisted(() => ({
  loading: true,
  config: { serverUrl: "https://server.test", authMode: "multi-user", accessToken: "alice" } as Record<string, unknown>,
  resolve: vi.fn(), owned: vi.fn()
}))
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({ useCanonicalConnectionConfig: () => ({ config: state.config, authorityLoading: state.loading }) }))
vi.mock("@/services/recipe-persistence-uncertainty", () => ({ resolveRecipePersistenceOwnerView: state.resolve }))
vi.mock("@/db/dexie/helpers", () => ({
  getAllPrompts: state.owned, getDeletedPrompts: state.owned,
  getSharedLocalPrompts: async () => [
    { id: "legacy", title: "Legacy local", content: "local", createdAt: 1 },
    { id: "private", title: "Private owned", ownerId: "alice", createdAt: 1 },
    { id: "linked", title: "Ambiguous linked", serverId: 17, createdAt: 1 }
  ]
}))
function Inventory() { return <>{usePromptInventory("inventory").data?.map(prompt => <span key={prompt.id}>{prompt.title}</span>)}</> }
function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })
  render(<QueryClientProvider client={client}><Inventory /></QueryClientProvider>)
  return client
}
beforeEach(() => {
  vi.clearAllMocks(); state.loading = true
  state.config = { serverUrl: "https://server.test", authMode: "multi-user", accessToken: "alice" }
})
it("shows only shared legacy content while authenticated ownership is pending", async () => {
  const client = mount()
  await screen.findByText("Legacy local")
  expect(screen.queryByText("Private owned")).not.toBeInTheDocument()
  expect(screen.queryByText("Ambiguous linked")).not.toBeInTheDocument()
  expect(state.owned).not.toHaveBeenCalled()
  client.clear()
})
it("does not reveal owned or linked rows when authenticated ownership rejects", async () => {
  state.loading = false; state.resolve.mockRejectedValue(new Error("unavailable"))
  const client = mount()
  await screen.findByText("Legacy local")
  expect(screen.queryByText("Private owned")).not.toBeInTheDocument()
  expect(screen.queryByText("Ambiguous linked")).not.toBeInTheDocument()
  expect(state.owned).not.toHaveBeenCalled()
  client.clear()
})
it("does not probe an authenticated facade merely to read unconfigured legacy data", async () => {
  state.loading = false; state.config = { serverUrl: "https://server.test", authMode: "single-user" }
  const client = mount()
  await screen.findByText("Legacy local")
  expect(state.resolve).not.toHaveBeenCalled()
  expect(state.owned).not.toHaveBeenCalled()
  client.clear()
})
