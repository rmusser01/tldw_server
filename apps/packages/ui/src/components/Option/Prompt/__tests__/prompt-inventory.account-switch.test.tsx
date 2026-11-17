import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { MemoryRouter } from "react-router-dom"
import { act, render, screen, waitFor } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"
import { usePromptPaletteCommands } from "../usePromptPaletteCommands"
const state = vi.hoisted(() => ({ owner: "recipe-owner:sha256:" + "b".repeat(64), config: { serverUrl: "https://server.test", authMode: "multi-user", accessToken: "bob" }, loading: false }))
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({ useCanonicalConnectionConfig: () => ({ config: state.config, loading: state.loading, authorityLoading: state.loading }) }))
vi.mock("@/services/recipe-persistence-uncertainty", () => ({ resolveRecipePersistenceOwnerView: async () => ({ ownerId: state.owner, authorizationRevision: "revision" }) }))
vi.mock("@/db/dexie/helpers", () => ({ getSharedLocalPrompts: async () => [], getAllPrompts: async () => [{ id: state.owner, title: state.owner.endsWith("b") ? "Bob private" : "Alice private", content: "private", is_system: false, createdAt: 1, favorite: true, ownerId: state.owner }] }))
function Commands() { return <>{usePromptPaletteCommands().map(c => <button key={c.id}>{c.label}</button>)}</> }
afterEach(() => vi.clearAllMocks())
it("retires cached Bob command inventory on an account change with the same QueryClient", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })
  const body = () => <QueryClientProvider client={client}><MemoryRouter><Commands /></MemoryRouter></QueryClientProvider>
  const view = render(body()); await screen.findByRole("button", { name: "Bob private" })
  state.owner = "recipe-owner:sha256:" + "a".repeat(64); state.config = { ...state.config, accessToken: "alice" }
  await act(async () => { window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed")); view.rerender(body()) })
  await waitFor(() => expect(screen.queryByRole("button", { name: "Bob private" })).not.toBeInTheDocument())
  await screen.findByRole("button", { name: "Alice private" }); client.clear()
})
