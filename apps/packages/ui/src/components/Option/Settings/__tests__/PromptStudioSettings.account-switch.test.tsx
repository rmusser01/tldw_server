import React from "react"
import { act, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, expect, it, vi } from "vitest"
import { PromptStudioSettings } from "../prompt-studio"
const state = vi.hoisted(() => ({ owner: "alice", ready: true, reads: vi.fn(), saves: vi.fn() }))
vi.mock("@/hooks/usePromptLocalAuthority", () => ({ usePromptLocalAuthority: () => ({ ownerId: state.owner, ready: state.ready }) }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_k: string, fallback: any) => typeof fallback === "string" ? fallback : _k }) }))
vi.mock("@/services/prompt-studio", () => ({ hasPromptStudio: async () => true, getPromptStudioStatus: vi.fn() }))
vi.mock("@/services/prompt-studio-settings", () => ({ getPromptStudioDefaultsForOwner: (...a: any[]) => state.reads(...a), setPromptStudioDefaults: (...a: any[]) => state.saves(...a) }))
beforeEach(() => {
  state.owner = "alice"; state.ready = true
  state.reads.mockReset().mockImplementation(async owner => ({ defaultProjectId: owner === "alice" ? 17 : null, pageSize: 10 }))
  state.saves.mockReset()
})
it("clears a cached default project from the held settings form after an account switch", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const body = () => <QueryClientProvider client={client}><PromptStudioSettings /></QueryClientProvider>
  const view = render(body())
  await waitFor(() => expect(screen.getByPlaceholderText("Optional project id")).toHaveValue("17"))
  state.owner = "bob"; await act(async () => { view.rerender(body()) })
  await waitFor(() => expect(screen.getByPlaceholderText("Optional project id")).toHaveValue(""))
  expect(state.reads).toHaveBeenCalledWith("bob")
  view.unmount(); client.clear()
})
