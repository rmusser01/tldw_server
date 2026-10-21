import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { expect, it, vi } from "vitest"
import { PromptStudioSettings } from "../prompt-studio"
const state = vi.hoisted(() => ({ resolve: vi.fn(), save: vi.fn(async () => ({ defaultProjectId: null, pageSize: 10 })) }))
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({ useCanonicalConnectionConfig: () => ({ config: { serverUrl: "https://server.test", authMode: "single-user" }, authorityLoading: false }) }))
vi.mock("@/services/recipe-persistence-uncertainty", () => ({ resolveRecipePersistenceOwnerView: state.resolve }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_key: string, fallback: unknown) => typeof fallback === "string" ? fallback : _key }) }))
vi.mock("@/services/prompt-studio", () => ({ hasPromptStudio: async () => false, getPromptStudioStatus: vi.fn() }))
vi.mock("@/services/prompt-studio-settings", () => ({ getPromptStudioDefaultsForOwner: async () => ({ defaultProjectId: null, pageSize: 10, autoSyncWorkspacePrompts: false, executeProvider: "openai", executeModel: "gpt-3.5-turbo", executeTemperature: 0.2, executeMaxTokens: 256, evalModelName: "gpt-3.5-turbo", evalTemperature: 0.2, evalMaxTokens: 512, warnSeconds: 30 }), setPromptStudioDefaults: state.save }))
it("offers the offline preferences form through the real authority hook without probing a private owner", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const view = render(<QueryClientProvider client={client}><PromptStudioSettings /></QueryClientProvider>)
  await screen.findAllByDisplayValue("gpt-3.5-turbo")
  fireEvent.click(screen.getByRole("button", { name: "Save" }))
  await waitFor(() => expect(state.save).toHaveBeenCalledWith(expect.objectContaining({ defaultProjectId: null }), null))
  expect(state.resolve).not.toHaveBeenCalled()
  view.unmount(); client.clear()
})
