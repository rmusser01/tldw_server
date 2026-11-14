import React from "react"
import { notification } from "antd"
import { act, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { MemoryRouter } from "react-router-dom"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
const state = vi.hoisted(() => ({ system: undefined as string | undefined, quick: undefined as string | undefined }))
const owner = "recipe-owner:sha256:" + "a".repeat(64)
vi.mock("@/hooks/useMessageOption", () => ({ useMessageOption: () => ({ setSelectedSystemPrompt: (id: string | undefined) => { state.system = id }, setSelectedQuickPrompt: (value: string | undefined) => { state.quick = value } }) }))
vi.mock("@/services/application", () => ({ getAllCopilotPrompts: async () => [], upsertCopilotPrompts: vi.fn() }))
vi.mock("@/services/prompt-studio", () => ({ hasPromptStudio: async () => false, getPrompt: vi.fn(), getLlmProviders: async () => ({ providers: [] }) }))
vi.mock("@/services/recipe-persistence-uncertainty", async original => ({ ...await original(), resolveRecipePersistenceOwnerView: async () => ({ ownerId: "recipe-owner:sha256:" + "a".repeat(64), authorizationRevision: "current" }) }))
import { usePromptInteractions } from "../hooks/usePromptInteractions"
import { useStoreChatModelSettings } from "@/store/model"
const record = { id: "alice-prompt", name: "Pirate", system_prompt: "Always say ARRR.", user_prompt: "Alice private template" }
const mount = (mark: () => Promise<void> = async () => undefined) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const wrapper = ({ children }: { children: React.ReactNode }) => <MemoryRouter><QueryClientProvider client={client}>{children}</QueryClientProvider></MemoryRouter>
  const hook = renderHook(() => usePromptInteractions({ queryClient: client, isOnline: false, ...{ expectedOwnerId: owner }, t: key => key, getPromptTexts: row => ({ systemText: row.system_prompt, userText: row.user_prompt }), getPromptKeywords: () => [], getPromptRecordById: () => record, getPromptModifiedAt: () => 0, getPromptUsageCount: () => 0, getPromptLastUsedAt: () => null, editorMarkPromptAsUsed: mark }), { wrapper })
  return { ...hook, client }
}
beforeEach(() => { state.system = undefined; state.quick = undefined; useStoreChatModelSettings.getState().reset() })
afterEach(() => { vi.restoreAllMocks() })

it("C01 applying a saved system prompt clears the previous temporary override", async () => {
  useStoreChatModelSettings.getState().setActiveSettingsScope("selected-model")
  useStoreChatModelSettings.getState().setSystemPrompt("Old temporary instruction")
  const view = mount()
  await act(async () => { await view.result.current.handleUsePromptInChat(record) })
  await act(async () => { await view.result.current.handleInsertChoice("system") })
  expect(state.system).toBe(record.id)
  expect(useStoreChatModelSettings.getState().getEffectiveSettings("selected-model").systemPrompt).toBeUndefined()
  view.unmount(); view.client.clear()
})

it.each(["system", "both", "quick"] as const)("C01 retired %s application cannot repopulate another account's Chat", async choice => {
  let finish!: () => void
  const mark = vi.fn(() => new Promise<void>(resolve => { finish = resolve }))
  const view = mount(mark)
  let applying!: Promise<unknown>
  if (choice !== "quick") await act(async () => { await view.result.current.handleUsePromptInChat(record) })
  await act(async () => { applying = choice === "quick" ? view.result.current.handleUsePromptInChat({ ...record, system_prompt: "" }) : view.result.current.handleInsertChoice(choice) })
  await waitFor(() => expect(mark).toHaveBeenCalledOnce())
  view.unmount()
  window.dispatchEvent(new Event("tldw:auth-principal-changed"))
  await act(async () => { finish(); await applying })
  expect(state.system).toBeUndefined()
  expect(state.quick).toBeUndefined()
  view.client.clear()
})

it("C01 account retirement blocks a pending application even before its editor unmounts", async () => {
  let finish!: () => void
  const mark = vi.fn(() => new Promise<void>(resolve => { finish = resolve }))
  const view = mount(mark)
  await act(async () => { await view.result.current.handleUsePromptInChat(record) })
  let applying!: Promise<unknown>
  await act(async () => { applying = view.result.current.handleInsertChoice("both") })
  await waitFor(() => expect(mark).toHaveBeenCalledOnce())
  window.dispatchEvent(new Event("tldw:auth-principal-changed"))
  await act(async () => { finish(); await applying })
  expect(state.system).toBeUndefined()
  expect(state.quick).toBeUndefined()
  view.unmount(); view.client.clear()
})


it.each([
  { label: "system", choice: "system", system: "Sail from {{port}}.", user: "Resolved user message" },
  { label: "both", choice: "both", system: "Always say ARRR.", user: "Sail from {{port}}." },
  { label: "quick fallback", choice: "quick", system: "Sail from {{port}}.", user: undefined },
  { label: "direct quick", choice: null, system: "", user: "Sail from {{port}}." }
] as const)("C01 unresolved $label application explains the missing values without changing Chat", async input => {
  const warning = vi.spyOn(notification, "warning").mockImplementation(() => {})
  const mark = vi.fn(async () => undefined)
  state.system = "prior-prompt"
  state.quick = "Prior quick text"
  useStoreChatModelSettings.getState().setActiveSettingsScope("selected-model")
  useStoreChatModelSettings.getState().setSystemPrompt("Prior temporary instruction")
  const view = mount(mark)
  await act(async () => { await view.result.current.handleUsePromptInChat({ ...record, system_prompt: input.system, user_prompt: input.user }) })
  if (input.choice) await act(async () => { await view.result.current.handleInsertChoice(input.choice) })
  expect(warning).toHaveBeenCalledOnce()
  expect(warning).toHaveBeenCalledWith(expect.objectContaining({
    message: "managePrompts.template.unresolvedTitle",
    description: "managePrompts.template.unresolvedDescription"
  }))
  expect(mark).not.toHaveBeenCalled()
  expect(state.system).toBe("prior-prompt")
  expect(state.quick).toBe("Prior quick text")
  expect(useStoreChatModelSettings.getState().getEffectiveSettings("selected-model").systemPrompt).toBe("Prior temporary instruction")
  view.unmount(); view.client.clear()
})

it.each([
  { label: "system ignores unused user variables", choice: "system", system: "Always say ARRR.", user: "Sail from {{port}}.", selectedSystem: record.id, selectedQuick: undefined },
  { label: "quick ignores unused system variables", choice: "quick", system: "Sail from {{port}}.", user: "Always say ARRR.", selectedSystem: undefined, selectedQuick: "Always say ARRR." },
  { label: "single-brace built-ins", choice: "both", system: "You are {model}.", user: "Today is {current_date_time}.", selectedSystem: record.id, selectedQuick: "Today is {current_date_time}." }
] as const)("C01 resolved $label applies only the selected fields", async input => {
  const warning = vi.spyOn(notification, "warning").mockImplementation(() => {})
  const mark = vi.fn(async () => undefined)
  const view = mount(mark)
  await act(async () => { await view.result.current.handleUsePromptInChat({ ...record, system_prompt: input.system, user_prompt: input.user }) })
  await act(async () => { await view.result.current.handleInsertChoice(input.choice) })
  expect(warning).not.toHaveBeenCalled()
  expect(mark).toHaveBeenCalledOnce()
  expect(state.system).toBe(input.selectedSystem)
  expect(state.quick).toBe(input.selectedQuick)
  view.unmount(); view.client.clear()
})
