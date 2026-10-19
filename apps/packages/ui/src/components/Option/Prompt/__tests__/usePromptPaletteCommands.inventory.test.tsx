import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { MemoryRouter, useLocation } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { Prompt } from "@/db/dexie/types"
import { CLEAR_TASK_RECIPE } from "@/components/Common/PromptAssist/recipes/built-in-recipes"
import { usePromptPaletteCommands } from "../usePromptPaletteCommands"
import { CommandPaletteHost } from "@/components/Common/CommandPaletteHost"

const mocks = vi.hoisted(() => ({
  getAllPrompts: vi.fn(),
  searchPromptsServer: vi.fn()
}))
vi.mock("@/db/dexie/helpers", () => ({
  getAllPrompts: (...args: unknown[]) => mocks.getAllPrompts(...args),
  getSharedLocalPrompts: (...args: unknown[]) => mocks.getAllPrompts(...args)
}))
vi.mock("@/hooks/usePromptLocalAuthority", () => ({
  usePromptLocalAuthority: () => ({ ownerId: "palette-owner", ready: true })
}))
vi.mock("@/services/prompts-api", () => ({
  searchPromptsServer: (...args: unknown[]) => mocks.searchPromptsServer(...args)
}))
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string, fallback?: string) => fallback ?? key })
}))
vi.mock("@/hooks/keyboard/useShortcutConfig", () => ({
  useShortcutConfig: () => ({ shortcuts: {}, updateShortcut: vi.fn(),
    resetShortcuts: vi.fn(), resetShortcut: vi.fn() })
}))

const STUDIO_UUID = "123e4567-e89b-12d3-a456-426614174001"
const IMPORT_UUID = "123e4567-e89b-12d3-a456-426614174002"
const record = (overrides: Partial<Prompt> = {}): Prompt => ({
  ownerId: "palette-owner", id: STUDIO_UUID, title: "Pirate", name: "Pirate", content: "Say ARRR.",
  system_prompt: "Say ARRR.", user_prompt: "Explain why water wet.",
  is_system: true, createdAt: 1, keywords: ["nautical"],
  author: "Captain", details: "Sailing helper", ...overrides
})
const clients: QueryClient[] = []
function Commands({ query = "", enabled = true }: { query?: string; enabled?: boolean }) {
  const commands = usePromptPaletteCommands(query, enabled)
  const location = useLocation()
  return <>
    <output data-testid="location">{location.pathname + location.search}</output>
    {commands.map((command) => <button key={command.id} onClick={() => command.action?.()}>
      {command.label}
    </button>)}
  </>
}
function mount(query = "", enabled = true) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })
  clients.push(client)
  return render(<QueryClientProvider client={client}>
    <MemoryRouter initialEntries={["/home"]}><Commands query={query} enabled={enabled} /></MemoryRouter>
  </QueryClientProvider>)
}

function Location() {
  const location = useLocation()
  return <output data-testid="location">{location.pathname + location.search}</output>
}

function mountHost() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })
  clients.push(client)
  render(<QueryClientProvider client={client}>
    <MemoryRouter initialEntries={["/home"]}>
      <Location />
      <CommandPaletteHost includePromptCommands />
    </MemoryRouter>
  </QueryClientProvider>)
  return client
}

async function openHost() {
  act(() => { window.dispatchEvent(new CustomEvent("tldw:open-command-palette")) })
  const search = await screen.findByRole("textbox", { name: "Search commands" })
  await waitFor(() => expect(search).toHaveFocus())
  return search
}

beforeEach(() => {
  HTMLElement.prototype.scrollIntoView = vi.fn()
  mocks.getAllPrompts.mockResolvedValue([record()])
  mocks.searchPromptsServer.mockResolvedValue({
    items: [], total_matches: 0, page: 1, per_page: 15
  })
})

describe("actual prompt Host and CommandPalette consumer", () => {
  it.each(["Pirate", "ARRR", "water wet", "nautical", "Captain", "Sailing"])(
    "keeps actual %s inventory matches visible through the palette filter and UUID action",
    async (query) => {
      mountHost()
      const search = await openHost()
      fireEvent.change(search, { target: { value: query } })
      fireEvent.click(await screen.findByRole("option", { name: /^Pirate / }))
      expect(screen.getByTestId("location")).toHaveTextContent(`/prompts?prompt=${STUDIO_UUID}`)
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
      expect(mocks.searchPromptsServer).not.toHaveBeenCalled()
    }
  )

  it("refreshes mounted inventory after canonical create, edit and delete invalidations", async () => {
    let inventory = [record({ favorite: true })]
    mocks.getAllPrompts.mockImplementation(async () => inventory.map((prompt) => ({ ...prompt })))
    const client = mountHost()
    const search = await openHost()
    await screen.findByRole("option", { name: /^Pirate / })

    inventory = [...inventory, record({ id: IMPORT_UUID, title: "Voyage", name: "Voyage" })]
    await act(async () => { await client.invalidateQueries({ queryKey: ["fetchAllPrompts"] }) })
    fireEvent.change(search, { target: { value: "Voyage" } })
    await screen.findByRole("option", { name: /^Voyage / })

    inventory = inventory.map((prompt) => prompt.id === IMPORT_UUID
      ? { ...prompt, title: "Admiral", name: "Admiral" } : prompt)
    await act(async () => { await client.invalidateQueries({ queryKey: ["fetchAllPrompts"] }) })
    await waitFor(() => expect(screen.queryByRole("option", { name: /^Voyage / })).not.toBeInTheDocument())
    fireEvent.change(search, { target: { value: "Admiral" } })
    await screen.findByRole("option", { name: /^Admiral / })

    inventory = inventory.map((prompt) => prompt.id === IMPORT_UUID
      ? { ...prompt, deletedAt: 2 } : prompt)
    await act(async () => { await client.invalidateQueries({ queryKey: ["fetchAllPrompts"] }) })
    await waitFor(() => expect(screen.queryByRole("option", { name: /^Admiral / })).not.toBeInTheDocument())
    expect(screen.getByTestId("location")).toHaveTextContent("/home")
    expect(mocks.searchPromptsServer).not.toHaveBeenCalled()
  })
})
afterEach(() => {
  clients.splice(0).forEach((client) => client.clear())
  vi.clearAllMocks()
})

describe("prompt palette hydrated inventory", () => {
  it.each(["Pirate", "ARRR", "water wet", "nautical", "Captain", "Sailing"])(
    "matches %s through the same local fields and navigates the actual UUID",
    async (query) => {
      mount(query)
      fireEvent.click(await screen.findByRole("button", { name: "Pirate" }))
      expect(screen.getByTestId("location")).toHaveTextContent(`/prompts?prompt=${STUDIO_UUID}`)
      expect(mocks.searchPromptsServer).not.toHaveBeenCalled()
    }
  )

  it("keeps Studio and imported library rows distinct despite equal numeric IDs", async () => {
    mocks.getAllPrompts.mockResolvedValue([
      record({ title: "Studio Pirate", name: "Studio Pirate", serverId: 1, studioPromptId: 1, syncStatus: "synced" }),
      record({ id: IMPORT_UUID, title: "Imported Pirate", name: "Imported Pirate", serverLibraryId: 1,
        serverLibraryUuid: "library-uuid", syncStatus: "local" })
    ])
    mocks.searchPromptsServer.mockResolvedValue({
      items: [{ id: 1, uuid: "library-uuid", name: "Imported Pirate" }],
      total_matches: 1, page: 1, per_page: 15
    })
    mount("Pirate")
    const imported = await screen.findByRole("button", { name: "Imported Pirate" })
    expect(screen.getAllByRole("button")).toHaveLength(2)
    fireEvent.click(imported)
    expect(screen.getByTestId("location")).toHaveTextContent(`/prompts?prompt=${IMPORT_UUID}`)
    fireEvent.click(screen.getByRole("button", { name: "Studio Pirate" }))
    expect(screen.getByTestId("location")).toHaveTextContent(`/prompts?prompt=${STUDIO_UUID}`)
  })

  it("does not offer an unrelated library match as a Studio deep link", async () => {
    mocks.searchPromptsServer.mockResolvedValue({
      items: [{ id: 1, uuid: "unrelated-library", name: "NOT_PRESENT_714" }],
      total_matches: 1, page: 1, per_page: 15
    })
    mount("NOT_PRESENT_714")
    await waitFor(() => expect(mocks.getAllPrompts).toHaveBeenCalledOnce())
    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(screen.queryAllByRole("button")).toHaveLength(0)
    expect(screen.getByTestId("location")).toHaveTextContent("/home")
  })

  it("retains valid recipes and excludes deleted and quarantined records", async () => {
    mocks.getAllPrompts.mockResolvedValue([
      record({ title: "Active Pirate", name: "Active Pirate" }),
      record({ id: "deleted", title: "Deleted Pirate", name: "Deleted Pirate", deletedAt: 1 }),
      record({ id: "recipe", title: "Recipe Pirate", name: "Recipe Pirate", promptFormat: "structured",
        promptSchemaVersion: 2, structuredPromptDefinition: structuredClone(CLEAR_TASK_RECIPE.definition) }),
      record({ id: "quarantined", title: "Quarantined Pirate", name: "Quarantined Pirate", promptFormat: "structured",
        promptSchemaVersion: 2, structuredPromptDefinition: { schema_version: 2, definition_kind: "single_text_recipe" } })
    ])
    mount("Pirate")
    await screen.findByRole("button", { name: "Recipe Pirate" })
    expect(screen.getAllByRole("button").map((button) => button.textContent)).toEqual(["Active Pirate", "Recipe Pirate"])
  })

  it("keeps the search result cap", async () => {
    mocks.getAllPrompts.mockResolvedValue(Array.from({ length: 20 }, (_, index) =>
      record({ id: `uuid-${index}`, title: `Pirate ${index}`, name: `Pirate ${index}` })))
    mount("Pirate")
    await screen.findByRole("button", { name: "Pirate 0" })
    expect(screen.getAllByRole("button")).toHaveLength(15)
  })

  it.each(["", "P"])("retains favorite/recent static mode for short query %j", async (query) => {
    mocks.getAllPrompts.mockResolvedValue([
      record({ title: "Favorite", name: "Favorite", favorite: true, lastUsedAt: 20 }),
      record({ id: IMPORT_UUID, title: "Recent", name: "Recent", lastUsedAt: 10 }),
      record({ id: "unused", title: "Unused", name: "Unused" })
    ])
    mount(query)
    await screen.findByRole("button", { name: "Favorite" })
    expect(screen.getAllByRole("button").map((button) => button.textContent)).toEqual(["Favorite", "Recent"])
    fireEvent.click(screen.getByRole("button", { name: "Favorite" }))
    expect(screen.getByTestId("location")).toHaveTextContent(`/prompts?edit=${STUDIO_UUID}`)
  })

  it("does not hydrate or offer commands when disabled", async () => {
    mount("Pirate", false)
    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(screen.queryAllByRole("button")).toHaveLength(0)
    expect(mocks.getAllPrompts).not.toHaveBeenCalled()
    expect(mocks.searchPromptsServer).not.toHaveBeenCalled()
  })
})
