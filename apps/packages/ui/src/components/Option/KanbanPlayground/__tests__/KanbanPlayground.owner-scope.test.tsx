// @vitest-environment jsdom
import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { App, ConfigProvider, message } from "antd"
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { KanbanPlayground } from "../index"
import { createKanbanClient } from "@/services/kanban"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"

const transport = vi.hoisted(() => ({
  canonical: vi.fn(), getConfig: vi.fn(), get: vi.fn(),
  fetch: vi.fn<(input: RequestInfo | URL, init?: RequestInit) => Promise<Response>>()
}))
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({ useCanonicalConnectionConfig: transport.canonical }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: { getConfig: transport.getConfig } }))
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({ get: transport.get, set: vi.fn(), remove: vi.fn(), watch: vi.fn(), unwatch: vi.fn() }),
  safeStorageSerde: { serialize: (value: unknown) => value, deserialize: (value: unknown) => value }
}))
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null, isCookieSessionConfigInvalidated: () => false
}))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }))
// Native DnD positioning is outside this owner/transport boundary.
vi.mock("@dnd-kit/react", () => ({ DragDropProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>, DragOverlay: () => null }))
vi.mock("@dnd-kit/react/sortable", () => ({ useSortable: () => ({ ref: vi.fn(), handleRef: vi.fn(), isDragging: false }) }))

const config = (owner = 2, serverUrl = "https://kanban.example.test", revision = 1): TldwConfig => ({
  serverUrl, authMode: "multi-user",
  accessToken: `fixture.${btoa(JSON.stringify({ sub: String(owner), rev: revision }))}.signature`
})
let current: TldwConfig
let authorityLoading: boolean
let client: QueryClient
const owner = (init?: RequestInit) => {
  const token = new Headers(init?.headers).get("Authorization")?.replace(/^Bearer /, "")
  if (!token) throw new Error("Fixture requires an explicit authenticated owner")
  return JSON.parse(atob(token.split(".")[1])).sub === "2" ? "Alice" : "Bob"
}
const card = (name: string) => ({
  id: 1, uuid: `${name}-card-uuid`, list_id: name === "Alice" ? 2 : 1, title: `${name} private card`,
  description: `${name}-CARD-${name === "Alice" ? "ORBIT-742" : "COMET-913"}`,
  due_date: null, due_complete: false, priority: null, archived: false,
  created_at: "2026-10-09T10:00:00Z", updated_at: "2026-10-09T10:00:00Z", labels: []
})
const board = (name: string) => ({
  id: 1, uuid: `${name}-board-uuid`, name: "Shared board name", description: `${name} board`,
  archived: false, version: 1, created_at: "2026-10-09T10:00:00Z", updated_at: "2026-10-09T10:00:00Z",
  lists: [1, 2].map((id) => ({ id, uuid: `${name}-list-${id}`, board_id: 1, name: id === 1 ? "To Do" : "Done",
    position: id, archived: false, cards: card(name).list_id === id ? [card(name)] : [] }))
})
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { "Content-Type": "application/json" } })
const response = (url: string, init?: RequestInit) => {
  const path = new URL(url).pathname
  const name = owner(init)
  if (path === "/api/v1/kanban/boards") return json({ boards: [board(name)], pagination: { total: 1 } })
  if (path === "/api/v1/kanban/boards/1") return json(board(name))
  if (path.endsWith("/comments")) return json({ comments: [{ id: 1, content: `${name} private comment`, created_at: "2026-10-09T10:00:00Z" }], pagination: { has_more: false } })
  if (path.endsWith("/checklists")) return json({ checklists: [] })
  if (path.endsWith("/labels")) return json([])
  if (path === "/api/v1/kanban/cards/1" && init?.method === "PATCH") return json({ ...card(name), ...JSON.parse(String(init.body)) })
  throw new Error(`Unexpected fixture request: ${init?.method} ${path}`)
}
const tree = () => <QueryClientProvider client={client}><ConfigProvider theme={{ token: { motion: false } }}><App><KanbanPlayground /></App></ConfigProvider></QueryClientProvider>
async function openCard(name = "Alice") {
  fireEvent.click(await screen.findByRole("button", { name: /Shared board name/ }))
  fireEvent.click(await screen.findByText(`${name} private card`))
  const dialog = (await screen.findByText("Edit Card")).closest<HTMLElement>("[role='dialog']")!
  expect(dialog).not.toBeNull()
  await waitFor(() => expect(dialog).toBeVisible())
  expect(within(dialog).getByPlaceholderText("Add a description...")).toHaveValue(card(name).description)
  await within(dialog).findByText(`${name} private comment`)
  return dialog
}

function openActions(container: HTMLElement) {
  const button = Array.from(container.querySelectorAll("button")).find((node) => node.querySelector(".lucide-ellipsis-vertical, .lucide-more-vertical"))
  if (!button) throw new Error("Kanban actions button missing")
  fireEvent.click(button)
}
async function openImport(container: HTMLElement) {
  await screen.findByRole("button", { name: /Shared board name/ })
  openActions(container)
  fireEvent.click(await screen.findByText("Import Board..."))
  const input = document.querySelector<HTMLInputElement>('input[type="file"]')!
  const data = { format: "tldw_kanban_v1", board: { name: "Alice private import" }, lists: [] }
  fireEvent.change(input, { target: { files: [new File([JSON.stringify(data)], "alice-private.json", { type: "application/json" })] } })
  const preview = await screen.findByText("File Preview")
  return preview.closest<HTMLElement>("[role='dialog']")!
}

beforeAll(() => {
  ConfigProvider.config({ holderRender: (children) =>
    <ConfigProvider theme={{ token: { motion: false } }}>{children}</ConfigProvider> })
})
afterAll(() => { ConfigProvider.config({ holderRender: undefined }) })

beforeEach(() => {
  vi.resetAllMocks()
  current = config(); authorityLoading = false
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  transport.canonical.mockImplementation(() => ({ config: current, authorityLoading }))
  transport.getConfig.mockImplementation(async () => current)
  transport.get.mockImplementation(async (key: string) => key === "tldwConfig" ? current : null)
  transport.fetch.mockImplementation(async (input, init) => response(String(input), init))
  vi.stubGlobal("fetch", transport.fetch)
})
afterEach(() => {
  cleanup(); client.clear(); message.destroy()
  vi.restoreAllMocks(); vi.unstubAllGlobals()
})

describe("Kanban workspace owner lifetime", () => {
  it.each(["account", "server"])("retires a held Alice editor after %s changes and loads Bob's colliding numeric IDs", async (change) => {
    const view = render(tree())
    const dialog = await openCard()
    current = change === "account" ? config(3) : config(3, "https://bob.example.test")
    view.rerender(tree())
    await waitFor(() => expect(dialog).not.toBeInTheDocument())
    expect(screen.queryByDisplayValue(card("Alice").description)).not.toBeInTheDocument()
    const successor = await openCard("Bob")
    expect(within(successor).queryByText("Alice private comment")).not.toBeInTheDocument()
    expect(transport.fetch.mock.calls.filter(([, init]) => init?.method === "PATCH")).toEqual([])
  })

  it("removes private state while canonical authority is unresolved", async () => {
    const view = render(tree())
    const dialog = await openCard()
    authorityLoading = true
    view.rerender(tree())
    expect(dialog).not.toBeInTheDocument()
    expect(screen.queryByText("Alice private card")).not.toBeInTheDocument()
    expect(transport.fetch.mock.calls.filter(([, init]) => init?.method === "PATCH")).toEqual([])
  })

  it("aborts a pending Alice save and ignores its late ACK after Bob opens card 1", async () => {
    let finish!: (value: Response) => void
    transport.fetch.mockImplementation(async (input, init) =>
      init?.method === "PATCH" ? await new Promise<Response>((resolve) => { finish = resolve }) : response(String(input), init))
    const view = render(tree())
    const dialog = await openCard()
    fireEvent.change(within(dialog).getByPlaceholderText("Add a description..."), { target: { value: "Alice revised private text" } })
    fireEvent.click(within(dialog).getByRole("button", { name: "Save Changes" }))
    await waitFor(() => expect(transport.fetch.mock.calls.filter(([, init]) => init?.method === "PATCH")).toHaveLength(1))
    const saved = transport.fetch.mock.calls.find(([, init]) => init?.method === "PATCH")!
    expect(String(saved[0])).toBe(`${config().serverUrl}/api/v1/kanban/cards/1`)
    expect(new Headers(saved[1]?.headers).get("Authorization")).toBe(`Bearer ${config().accessToken}`)
    current = config(3)
    view.rerender(tree())
    await waitFor(() => expect(dialog).not.toBeInTheDocument())
    const bob = await openCard("Bob")
    expect(saved[1]?.signal?.aborted).toBe(true)
    await act(async () => finish(json({ ...card("Alice"), description: "Alice revised private text" })))
    expect(within(bob).getByPlaceholderText("Add a description...")).toHaveValue(card("Bob").description)
    expect(screen.queryByDisplayValue("Alice revised private text")).not.toBeInTheDocument()
    expect(transport.fetch.mock.calls.filter(([, init]) => init?.method === "PATCH")).toHaveLength(1)
  })

  it("keeps a same-owner editor and dispatches the refreshed token with the unchanged authored patch", async () => {
    const view = render(tree())
    const dialog = await openCard()
    current = config(2, current.serverUrl, 2)
    view.rerender(tree())
    expect(within(dialog).getByPlaceholderText("Add a description...")).toHaveValue(card("Alice").description)
    fireEvent.change(within(dialog).getByPlaceholderText("Add a description..."), { target: { value: "Refreshed owner text" } })
    fireEvent.click(within(dialog).getByRole("button", { name: "Save Changes" }))
    await waitFor(() => expect(transport.fetch.mock.calls.filter(([, init]) => init?.method === "PATCH")).toHaveLength(1))
    const [url, init] = transport.fetch.mock.calls.find(([, request]) => request?.method === "PATCH")!
    expect(String(url)).toBe(`${current.serverUrl}/api/v1/kanban/cards/1`)
    expect(new Headers(init?.headers).get("Authorization")).toBe(`Bearer ${current.accessToken}`)
    expect(JSON.parse(String(init?.body))).toEqual({ description: "Refreshed owner text" })
  })

  it("separates colliding comments, checklist details and label queries by the current owner", async () => {
    transport.fetch.mockImplementation(async (input, init) => {
      const path = new URL(String(input)).pathname
      const name = owner(init)
      if (path.endsWith("/checklists")) return json({ checklists: [{ id: 1, card_id: 1, name: name + " private checklist" }] })
      if (path === "/api/v1/kanban/checklists/1") return json({ id: 1, card_id: 1, name: name + " private checklist", items: [] })
      if (path.endsWith("/labels")) return json([{ id: 1, name: name + " private label", color: "#336699" }])
      return response(String(input), init)
    })
    const view = render(tree())
    const alice = await openCard()
    await within(alice).findByText("Alice private checklist")
    fireEvent.click(within(alice).getByRole("button", { name: "Add Label" }))
    await screen.findByText("Alice private label")
    current = config(3); view.rerender(tree())
    await waitFor(() => expect(alice).not.toBeInTheDocument())
    const bob = await openCard("Bob")
    await within(bob).findByText("Bob private checklist")
    fireEvent.click(within(bob).getByRole("button", { name: "Add Label" }))
    await screen.findByText("Bob private label")
    expect(screen.queryByText("Alice private label")).not.toBeInTheDocument()
    expect(screen.queryByText("Alice private checklist")).not.toBeInTheDocument()
    const details = transport.fetch.mock.calls.filter(([url]) => new URL(String(url)).pathname === "/api/v1/kanban/checklists/1")
    expect(details.map(([, init]) => new Headers(init?.headers).get("Authorization"))).toEqual(["Bearer " + config().accessToken, "Bearer " + current.accessToken])
  })

  it("does not download a retired export that resolves after an account change", async () => {
    let finish!: (value: Response) => void
    transport.fetch.mockImplementation(async (input, init) =>
      String(input).includes("/export") ? await new Promise<Response>((resolve) => { finish = resolve }) : response(String(input), init))
    const create = vi.fn(() => "blob:kanban-fixture")
    const revoke = vi.fn()
    const NativeURL = URL
    vi.stubGlobal("URL", class extends NativeURL { static createObjectURL = create; static revokeObjectURL = revoke })
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {})
    const view = render(tree())
    fireEvent.click(await screen.findByRole("button", { name: /Shared board name/ }))
    await screen.findByText("Alice private card")
    openActions(view.container)
    fireEvent.click(await screen.findByText("Export Board..."))
    await waitFor(() => expect(transport.fetch.mock.calls.some(([url]) => String(url).includes("/export"))).toBe(true))
    const exporting = transport.fetch.mock.calls.find(([url]) => String(url).includes("/export"))!
    current = config(3); view.rerender(tree())
    await screen.findByRole("button", { name: /Shared board name/ })
    expect(exporting[1]?.signal?.aborted).toBe(true)
    await act(async () => finish(json({ format: "tldw_kanban_v1", board: board("Alice") })))
    expect(create).not.toHaveBeenCalled()
    expect(click).not.toHaveBeenCalled()
    expect(screen.queryByText("Board exported")).not.toBeInTheDocument()
  })

  it("retires an import preview without posting its private source into a successor", async () => {
    const view = render(tree())
    const dialog = await openImport(view.container)
    expect(within(dialog).getByText("Alice private import")).toBeInTheDocument()
    current = config(3); view.rerender(tree())
    await waitFor(() => expect(dialog).not.toBeInTheDocument())
    expect(screen.queryByText("Alice private import")).not.toBeInTheDocument()
    expect(transport.fetch.mock.calls.filter(([, init]) => init?.method === "POST")).toEqual([])
  })

  it("aborts an admitted import and refuses late result selection or success notice in Bob's workspace", async () => {
    let finish!: (value: Response) => void
    transport.fetch.mockImplementation(async (input, init) =>
      new URL(String(input)).pathname.endsWith("/import") ? await new Promise<Response>((resolve) => { finish = resolve }) : response(String(input), init))
    const success = vi.spyOn(message, "success")
    const view = render(tree())
    const dialog = await openImport(view.container)
    fireEvent.click(within(dialog).getByRole("button", { name: "Import Board" }))
    await waitFor(() => expect(transport.fetch.mock.calls.filter(([, init]) => init?.method === "POST")).toHaveLength(1))
    const [url, init] = transport.fetch.mock.calls.find(([, value]) => value?.method === "POST")!
    expect(String(url)).toBe(config().serverUrl + "/api/v1/kanban/boards/import")
    expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer " + config().accessToken)
    expect(JSON.parse(String(init?.body)).data.board.name).toBe("Alice private import")
    current = config(3); view.rerender(tree())
    await waitFor(() => expect(dialog).not.toBeInTheDocument())
    await screen.findByRole("button", { name: /Shared board name/ })
    expect(init?.signal?.aborted).toBe(true)
    await act(async () => finish(json({ board: board("Alice"), import_stats: {} })))
    expect(success).not.toHaveBeenCalled()
    expect(screen.queryByText("Alice private card")).not.toBeInTheDocument()
    expect(transport.fetch.mock.calls.filter(([, value]) => value?.method === "POST")).toHaveLength(1)
  })

  it("retires the archived-items drawer and its private restore controls", async () => {
    transport.fetch.mockImplementation(async (input, init) => {
      const url = new URL(String(input))
      if (url.pathname === "/api/v1/kanban/boards" && url.searchParams.get("include_archived") === "true")
        return json({ boards: [{ ...board(owner(init)), id: 2, name: owner(init) + " private archive", archived: true }] })
      return response(String(input), init)
    })
    const view = render(tree())
    await screen.findByRole("button", { name: /Shared board name/ })
    openActions(view.container)
    fireEvent.click(await screen.findByText("Archived Items"))
    const archived = await screen.findByText("Alice private archive")
    const dialog = archived.closest<HTMLElement>("[role='dialog']")!
    expect(within(dialog).getByRole("button", { name: "Restore" })).toBeInTheDocument()
    current = config(3); view.rerender(tree())
    await waitFor(() => expect(dialog).not.toBeInTheDocument())
    expect(screen.queryByText("Alice private archive")).not.toBeInTheDocument()
    expect(transport.fetch.mock.calls.filter(([, init]) => init?.method === "POST")).toEqual([])
  })

  it("closes the owned archive Undo toast and refuses its retained callback after retirement", async () => {
    transport.fetch.mockImplementation(async (input, init) =>
      new URL(String(input)).pathname.endsWith("/archive") ? json(card("Alice")) : response(String(input), init))
    const opened = vi.spyOn(message, "open")
    const view = render(tree())
    const dialog = await openCard()
    await act(async () => { fireEvent.click(within(dialog).getByTitle("Archive card")) })
    await waitFor(() => expect(opened).toHaveBeenCalled())
    expect(transport.fetch.mock.calls.filter(([url]) => String(url).endsWith("/archive"))).toHaveLength(1)
    await screen.findByRole("button", { name: "Undo" })
    const content = opened.mock.calls.find(([args]) => React.isValidElement(args.content))![0].content as React.ReactElement<{ children: React.ReactNode[] }>
    const undo = React.Children.toArray(content.props.children).find((child) =>
      React.isValidElement<{ children?: React.ReactNode }>(child) && child.props.children === "Undo") as React.ReactElement<{ onClick: () => void }>
    expect(undo).toBeDefined()
    current = config(3); view.rerender(tree())
    await waitFor(() => expect(screen.queryByRole("button", { name: "Undo" })).not.toBeInTheDocument())
    await act(async () => undo.props.onClick())
    expect(transport.fetch.mock.calls.filter(([url]) => String(url).includes("/unarchive"))).toEqual([])
    expect(transport.fetch.mock.calls.filter(([, init]) => init?.method === "POST")).toHaveLength(1)
  })

  it.each(["comments", "checklists"])("refuses a retired %s continuation before its next nested request", async (kind) => {
    const controller = new AbortController()
    const captured = { ...current }
    const api = createKanbanClient(() => ({ configSnapshot: captured, abortSignal: controller.signal }))
    let finish!: (value: Response) => void
    transport.fetch.mockImplementationOnce(() => new Promise<Response>((resolve) => { finish = resolve }))
    const pending = kind === "comments" ? api.listComments(1) : api.listChecklists(1)
    const refused = expect(pending).rejects.toMatchObject({ name: "AbortError" })
    await waitFor(() => expect(transport.fetch).toHaveBeenCalledOnce())
    const [url, init] = transport.fetch.mock.calls[0]
    expect(String(url)).toContain("/api/v1/kanban/cards/1/" + kind)
    expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer " + captured.accessToken)
    controller.abort(); current = config(3)
    finish(json(kind === "comments"
      ? { comments: [{ id: 1, content: "Alice private page" }], pagination: { has_more: true, offset: 0 } }
      : { checklists: [{ id: 1, name: "Alice private checklist" }] }))
    await refused
    expect(transport.fetch).toHaveBeenCalledOnce()
  })

  it("preserves the normal legacy single-user request and expected-version guard", async () => {
    current = { serverUrl: config().serverUrl, authMode: "single-user", apiKey: "synthetic-kanban-owner-key" }
    const api = createKanbanClient(() => ({ configSnapshot: { ...current }, abortSignal: new AbortController().signal }))
    transport.fetch.mockImplementation(async () => json(board("Alice")))
    await api.updateBoard(1, { name: "Legacy owner board" }, 7)
    expect(transport.fetch).toHaveBeenCalledOnce()
    const [url, init] = transport.fetch.mock.calls[0]
    expect(String(url)).toBe(current.serverUrl + "/api/v1/kanban/boards/1")
    expect(new Headers(init?.headers).get("X-API-KEY")).toBe(current.apiKey)
    expect(new Headers(init?.headers).get("X-Expected-Version")).toBe("7")
    expect(JSON.parse(String(init?.body))).toEqual({ name: "Legacy owner board" })
  })

  it("keeps a current-owner refused save dirty and surfaces its existing bounded error", async () => {
    transport.fetch.mockImplementation(async (input, init) =>
      init?.method === "PATCH" ? json({ detail: "card_not_found" }, 404) : response(String(input), init))
    const error = vi.spyOn(message, "error")
    render(tree())
    const dialog = await openCard()
    fireEvent.change(within(dialog).getByPlaceholderText("Add a description..."), { target: { value: "Unaccepted authored text" } })
    fireEvent.click(within(dialog).getByRole("button", { name: "Save Changes" }))
    await waitFor(() => expect(error).toHaveBeenCalledWith("Failed to update card. Please try again."))
    expect(within(dialog).getByPlaceholderText("Add a description...")).toHaveValue("Unaccepted authored text")
    expect(within(dialog).getByRole("button", { name: "Save Changes" })).toBeEnabled()
    expect(transport.fetch.mock.calls.filter(([, init]) => init?.method === "PATCH")).toHaveLength(1)
  })
})
