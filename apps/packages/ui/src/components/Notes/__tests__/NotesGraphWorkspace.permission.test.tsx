import {
  QueryClient,
  QueryClientProvider,
  onlineManager
} from "@tanstack/react-query"
import {
  act,
  cleanup,
  fireEvent,
  render,
  renderHook,
  screen,
  waitFor
} from "@testing-library/react"
import { createInstance } from "i18next"
import React from "react"
import { I18nextProvider } from "react-i18next"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import commonEn from "../../../assets/locale/en/common.json"
import optionEn from "../../../assets/locale/en/option.json"
import ICUWithInterpolation from "../../../i18n/icu-format"
import NotesGraphWorkspace from "../NotesGraphWorkspace"
import { useNotesGraphWorkspace } from "../hooks/useNotesGraphWorkspace"

const { request } = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: request }))
// Cytoscape requires a real canvas. Keep workspace, query hook, service validation,
// toolbar, inspector and translation real; assertions target those consumers.
vi.mock("../NotesGraphCanvas", async () => {
  const ReactModule = await import("react")
  return { default: ReactModule.forwardRef(() => null) }
})

const graph = (more = false) => ({
  nodes: [
    {
      id: "note:a",
      type: "note",
      label: "Private Alpha note",
      created_at: null,
      deleted: false,
      degree: 0,
      tag_count: 0,
      primary_source_id: null
    }
  ],
  edges: [],
  truncated: false,
  truncated_by: [],
  has_more: more,
  cursor: more ? "next-page" : null,
  limits: { max_nodes: 120, max_edges: 480, max_degree: 40 },
  radius_cap_applied: false,
  active_note_count: 1,
  all_notes_note_cap: 100,
  all_notes_eligible: true,
  suggestions_authorized: false
})
const failure = (status: number) =>
  Object.assign(new Error("server diagnostic must not render"), { status })
const clients: QueryClient[] = []
const props = {
  authorityScope: "alice",
  isOnline: true,
  initialFocusNoteId: "a",
  selectedNoteId: "a",
  hasActiveNotes: true,
  onSelectNote: vi.fn(),
  onCreateNote: vi.fn()
}
const permissionText =
  "Notes graph is unavailable for this account. Ask an administrator for access."

async function setup(overrides = {}) {
  const i18n = createInstance()
  await i18n.use(ICUWithInterpolation).init({
    lng: "en",
    fallbackLng: false,
    ns: ["option", "common"],
    defaultNS: "option",
    resources: { en: { option: optionEn, common: commonEn } },
    interpolation: { escapeValue: false }
  })
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } }
  })
  clients.push(client)
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <I18nextProvider i18n={i18n}>
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    </I18nextProvider>
  )
  const view = render(<NotesGraphWorkspace {...props} {...overrides} />, {
    wrapper
  })
  return { ...view, client, wrapper }
}

function hookWorkspace() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: Infinity } }
  })
  clients.push(client)
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
  const useGraph = ({
    radius = 1,
    authorityScope = "alice"
  }: {
    radius?: 1 | 2
    authorityScope?: string
  }) =>
    useNotesGraphWorkspace({
      authorityScope,
      enabled: true,
      isOnline: true,
      initialFocusNoteId: "a",
      radius
    })
  return { client, wrapper, useGraph }
}

beforeEach(() => {
  request.mockReset()
  onlineManager.setOnline(true)
})
afterEach(() => {
  cleanup()
  clients.forEach((client) => client.clear())
  clients.length = 0
  onlineManager.setOnline(true)
})

describe("Notes Graph permission boundary", () => {
  it("explains the actual service403 without claiming an empty graph or rendering server diagnostics", async () => {
    request.mockRejectedValue(failure(403))
    await setup()
    expect(await screen.findByRole("alert")).toHaveTextContent(permissionText)
    expect(
      screen.queryByText("Could not load the notes graph.")
    ).not.toBeInTheDocument()
    expect(
      screen.queryByText(/No relationships are visible/)
    ).not.toBeInTheDocument()
    expect(screen.queryByText(/server diagnostic/)).not.toBeInTheDocument()
    expect(
      screen.queryByTestId("notes-graph-primary-view")
    ).not.toBeInTheDocument()
    expect(request).toHaveBeenCalledTimes(1)
  })

  it("does not automatically request a denied graph again on reconnect", async () => {
    request.mockRejectedValue(failure(403))
    const view = await setup()
    await screen.findByRole("alert")
    view.rerender(<NotesGraphWorkspace {...props} isOnline={false} />)
    await act(async () => {
      onlineManager.setOnline(false)
    })
    view.rerender(<NotesGraphWorkspace {...props} isOnline />)
    await act(async () => {
      onlineManager.setOnline(true)
      await new Promise((resolve) => setTimeout(resolve, 30))
    })
    expect(request).toHaveBeenCalledTimes(1)
    expect(screen.getByRole("alert")).toHaveTextContent(permissionText)
    expect(
      screen.queryByTestId("notes-graph-offline-state")
    ).not.toBeInTheDocument()
  })

  it("lets an explicit refresh recover after permission changes", async () => {
    request.mockRejectedValueOnce(failure(403)).mockResolvedValue(graph())
    await setup()
    expect(await screen.findByRole("alert")).toHaveTextContent(permissionText)
    fireEvent.click(screen.getByRole("button", { name: "Refresh graph" }))
    expect(
      await screen.findByTestId("notes-graph-primary-view")
    ).toBeInTheDocument()
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    expect(
      screen.getByRole("searchbox", { name: "Search loaded nodes" })
    ).toBeInTheDocument()
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 30))
    })
    expect(request).toHaveBeenCalledTimes(2)
  })

  it("removes cached graph data, search and inspector after a refresh403", async () => {
    request.mockResolvedValueOnce(graph()).mockRejectedValue(failure(403))
    await setup()
    await screen.findByTestId("notes-graph-primary-view")
    fireEvent.change(
      screen.getByRole("searchbox", { name: "Search loaded nodes" }),
      { target: { value: "Private" } }
    )
    expect(screen.getAllByText("Private Alpha note").length).toBeGreaterThan(0)
    fireEvent.click(screen.getByRole("button", { name: "Refresh graph" }))
    expect(await screen.findByRole("alert")).toHaveTextContent(permissionText)
    expect(
      screen.queryByTestId("notes-graph-primary-view")
    ).not.toBeInTheDocument()
    expect(
      screen.queryByTestId("notes-graph-inspector-region")
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("searchbox", { name: "Search loaded nodes" })
    ).not.toBeInTheDocument()
    expect(screen.queryByText("Private Alpha note")).not.toBeInTheDocument()
    expect(
      screen.queryByTestId("notes-graph-degraded-state")
    ).not.toBeInTheDocument()
  })

  it("does not carry Alice's denial into a replacement authority", async () => {
    request.mockRejectedValueOnce(failure(403)).mockResolvedValue(graph())
    const view = await setup()
    await screen.findByRole("alert")
    view.rerender(<NotesGraphWorkspace {...props} authorityScope="admin" />)
    expect(
      await screen.findByTestId("notes-graph-primary-view")
    ).toBeInTheDocument()
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("withdraws cached pages when the explicit cursor request receives403", async () => {
    request.mockResolvedValueOnce(graph(true)).mockRejectedValue(failure(403))
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false, gcTime: 0 } }
    })
    clients.push(client)
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    )
    const { result } = renderHook(
      () =>
        useNotesGraphWorkspace({
          authorityScope: "alice",
          enabled: true,
          isOnline: true,
          initialFocusNoteId: "a"
        }),
      { wrapper }
    )
    await waitFor(() => expect(result.current.graph?.nodes).toHaveLength(1))
    // Catch at the command boundary so the original implementation's rejection
    // cannot become a test-runner unhandled error instead of our causal assertion.
    await act(async () => {
      await result.current.expand().catch(() => null)
    })
    await waitFor(() => expect(result.current.graph).toBeNull())
    expect(result.current.searchResults).toEqual([])
    expect(result.current.canExpand).toBe(false)
  })

  it("keeps the current denial when a previous authority's request rejects late", async () => {
    let rejectAlice!: (error: Error) => void
    request
      .mockReturnValueOnce(
        new Promise((_resolve, reject) => {
          rejectAlice = reject
        })
      )
      .mockRejectedValueOnce(failure(403))
      .mockResolvedValue(graph())
    const view = await setup()
    view.rerender(<NotesGraphWorkspace {...props} authorityScope="bob" />)
    expect(await screen.findByRole("alert")).toHaveTextContent(permissionText)
    await act(async () => {
      rejectAlice(failure(403))
      await new Promise((resolve) => setTimeout(resolve, 30))
    })
    expect(screen.getByRole("alert")).toHaveTextContent(permissionText)
    expect(request).toHaveBeenCalledTimes(2)
    expect(
      screen.queryByTestId("notes-graph-primary-view")
    ).not.toBeInTheDocument()
  })

  it.each(["refresh", "expand"] as const)(
    "does not redisplay denied cached pages when reopening after %s",
    async (command) => {
      request.mockResolvedValueOnce(graph(true)).mockRejectedValue(failure(403))
      const client = new QueryClient({
        defaultOptions: { queries: { retry: false, gcTime: Infinity } }
      })
      clients.push(client)
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      )
      const useGraph = () =>
        useNotesGraphWorkspace({
          authorityScope: "alice",
          enabled: true,
          isOnline: true,
          initialFocusNoteId: "a"
        })
      const first = renderHook(useGraph, { wrapper })
      await waitFor(() =>
        expect(first.result.current.graph?.nodes).toHaveLength(1)
      )
      await act(async () => {
        await first.result.current[command]().catch(() => null)
      })
      await waitFor(() => expect(first.result.current.graph).toBeNull())
      first.unmount()
      const reopened = renderHook(useGraph, { wrapper })
      expect(reopened.result.current.graph).toBeNull()
      await waitFor(() =>
        expect(reopened.result.current.isPermissionDenied).toBe(true)
      )
    }
  )

  it.each([false, true])(
    "prevents a late base success from restoring denied data on reopen (old observer retained: %s)",
    async (keepOldObserver) => {
      let resolveOld!: (value: ReturnType<typeof graph>) => void
      request
        .mockReturnValueOnce(
          new Promise((resolve) => {
            resolveOld = resolve
          })
        )
        .mockRejectedValue(failure(403))
      const { wrapper, useGraph } = hookWorkspace()
      const first = renderHook(useGraph, {
        wrapper,
        initialProps: { radius: 1 as 1 | 2 }
      })
      await waitFor(() => expect(request).toHaveBeenCalledTimes(1))
      const denied = keepOldObserver
        ? renderHook(useGraph, { wrapper, initialProps: { radius: 2 } })
        : first
      if (!keepOldObserver) first.rerender({ radius: 2 })
      await waitFor(() =>
        expect(denied.result.current.isPermissionDenied).toBe(true)
      )
      await act(async () => {
        resolveOld(graph())
        await new Promise((resolve) => setTimeout(resolve, 30))
      })
      expect(denied.result.current.graph).toBeNull()
      first.unmount()
      if (keepOldObserver) denied.unmount()
      const reopened = renderHook(useGraph, {
        wrapper,
        initialProps: { radius: 1 }
      })
      expect(reopened.result.current.graph).toBeNull()
      await waitFor(() =>
        expect(reopened.result.current.isPermissionDenied).toBe(true)
      )
    }
  )

  it("settles a revoked cursor command without letting its late success restore pages", async () => {
    let resolveCursor!: (value: ReturnType<typeof graph>) => void
    request
      .mockResolvedValueOnce(graph(true))
      .mockReturnValueOnce(
        new Promise((resolve) => {
          resolveCursor = resolve
        })
      )
      .mockRejectedValue(failure(403))
    const { wrapper, useGraph } = hookWorkspace()
    const first = renderHook(useGraph, { wrapper, initialProps: { radius: 1 } })
    await waitFor(() => expect(first.result.current.canExpand).toBe(true))
    let settled = false
    let cursorResult: unknown = "pending"
    let cursorPromise!: Promise<unknown>
    act(() => {
      cursorPromise = first.result.current.expand().then(
        (value) => {
          settled = true
          cursorResult = value
        },
        (error) => {
          settled = true
          cursorResult = error
        }
      )
    })
    await waitFor(() => expect(request).toHaveBeenCalledTimes(2))
    const denied = renderHook(useGraph, {
      wrapper,
      initialProps: { radius: 2 }
    })
    await waitFor(() =>
      expect(denied.result.current.isPermissionDenied).toBe(true)
    )
    await waitFor(() => expect(settled).toBe(true))
    expect(cursorResult).toBeNull()
    await act(async () => {
      resolveCursor(graph())
      await cursorPromise
      await new Promise((resolve) => setTimeout(resolve, 30))
    })
    expect(first.result.current.graph).toBeNull()
    first.unmount()
    denied.unmount()
    const reopened = renderHook(useGraph, {
      wrapper,
      initialProps: { radius: 1 }
    })
    expect(reopened.result.current.graph).toBeNull()
  })

  it("ignores a canceled request's late403 after explicit permission recovery", async () => {
    let rejectOld!: (error: Error) => void
    request
      .mockReturnValueOnce(
        new Promise((_resolve, reject) => {
          rejectOld = reject
        })
      )
      .mockRejectedValueOnce(failure(403))
      .mockResolvedValue(graph())
    const { wrapper, useGraph } = hookWorkspace()
    renderHook(useGraph, { wrapper, initialProps: { radius: 1 } })
    await waitFor(() => expect(request).toHaveBeenCalledTimes(1))
    const current = renderHook(useGraph, {
      wrapper,
      initialProps: { radius: 2 }
    })
    await waitFor(() =>
      expect(current.result.current.isPermissionDenied).toBe(true)
    )
    await act(async () => {
      await current.result.current.refresh()
    })
    await waitFor(() =>
      expect(current.result.current.graph?.nodes).toHaveLength(1)
    )
    await act(async () => {
      rejectOld(failure(403))
      await new Promise((resolve) => setTimeout(resolve, 30))
    })
    expect(current.result.current.graph?.nodes).toHaveLength(1)
    expect(current.result.current.isPermissionDenied).toBe(false)
    expect(request).toHaveBeenCalledTimes(3)
  })

  it("leaves another authority's pending query and cached result available", async () => {
    let resolveBob!: (value: ReturnType<typeof graph>) => void
    request
      .mockReturnValueOnce(
        new Promise((resolve) => {
          resolveBob = resolve
        })
      )
      .mockRejectedValue(failure(403))
    const { wrapper, useGraph } = hookWorkspace()
    const bob = renderHook(useGraph, {
      wrapper,
      initialProps: { authorityScope: "bob" }
    })
    await waitFor(() => expect(request).toHaveBeenCalledTimes(1))
    const alice = renderHook(useGraph, {
      wrapper,
      initialProps: { authorityScope: "alice" }
    })
    await waitFor(() =>
      expect(alice.result.current.isPermissionDenied).toBe(true)
    )
    await act(async () => {
      resolveBob(graph())
    })
    await waitFor(() => expect(bob.result.current.graph?.nodes).toHaveLength(1))
    expect(bob.result.current.isPermissionDenied).toBe(false)
    expect(alice.result.current.graph).toBeNull()
    expect(request).toHaveBeenCalledTimes(2)
  })

  it("keeps unrelated cursor failures observable and preserves the existing graph", async () => {
    request.mockResolvedValueOnce(graph(true)).mockRejectedValue(failure(503))
    const { wrapper, useGraph } = hookWorkspace()
    const view = renderHook(useGraph, { wrapper, initialProps: {} })
    await waitFor(() => expect(view.result.current.canExpand).toBe(true))
    await act(async () => {
      await expect(view.result.current.expand()).rejects.toMatchObject({
        status: 503
      })
    })
    expect(view.result.current.graph?.nodes).toHaveLength(1)
    expect(view.result.current.isPermissionDenied).toBe(false)
  })

  it("keeps an ordinary500 retryable with the existing load-error message", async () => {
    request.mockRejectedValueOnce(failure(500)).mockResolvedValue(graph())
    await setup()
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Could not load the notes graph."
    )
    fireEvent.click(screen.getByRole("button", { name: "Refresh graph" }))
    expect(
      await screen.findByTestId("notes-graph-primary-view")
    ).toBeInTheDocument()
  })

  it("retains the existing graph on transient refresh failure and offline transition", async () => {
    request.mockResolvedValueOnce(graph()).mockRejectedValue(failure(503))
    const view = await setup()
    await screen.findByTestId("notes-graph-primary-view")
    fireEvent.click(screen.getByRole("button", { name: "Refresh graph" }))
    expect(
      await screen.findByTestId("notes-graph-degraded-state")
    ).toHaveTextContent("Refresh failed. Showing the last available graph.")
    expect(screen.getByTestId("notes-graph-primary-view")).toBeInTheDocument()
    view.rerender(<NotesGraphWorkspace {...props} isOnline={false} />)
    expect(screen.getByTestId("notes-graph-offline-state")).toHaveTextContent(
      "Offline: showing the last available graph."
    )
    expect(screen.getByTestId("notes-graph-primary-view")).toBeInTheDocument()
  })

  it("preserves the genuine loading state before a response", async () => {
    request.mockReturnValue(new Promise(() => {}))
    await setup()
    expect(screen.getByRole("status")).toHaveTextContent(commonEn.loading.title)
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })
})
