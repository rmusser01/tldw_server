import { MemoryRouter, Route, Routes } from "react-router-dom"
import { AdminRouteShell } from "@/components/Option/Admin/AdminRouteShell"
// @vitest-environment jsdom
import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { useQueryClient, type QueryClient } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { PrivateWorkspaceScope } from "@/hooks/usePrivateWorkspaceScope"

const mocks = vi.hoisted(() => ({ scope: null as PrivateWorkspaceScope | null, stats: vi.fn() }))
vi.mock("@/hooks/usePrivateWorkspaceScope", () => ({ usePrivateWorkspaceScope: () => ({ scope: mocks.scope, retry: vi.fn() }) }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: { getSystemStats: mocks.stats } }))

import { AdminQueryProvider } from "@/components/Option/Admin/AdminQueryProvider"
import { useSystemStats } from "@/services/tldw/adminQueries"

let capturedClient: QueryClient
const Probe = () => {
  const client = useQueryClient()
  React.useLayoutEffect(() => { capturedClient = client }, [client])
  const stats = useSystemStats({ timeoutMs: 10_000 })
  return <div>{stats.data?.privateLabel ?? (stats.error ? "forbidden" : "loading")}</div>
}
const owner = (id: number) => {
  const controller = new AbortController()
  return {
    controller,
    scope: {
      key: `cookie-owner-${id}`,
      config: { serverUrl: "http://same-server", authMode: "multi-user", authSource: "cookie-session" },
      requestScope: { config: { serverUrl: "http://same-server" }, userId: id },
      signal: controller.signal,
      isCurrent: () => !controller.signal.aborted
    } as PrivateWorkspaceScope
  }
}
const tree = (key = "page") => <AdminQueryProvider><Probe key={key} /></AdminQueryProvider>

describe("Admin private query lifetime", () => {
  beforeEach(() => { mocks.stats.mockReset(); mocks.scope = null })

  it("retires warm account A data before identical-cookie account B can mount", async () => {
    const a = owner(1)
    mocks.scope = a.scope
    mocks.stats.mockResolvedValueOnce({ privateLabel: "A-private" }).mockRejectedValueOnce(new Error("403"))
    const view = render(tree())
    await screen.findByText("A-private")
    const oldClient = capturedClient
    act(() => a.controller.abort())
    expect(oldClient.getQueryCache().getAll()).toHaveLength(0)
    mocks.scope = owner(2).scope
    view.rerender(tree("B"))
    expect(screen.queryByText("A-private")).toBeNull()
    await screen.findByText("forbidden")
    expect(mocks.stats).toHaveBeenCalledTimes(2)
    expect(mocks.stats.mock.calls[1][0]).toMatchObject({
      timeoutMs: 10_000,
      configSnapshot: mocks.scope.config,
      abortSignal: mocks.scope.signal,
      headers: { "X-TLDW-Expected-User-ID": "2" }
    })
  })

  it("ignores an old response after retirement and aborts its transport", async () => {
    const a = owner(1)
    mocks.scope = a.scope
    let resolveA!: (value: unknown) => void
    mocks.stats.mockImplementationOnce(() => new Promise(resolve => { resolveA = resolve }))
    const view = render(tree())
    await waitFor(() => expect(mocks.stats).toHaveBeenCalledTimes(1))
    const oldClient = capturedClient
    act(() => a.controller.abort())
    mocks.scope = owner(2).scope
    mocks.stats.mockResolvedValueOnce({ privateLabel: "B-private" })
    view.rerender(tree("B"))
    await screen.findByText("B-private")
    await act(async () => resolveA({ privateLabel: "A-late" }))
    expect(screen.queryByText("A-late")).toBeNull()
    expect(oldClient.getQueryCache().getAll()).toHaveLength(0)
    expect(mocks.stats.mock.calls[0][0].abortSignal.aborted).toBe(true)
  })

  it("keeps the owner cache above separate Server Admin and Monitoring route shells", async () => {
    mocks.scope = owner(1).scope
    mocks.stats.mockResolvedValue({ privateLabel: "A-private" })
    render(<MemoryRouter initialEntries={["/admin/server"]}>
      <AdminQueryProvider><Routes>
        <Route path="/admin/server" element={<AdminRouteShell path="/admin/server"><Probe /></AdminRouteShell>} />
        <Route path="/admin/monitoring" element={<AdminRouteShell path="/admin/monitoring"><Probe /></AdminRouteShell>} />
      </Routes></AdminQueryProvider>
    </MemoryRouter>)
    await screen.findByText("A-private")
    const client = capturedClient
    fireEvent.click(screen.getByRole("link", { name: "Monitoring" }))
    await screen.findByText("A-private")
    expect(capturedClient).toBe(client)
    expect(mocks.stats).toHaveBeenCalledTimes(1)
    expect(document.title).toContain("Monitoring")
  })

  it("dedupes navigation within one verified owner lifetime", async () => {
    mocks.scope = owner(1).scope
    mocks.stats.mockResolvedValue({ privateLabel: "A-private" })
    const view = render(tree())
    await screen.findByText("A-private")
    const client = capturedClient
    view.rerender(tree("next-page"))
    await screen.findByText("A-private")
    expect(capturedClient).toBe(client)
    expect(mocks.stats).toHaveBeenCalledTimes(1)
  })
})
