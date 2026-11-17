import React from "react"
import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { ConfigProvider } from "antd"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { ACPPlayground } from "../index"
import { useACPSessionsStore } from "@/store/acp-sessions"

const boundaries = vi.hoisted(() => ({
  getConfig: vi.fn(),
  storageWrites: [] as Array<[string, unknown]>
}))
vi.mock("react-i18next", () => ({ useTranslation: () => ({
  t: (key: string, fallback?: string | { defaultValue?: string }) =>
    typeof fallback === "string" ? fallback : fallback?.defaultValue ?? key
}) }))
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, fallback: unknown) => {
    const [value, setValue] = React.useState(fallback)
    return [value, (next: unknown) => {
      boundaries.storageWrites.push([key, next])
      setValue(next)
    }, { isLoading: false }]
  }
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: { getConfig: (...args: unknown[]) => boundaries.getConfig(...args) } }))
// Keep the actual media hooks, Header, Drawer/Tabs, SessionPanel, Create form,
// ToolsPanel, query/store and REST client. Only external/unrelated seams differ.
vi.mock("@/hooks/useACPSession", () => ({ useACPSession: () => ({
  state: "disconnected", isConnected: false, error: null, connect: vi.fn(), sendPrompt: vi.fn(),
  cancel: vi.fn(), approvePermission: vi.fn(), denyPermission: vi.fn()
}) }))
vi.mock("../ACPChatPanel", () => ({ ACPChatPanel: () => <div>Empty chat</div> }))
vi.mock("../ACPWorkspacePanel", () => ({ ACPWorkspacePanel: () => null }))
vi.mock("../ACPPermissionModal", () => ({ ACPPermissionModal: () => null }))

const clients: QueryClient[] = []
let fetchMock: ReturnType<typeof vi.fn>
let healthStatus: number
let resizeViewport: (width: number) => void
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { "Content-Type": "application/json" }
})
const mount = (width: number) => {
  // Width is supplied at the browser matchMedia boundary; useMobile/useDesktop
  // execute their production queries rather than returning mocked booleans.
  let currentWidth = width
  const matches = (media: string) =>
    [...media.matchAll(/\((min|max)-width:\s*(\d+)px\)/g)].every((match) =>
      match[1] === "min" ? currentWidth >= Number(match[2]) : currentWidth <= Number(match[2])
    ) && /width:/.test(media)
  const queries: Array<{ media: string; listeners: Set<(event: MediaQueryListEvent) => void> }> = []
  vi.spyOn(window, "matchMedia").mockImplementation((media) => {
    const listeners = new Set<(event: MediaQueryListEvent) => void>()
    const addEventListener = vi.fn()
    const removeEventListener = vi.fn()
    addEventListener.mockImplementation((type: string, listener: (event: MediaQueryListEvent) => void) => {
      if (type === "change") listeners.add(listener)
    })
    removeEventListener.mockImplementation((type: string, listener: (event: MediaQueryListEvent) => void) => {
      if (type === "change") listeners.delete(listener)
    })
    queries.push({ media, listeners })
    return {
      get matches() { return matches(media) },
      media, onchange: null, addListener: vi.fn(), removeListener: vi.fn(),
      addEventListener, removeEventListener, dispatchEvent: vi.fn()
    }
  })
  resizeViewport = (nextWidth) => {
    const previous = queries.map((query) => matches(query.media))
    act(() => {
      currentWidth = nextWidth
      queries.forEach((query, index) => {
        const nextMatch = matches(query.media)
        if (previous[index] !== nextMatch) {
          const event = Object.assign(new Event("change"), { matches: nextMatch, media: query.media })
          query.listeners.forEach((listener) => listener(event))
        }
      })
    })
  }
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  clients.push(client)
  render(<ConfigProvider theme={{ token: { motion: false } }}>
    <QueryClientProvider client={client}><ACPPlayground /></QueryClientProvider>
  </ConfigProvider>)
}
const ready = async (width: number) => {
  mount(width)
  await screen.findByTestId("acp-health-indicator")
  await waitFor(() => expect(screen.queryAllByTestId("acp-panel-loading")).toHaveLength(0))
  return screen.getByRole("banner")
}
const noPost = () => expect(fetchMock.mock.calls.filter(([, init]) => init?.method === "POST")).toHaveLength(0)
const closeDrawer = async (drawer: HTMLElement) => {
  await userEvent.click(within(drawer).getByRole("button", { name: "Close" }))
  await waitFor(() => expect(drawer).not.toBeVisible())
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
  window.history.pushState({}, "", "/acp-playground?agent=opencode")
  useACPSessionsStore.getState().reset()
  boundaries.storageWrites.length = 0
  boundaries.getConfig.mockResolvedValue({ serverUrl: window.location.origin, authMode: "single-user", authSource: "cookie-session" })
  healthStatus = 200
  fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const path = new URL(String(input), window.location.origin).pathname
    if (path.endsWith("/acp/health")) return healthStatus === 200
      ? json({ overall: "ok", runner: { status: "ok" }, agents: [] })
      : json({ detail: "Runner unavailable" }, healthStatus)
    if (path.endsWith("/acp/sessions")) return json({ sessions: [], total: 0 })
    if (path.endsWith("/acp/agents")) return json({ default_agent: "opencode", agents: [{
      type: "opencode", name: "OpenCode", description: "Native ACP agent", is_configured: true,
      entrypoint: { entrypoint_strategy: "native_acp", probe_state: "ready_to_probe", blockers: [] }
    }] })
    throw new Error(`Unexpected ACP fixture request: ${path}`)
  })
  vi.stubGlobal("fetch", fetchMock)
})
afterEach(() => {
  cleanup()
  clients.splice(0).forEach((client) => client.clear())
  useACPSessionsStore.getState().reset()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

describe("ACP tablet controls", () => {
  it("opens the actual session drawer and Create form at the observed 858px width", async () => {
    const banner = await ready(858)
    // Reach the existing real control on unchanged source to isolate its handler
    // from the additional accessible-label repair and JSDom's absent Tailwind CSS.
    const control = within(banner).getAllByRole("button")[0]
    await userEvent.click(control)
    const drawer = await screen.findByRole("dialog", { name: "Sessions" })
    await waitFor(() => expect(drawer).toBeVisible())
    expect(control).toHaveAccessibleName("Hide sessions")
    expect(control).toHaveAttribute("aria-pressed", "true")
    await userEvent.click(within(drawer).getByRole("button", { name: "Create Session" }))
    const dialog = await waitFor(() => {
      const candidate = screen.getAllByRole("dialog").find((element) =>
        within(element).queryByLabelText("Working Directory")
      )
      expect(candidate).toBeDefined()
      return candidate!
    })
    await waitFor(() => expect(dialog).toBeVisible())
    expect(within(dialog).getByText("Create ACP Session")).toBeVisible()
    expect(within(dialog).getByLabelText("Working Directory")).toBeVisible()
    await userEvent.click(within(dialog).getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(dialog).not.toBeInTheDocument())
    await closeDrawer(drawer)
    expect(control).toHaveAccessibleName("Show sessions")
    expect(control).toHaveAttribute("aria-pressed", "false")
    expect(boundaries.storageWrites).toEqual([])
    noPost()
  }, 30000)

  it.each([768, 1023])("exposes the accessible session drawer at tablet boundary %spx", async (width) => {
    const banner = await ready(width)
    await userEvent.click(within(banner).getByRole("button", { name: "Show sessions" }))
    const drawer = await screen.findByRole("dialog", { name: "Sessions" })
    expect(within(drawer).getByRole("button", { name: "Create Session" })).toBeInTheDocument()
    expect(boundaries.storageWrites).toEqual([])
    noPost()
  }, 30000)

  it("opens and closes real Tools in the tablet drawer without changing desktop preferences", async () => {
    const banner = await ready(858)
    const control = within(banner).getByRole("button", { name: "Show tools" })
    await userEvent.click(control)
    const drawer = await screen.findByRole("dialog", { name: "Tools" })
    expect(await within(drawer).findByText("Select a session to view capabilities")).toBeInTheDocument()
    expect(control).toHaveAccessibleName("Hide tools")
    await closeDrawer(drawer)
    expect(control).toHaveAccessibleName("Show tools")
    expect(boundaries.storageWrites).toEqual([])
    noPost()
  }, 30000)

  it("keeps 1024px desktop toggles bound to persisted side panes", async () => {
    const banner = await ready(1024)
    await userEvent.click(within(banner).getByRole("button", { name: "Hide sessions" }))
    expect(within(banner).getByRole("button", { name: "Show sessions" })).toHaveAttribute("aria-pressed", "false")
    expect(screen.queryByRole("dialog", { name: "Sessions" })).not.toBeInTheDocument()
    await userEvent.click(within(banner).getByRole("button", { name: "Show sessions" }))
    await userEvent.click(within(banner).getByRole("button", { name: "Hide tools" }))
    expect(within(banner).getByRole("button", { name: "Show tools" })).toHaveAttribute("aria-pressed", "false")
    expect(screen.queryByRole("dialog", { name: "Tools" })).not.toBeInTheDocument()
    expect(boundaries.storageWrites).toEqual([
      ["acp-playground-left-pane", false], ["acp-playground-left-pane", true],
      ["acp-playground-right-pane", false]
    ])
    noPost()
  }, 30000)

  it("keeps the tablet Tools in-panel Hide independent of saved desktop preference", async () => {
    const banner = await ready(858)
    await userEvent.click(within(banner).getByRole("button", { name: "Show tools" }))
    const drawer = await screen.findByRole("dialog", { name: "Tools" })
    const heading = await within(drawer).findByRole("heading", { name: "Tools & Capabilities" })
    await userEvent.click(within(heading.parentElement!).getByRole("button"))
    await waitFor(() => expect(drawer).not.toBeVisible())
    expect(boundaries.storageWrites).toEqual([])
    expect(within(banner).getByRole("button", { name: "Show tools" })).toHaveAttribute("aria-pressed", "false")
    noPost()
  }, 30000)

  it("keeps 767px mobile tabs and effective Create without header toggles or drawers", async () => {
    const banner = await ready(767)
    expect(within(banner).queryAllByRole("button")).toHaveLength(0)
    expect(screen.getByRole("tab", { name: "Chat" })).toHaveAttribute("aria-selected", "true")
    await userEvent.click(screen.getByRole("tab", { name: "Sessions" }))
    await userEvent.click(await screen.findByRole("button", { name: "Create Session" }))
    const dialog = await screen.findByRole("dialog", { name: "Create ACP Session" })
    expect(within(dialog).getByLabelText("Working Directory")).toBeInTheDocument()
    await userEvent.click(within(dialog).getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    await userEvent.click(screen.getByRole("tab", { name: "Tools" }))
    expect(await screen.findByText("Select a session to view capabilities")).toBeInTheDocument()
    expect(boundaries.storageWrites).toEqual([])
    noPost()
  }, 30000)

  it("keeps unavailable tablet recovery free of session controls", async () => {
    healthStatus = 503
    mount(858)
    await screen.findByTestId("acp-playground-capability-recovery")
    expect(within(screen.getByRole("banner")).queryAllByRole("button")).toHaveLength(0)
    expect(screen.queryByRole("button", { name: "Create Session" })).not.toBeInTheDocument()
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    noPost()
  }, 30000)

  it.each(["Sessions", "Tools"])("retires the open %s tablet drawer when resized to desktop and back", async (panel) => {
    const banner = await ready(858)
    const label = panel.toLowerCase()
    await userEvent.click(within(banner).getByRole("button", { name: `Show ${label}` }))
    const drawer = await screen.findByRole("dialog", { name: panel })
    const drawerRoot = drawer.closest(".ant-drawer")!
    const mask = drawerRoot.querySelector(".ant-drawer-mask")!
    expect(drawer).toBeVisible()
    expect(mask).toBeVisible()

    resizeViewport(1024)
    await waitFor(() => expect(drawerRoot).not.toHaveClass("ant-drawer-open"))
    expect(drawer).not.toBeVisible()
    expect(mask).not.toBeVisible()
    expect(within(banner).getByRole("button", { name: `Hide ${label}` })).toHaveAttribute("aria-pressed", "true")
    expect(boundaries.storageWrites).toEqual([])

    resizeViewport(858)
    expect(within(banner).getByRole("button", { name: `Show ${label}` })).toHaveAttribute("aria-pressed", "false")
    expect(drawer).not.toBeVisible()
    expect(mask).not.toBeVisible()
    expect(boundaries.storageWrites).toEqual([])
    noPost()
  }, 30000)
})
