// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const connectionState = {
  online: true,
  uxState: "connected_ok" as
    | "connected_ok"
    | "testing"
    | "configuring_url"
    | "configuring_auth"
    | "error_auth"
    | "error_unreachable"
    | "unconfigured",
  navigate: vi.fn(),
  search: ""
}

const mockBlocklist = {
  rawText: "",
  setRawText: vi.fn(),
  rawLint: null,
  isDirtyRaw: false,
  loading: false,
  loadRaw: vi.fn().mockResolvedValue(undefined),
  saveRaw: vi.fn().mockResolvedValue(undefined),
  saveRawText: vi.fn().mockResolvedValue(undefined),
  lintRaw: vi.fn().mockResolvedValue(undefined),
  managedItems: [],
  managedVersion: "",
  managedLine: "",
  setManagedLine: vi.fn(),
  managedLint: null,
  loadManaged: vi.fn().mockResolvedValue(undefined),
  appendManaged: vi.fn().mockResolvedValue(undefined),
  appendLine: vi.fn().mockResolvedValue(undefined),
  deleteManaged: vi.fn().mockResolvedValue(undefined),
  lintManagedLine: vi.fn().mockResolvedValue(undefined),
  lintLine: vi.fn().mockResolvedValue({ items: [], valid_count: 0, invalid_count: 0 })
}
const markMilestone = vi.fn()

vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({ data: null, isFetching: false, error: null, refetch: vi.fn() })
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_k: string, fb?: string) => fb || _k })
}))
vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useNavigate: () => connectionState.navigate,
    useSearchParams: () => [new URLSearchParams(connectionState.search)]
  }
})
vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => connectionState.online
}))
vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => ({
    uxState: connectionState.uxState,
    hasCompletedFirstRun: true
  })
}))
vi.mock("@/services/moderation", () => ({
  getModerationSettings: vi.fn(),
  getEffectivePolicy: vi.fn(),
  reloadModeration: vi.fn(),
  listUserOverrides: vi.fn(),
  testModeration: vi.fn(),
  getUserOverride: vi.fn()
}))
vi.mock("@/store/milestones", () => ({
  useMilestoneStore: (selector: (state: { markMilestone: typeof markMilestone }) => unknown) =>
    selector({ markMilestone })
}))
vi.mock("../hooks/useBlocklist", () => ({
  useBlocklist: () => mockBlocklist
}))

import { ModerationPlaygroundShell } from "../ModerationPlaygroundShell"

describe("ModerationPlaygroundShell", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.setItem("moderation-playground-onboarded", "true")
    connectionState.online = true
    connectionState.uxState = "connected_ok"
    connectionState.search = ""
    mockBlocklist.isDirtyRaw = false
  })

  it("marks content rules reviewed when the playground opens", () => {
    render(<ModerationPlaygroundShell />)

    expect(markMilestone).toHaveBeenCalledWith("content_rules_reviewed")
  })

  it("renders 5 tab buttons", () => {
    render(<ModerationPlaygroundShell />)
    expect(screen.getByRole("tab", { name: /policy/i })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /blocklist/i })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /overrides/i })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /test/i })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /advanced/i })).toBeInTheDocument()
  })

  it("uses Content Rules naming for the rules configuration surface", () => {
    render(<ModerationPlaygroundShell />)

    expect(
      screen.getByRole("heading", { name: "Content Rules" })
    ).toBeInTheDocument()
    expect(screen.queryByText("Moderation Playground")).not.toBeInTheDocument()
    expect(
      screen.getByText(/policies, blocklists, user overrides, and rule tests/i)
    ).toBeInTheDocument()
  })

  it("shows Policy tab content by default", async () => {
    render(<ModerationPlaygroundShell />)
    expect(await screen.findByText(/personal data protection/i)).toBeInTheDocument()
  })

  it("switches to Blocklist tab on click", async () => {
    render(<ModerationPlaygroundShell />)
    fireEvent.click(screen.getByRole("tab", { name: /blocklist/i }))
    expect(await screen.findByText(/syntax reference/i)).toBeInTheDocument()
  })

  it("marks content rules tested when the test sandbox opens", async () => {
    render(<ModerationPlaygroundShell />)

    fireEvent.click(screen.getByRole("tab", { name: /test/i }))

    await waitFor(() => {
      expect(markMilestone).toHaveBeenCalledWith("content_rules_tested")
    })
  })

  it("opens the Test Sandbox from the tab query parameter", async () => {
    connectionState.search = "tab=test"

    render(<ModerationPlaygroundShell />)

    await waitFor(() => {
      expect(screen.getByRole("tab", { name: /test/i })).toHaveAttribute("aria-selected", "true")
    })
    expect(markMilestone).toHaveBeenCalledWith("content_rules_tested")
  })

  it("renders context bar with scope selector", () => {
    render(<ModerationPlaygroundShell />)
    // The scope selector is a <select> with Server and User options
    const option = screen.getByRole("option", { name: /server/i }) as HTMLOptionElement
    expect(option).toBeInTheDocument()
    expect(option.selected).toBe(true)
  })

  it("shows the unsaved indicator when the raw blocklist editor is dirty", () => {
    mockBlocklist.isDirtyRaw = true

    render(<ModerationPlaygroundShell />)

    expect(screen.getByText(/unsaved/i)).toBeInTheDocument()
  })
})
