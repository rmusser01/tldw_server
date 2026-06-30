// @vitest-environment jsdom
import { fireEvent, render, screen } from "@testing-library/react"
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
  navigate: vi.fn()
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
      useSearchParams: () => [new URLSearchParams()]
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
vi.mock("../hooks/useBlocklist", () => ({
  useBlocklist: () => mockBlocklist
}))

import { ModerationPlaygroundShell } from "../ModerationPlaygroundShell"

describe("ModerationPlaygroundShell connection warning", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.setItem("moderation-playground-onboarded", "true")
    connectionState.online = true
    connectionState.uxState = "connected_ok"
  })

  it("shows credential guidance when auth is missing", () => {
    connectionState.online = false
    connectionState.uxState = "error_auth"

    render(<ModerationPlaygroundShell />)

    expect(
      screen.getByText("Add your credentials to use moderation controls.")
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Open Settings" }))
    expect(connectionState.navigate).toHaveBeenCalledWith("/settings/tldw")
  })

  it("shows setup guidance when setup is incomplete", () => {
    connectionState.online = false
    connectionState.uxState = "unconfigured"

    render(<ModerationPlaygroundShell />)

    expect(
      screen.getByText("Finish setup to use moderation controls.")
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Finish Setup" }))
    expect(connectionState.navigate).toHaveBeenCalledWith("/")
  })

  it("suppresses the warning while connection checks are still testing", () => {
    connectionState.online = false
    connectionState.uxState = "testing"

    render(<ModerationPlaygroundShell />)

    expect(
      screen.queryByText("Add your credentials to use moderation controls.")
    ).not.toBeInTheDocument()
    expect(
      screen.queryByText("Connect to your tldw server to use moderation controls.")
    ).not.toBeInTheDocument()
  })
})
