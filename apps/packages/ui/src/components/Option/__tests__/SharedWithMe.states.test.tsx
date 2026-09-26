import React from "react"
import "./shared-with-me-i18n"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import type { useSharedWithMe as useSharedWithMeHook } from "@/hooks/useSharing"
import type {
  SharedWithMeItem,
  SharedWithMeResponse
} from "@/types/sharing"
import { SharedWithMe } from "../SharedWithMe"
import { TldwApiError } from "@/services/tldw/api-error"

type SharedWithMeHookState = Pick<
  ReturnType<typeof useSharedWithMeHook>,
  "data" | "isLoading" | "error"
>

const canonicalShare = {
  share_id: 7,
  workspace_id: "workspace-1",
  workspace_name: "Policy Deck",
  workspace_description: "Shared policy notes",
  owner_user_id: 42,
  access_level: "view_chat",
  allow_clone: true
} satisfies SharedWithMeItem

const canonicalResponse = {
  items: [canonicalShare],
  total: 1
} satisfies SharedWithMeResponse

// Deliberately simulates runtime server drift outside the canonical typed contract.
const unknownAccessResponse = {
  items: [{ ...canonicalShare, access_level: "mystery_access" }],
  total: 1
} as unknown as SharedWithMeResponse

const sharingMocks = vi.hoisted(() => ({
  useSharedWithMe: vi.fn<() => SharedWithMeHookState>(),
  mutate: vi.fn(),
  navigate: vi.fn()
}))

vi.mock("@/hooks/useSharing", () => ({
  useSharedWithMe: () => sharingMocks.useSharedWithMe()
}))

vi.mock("@/hooks/useSharedWorkspaceClones", () => ({
  useSharedWorkspaceClones: () => ({ rows: [], scope: "scope", status: "ready", begin: sharingMocks.mutate, refresh: vi.fn() })
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => sharingMocks.navigate
}))

describe("SharedWithMe states", () => {
  beforeEach(() => {
    sharingMocks.mutate.mockReset()
    sharingMocks.navigate.mockReset()
    sharingMocks.useSharedWithMe.mockReturnValue({
      data: canonicalResponse,
      isLoading: false,
      error: null
    })
  })

  it("renders shares from the canonical response envelope", () => {
    render(<SharedWithMe />)

    expect(screen.getByText("Policy Deck")).toBeInTheDocument()
    expect(screen.getByText("Read-only")).toBeInTheDocument()
    expect(screen.getByText("Shared policy notes")).toBeInTheDocument()
  })

  it("explains missing permission and hides stale share actions", () => {
    sharingMocks.useSharedWithMe.mockReturnValue({
      data: canonicalResponse,
      isLoading: false,
      error: new TldwApiError("Permission required", 403, {
        code: "sharing_permission_required"
      })
    })

    render(<SharedWithMe />)

    expect(screen.getByRole("alert")).toHaveTextContent(
      "Your account does not have permission to access shared workspaces. Ask your server administrator to grant sharing.read, then try again."
    )
    expect(screen.queryByText("Policy Deck")).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Open Policy Deck" })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Clone Policy Deck" })).not.toBeInTheDocument()
  })

  it("falls back to the workspace id when the API returns a null name", () => {
    const unnamedResponse = {
      items: [{ ...canonicalShare, workspace_name: null }],
      total: 1
    } satisfies SharedWithMeResponse
    sharingMocks.useSharedWithMe.mockReturnValue({
      data: unnamedResponse,
      isLoading: false,
      error: null
    })

    render(<SharedWithMe />)

    expect(screen.getByText("workspace-1")).toBeInTheDocument()
  })

  it("gives repeated row actions workspace-specific accessible names", () => {
    const multiRowResponse = {
      items: [
        canonicalShare,
        {
          ...canonicalShare,
          share_id: 8,
          workspace_id: "workspace-2",
          workspace_name: "Discovery Notes"
        }
      ],
      total: 2
    } satisfies SharedWithMeResponse
    sharingMocks.useSharedWithMe.mockReturnValue({
      data: multiRowResponse,
      isLoading: false,
      error: null
    })

    render(<SharedWithMe />)

    expect(
      screen.getByRole("button", { name: "Open Policy Deck" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Open Discovery Notes" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Clone Policy Deck" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Clone Discovery Notes" })
    ).toBeInTheDocument()
  })

  it("renders populated shares without deprecated Ant Design components or props", () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    const warningSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined)

    render(<SharedWithMe />)

    const diagnostics = [...errorSpy.mock.calls, ...warningSpy.mock.calls]
      .flat()
      .join(" ")
    expect(diagnostics).not.toContain("[antd: List]")
    expect(diagnostics).not.toContain("[antd: Space] `direction` is deprecated")
  })

  it("renders the canonical empty response", () => {
    const emptyResponse = {
      items: [],
      total: 0
    } satisfies SharedWithMeResponse
    sharingMocks.useSharedWithMe.mockReturnValue({
      data: emptyResponse,
      isLoading: false,
      error: null
    })

    render(<SharedWithMe />)

    expect(
      screen.getByText("No shared workspaces available yet.")
    ).toBeInTheDocument()
  })

  it("renders the loading state", () => {
    sharingMocks.useSharedWithMe.mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null
    })

    const { container } = render(<SharedWithMe />)

    expect(container.querySelector(".ant-spin")).not.toBeNull()
    expect(screen.queryByText("Policy Deck")).not.toBeInTheDocument()
  })

  it("renders the query error", () => {
    sharingMocks.useSharedWithMe.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error("Could not load shared workspaces")
    })

    render(<SharedWithMe />)

    expect(
      screen.getByText("Could not load shared workspaces.")
    ).toBeInTheDocument()
  })

  it("renders fallback access labels and friendlier owner text", () => {
    sharingMocks.useSharedWithMe.mockReturnValue({
      data: unknownAccessResponse,
      isLoading: false,
      error: null
    })

    render(<SharedWithMe />)

    expect(screen.getByText("Policy Deck")).toBeInTheDocument()
    expect(screen.getByText("mystery_access")).toBeInTheDocument()
    expect(
      screen.getByText("Shared by workspace owner")
    ).toBeInTheDocument()
  })

  it("starts the durable clone lifecycle from the row action", () => {
    render(<SharedWithMe />)

    fireEvent.click(screen.getByRole("button", { name: "Clone Policy Deck" }))

    expect(sharingMocks.mutate).toHaveBeenCalledTimes(1)
    expect(sharingMocks.mutate).toHaveBeenCalledWith(7, "Policy Deck")
  })
})
