import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import type { useSharedWithMe as useSharedWithMeHook } from "@/hooks/useSharing"
import type {
  SharedWithMeItem,
  SharedWithMeResponse
} from "@/types/sharing"
import { SharedWithMe } from "../SharedWithMe"

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
  allow_clone: false
} satisfies SharedWithMeItem

const canonicalResponse = {
  items: [canonicalShare],
  total: 1
} satisfies SharedWithMeResponse

const sharingMocks = vi.hoisted(() => ({
  useSharedWithMe: vi.fn<() => SharedWithMeHookState>(),
  useCloneWorkspace: vi.fn(),
  navigate: vi.fn()
}))

vi.mock("@/hooks/useSharing", () => ({
  useSharedWithMe: () => sharingMocks.useSharedWithMe(),
  useCloneWorkspace: () => sharingMocks.useCloneWorkspace()
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => sharingMocks.navigate
}))

describe("SharedWithMe Research Workspace route", () => {
  beforeEach(() => {
    sharingMocks.navigate.mockReset()
    sharingMocks.useSharedWithMe.mockReturnValue({
      data: canonicalResponse,
      isLoading: false,
      error: null
    })
    sharingMocks.useCloneWorkspace.mockReturnValue({
      isPending: false,
      variables: null,
      mutate: vi.fn()
    })
  })

  it("opens shared workspaces through client-side Research Workspace navigation", () => {
    render(<SharedWithMe />)

    fireEvent.click(screen.getByRole("button", { name: "Open Policy Deck" }))

    expect(sharingMocks.navigate).toHaveBeenCalledWith("/research-workspace?shared=7")
  })
})
