import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { SharedWithMe } from "../SharedWithMe"

const sharingMocks = vi.hoisted(() => ({
  useSharedWithMe: vi.fn(),
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

describe("SharedWithMe Research Studio route", () => {
  beforeEach(() => {
    sharingMocks.navigate.mockReset()
    sharingMocks.useSharedWithMe.mockReturnValue({
      data: [
        {
          share_id: 7,
          workspace_id: "workspace-1",
          workspace_name: "Policy Deck",
          workspace_description: "Shared policy notes",
          owner_user_id: 42,
          access_level: "view_chat",
          allow_clone: false
        }
      ],
      isLoading: false,
      error: null
    })
    sharingMocks.useCloneWorkspace.mockReturnValue({
      isPending: false,
      variables: null,
      mutate: vi.fn()
    })
  })

  it("opens shared workspaces through client-side Research Studio navigation", () => {
    render(<SharedWithMe />)

    fireEvent.click(screen.getByRole("button", { name: "Open shared workspace" }))

    expect(sharingMocks.navigate).toHaveBeenCalledWith("/research-studio?shared=7")
  })
})
