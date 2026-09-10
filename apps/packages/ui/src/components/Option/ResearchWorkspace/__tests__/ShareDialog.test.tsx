import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, beforeEach, vi } from "vitest"
import { ShareDialog } from "../ShareDialog"

const {
  mockCreateToken,
  mockShareWorkspace,
  mockUpdateShare,
  mockRevokeShare,
  mockRevokeToken,
  mockMessage,
  mockModalConfirm,
  mockStaticMessage,
  mockStaticModalConfirm,
  mockClipboardWriteText,
  sharingState
} = vi.hoisted(() => ({
  mockCreateToken: vi.fn(),
  mockShareWorkspace: vi.fn(),
  mockUpdateShare: vi.fn(),
  mockRevokeShare: vi.fn(),
  mockRevokeToken: vi.fn(),
  mockMessage: {
    success: vi.fn(),
    error: vi.fn()
  },
  mockModalConfirm: vi.fn(),
  mockStaticMessage: {
    success: vi.fn(),
    error: vi.fn()
  },
  mockStaticModalConfirm: vi.fn(),
  mockClipboardWriteText: vi.fn(),
  sharingState: {
    shares: [] as Array<Record<string, unknown>>,
    tokens: [] as Array<Record<string, unknown>>
  }
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  const Modal = Object.assign(actual.Modal, {
    confirm: mockStaticModalConfirm
  })
  const App = Object.assign(actual.App, {
    useApp: () => ({
      message: mockMessage,
      modal: { confirm: mockModalConfirm },
      notification: {}
    })
  })
  return {
    ...actual,
    App,
    Modal,
    message: {
      ...actual.message,
      success: mockStaticMessage.success,
      error: mockStaticMessage.error
    }
  }
})

vi.mock("@/hooks/useSharing", () => ({
  useWorkspaceShares: () => ({
    data: {
      shares: sharingState.shares,
      total: sharingState.shares.length
    },
    isLoading: false
  }),
  useShareWorkspace: () => ({
    mutateAsync: mockShareWorkspace,
    isPending: false
  }),
  useUpdateShare: () => ({
    mutateAsync: mockUpdateShare,
    isPending: false
  }),
  useRevokeShare: () => ({
    mutateAsync: mockRevokeShare,
    isPending: false
  }),
  useShareTokens: () => ({
    data: {
      tokens: sharingState.tokens,
      total: sharingState.tokens.length
    }
  }),
  useCreateToken: () => ({
    mutateAsync: mockCreateToken,
    isPending: false
  }),
  useRevokeToken: () => ({
    mutateAsync: mockRevokeToken,
    isPending: false
  })
}))

describe("ShareDialog", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    sharingState.shares = []
    sharingState.tokens = []
    mockClipboardWriteText.mockResolvedValue(undefined)
    Object.defineProperty(window, "location", {
      configurable: true,
      value: {
        origin: "http://localhost:3000"
      }
    })
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: mockClipboardWriteText
      }
    })
  })

  it("shows a generated share link with an accessible copy action and feedback", async () => {
    mockCreateToken.mockResolvedValue({
      id: 12,
      raw_token: "workspace-secret-token",
      token_prefix: "workspace",
      resource_type: "workspace",
      resource_id: "workspace-alpha",
      access_level: "view_chat",
      allow_clone: true,
      is_password_protected: false,
      max_uses: null,
      use_count: 0,
      expires_at: null,
      is_revoked: false
    })

    render(
      <ShareDialog
        workspaceId="workspace-alpha"
        open={true}
        onClose={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("tab", { name: "Share Link" }))
    fireEvent.click(screen.getByRole("button", { name: "Generate Link" }))

    const generatedLink = await screen.findByLabelText("Generated share link")
    expect(generatedLink).toHaveValue(
      "http://localhost:3000/share/workspace-secret-token"
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Copy generated share link" })
    )

    await waitFor(() => {
      expect(mockClipboardWriteText).toHaveBeenCalledWith(
        "http://localhost:3000/share/workspace-secret-token"
      )
      expect(mockMessage.success).toHaveBeenCalledWith(
        "Share link copied to clipboard"
      )
    })
  })

  it("shows an actionable error when clipboard access is unavailable", async () => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: undefined
    })
    mockCreateToken.mockResolvedValue({
      id: 12,
      raw_token: "workspace-secret-token",
      token_prefix: "workspace",
      resource_type: "workspace",
      resource_id: "workspace-alpha",
      access_level: "view_chat",
      allow_clone: true,
      is_password_protected: false,
      max_uses: null,
      use_count: 0,
      expires_at: null,
      is_revoked: false
    })

    render(
      <ShareDialog
        workspaceId="workspace-alpha"
        open={true}
        onClose={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("tab", { name: "Share Link" }))
    fireEvent.click(screen.getByRole("button", { name: "Generate Link" }))

    await screen.findByLabelText("Generated share link")
    fireEvent.click(
      screen.getByRole("button", { name: "Copy generated share link" })
    )

    expect(mockClipboardWriteText).not.toHaveBeenCalled()
    expect(mockMessage.error).toHaveBeenCalledWith(
      "Clipboard access is not supported in this browser context"
    )
  })

  it("validates an empty team or organization target inline before submitting", async () => {
    render(
      <ShareDialog
        workspaceId="workspace-alpha"
        open={true}
        onClose={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Share" }))

    expect(
      await screen.findByText("Enter a team or organization ID before sharing.")
    ).toBeInTheDocument()
    expect(mockShareWorkspace).not.toHaveBeenCalled()
  })

  it("shows active share security details and confirms revocation with feedback", async () => {
    sharingState.shares = [
      {
        id: 42,
        workspace_id: "workspace-alpha",
        owner_user_id: 1,
        share_scope_type: "team",
        share_scope_id: 7,
        access_level: "full_edit",
        allow_clone: false,
        created_by: 1,
        is_revoked: false
      }
    ]
    sharingState.tokens = [
      {
        id: 99,
        token_prefix: "tok_abc",
        raw_token: "do-not-render-secret",
        resource_type: "workspace",
        resource_id: "workspace-alpha",
        access_level: "view_chat_add",
        allow_clone: true,
        is_password_protected: true,
        max_uses: 5,
        use_count: 2,
        expires_at: "2026-07-01T00:00:00.000Z",
        is_revoked: false
      }
    ]
    mockRevokeShare.mockResolvedValue(undefined)
    mockRevokeToken.mockResolvedValue(undefined)

    render(
      <ShareDialog
        workspaceId="workspace-alpha"
        open={true}
        onClose={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("tab", { name: "Active Shares" }))

    expect(
      screen.getByText(
        "Revoking access prevents future workspace reads and questions. It does not erase content or answers recipients saved while they had access. Recipients may use their own configured model provider, which can receive selected shared passages when they ask a question."
      )
    ).toBeInTheDocument()
    expect(screen.getByText("Team #7")).toBeInTheDocument()
    expect(screen.getAllByText("Full access").length).toBeGreaterThan(0)
    expect(screen.getByText("Clone disabled")).toBeInTheDocument()
    expect(screen.getByText("tok_abc...")).toBeInTheDocument()
    expect(screen.getAllByText("Can add sources").length).toBeGreaterThan(0)
    expect(screen.getByText("2 / 5 uses")).toBeInTheDocument()
    expect(screen.getByText("Password required")).toBeInTheDocument()
    expect(screen.getByText("Expires Jul 1, 2026")).toBeInTheDocument()
    expect(screen.queryByText("do-not-render-secret")).not.toBeInTheDocument()

    fireEvent.click(
      screen.getByRole("button", { name: "Revoke team share Team #7" })
    )
    expect(mockModalConfirm).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Revoke Team #7?"
      })
    )

    const revokeShareConfig = mockModalConfirm.mock.calls.at(-1)?.[0] as {
      onOk?: (close?: () => void) => Promise<void>
    }
    const closeShareConfirm = vi.fn()
    await revokeShareConfig.onOk?.(closeShareConfirm)

    expect(mockRevokeShare).toHaveBeenCalledWith(42)
    expect(mockMessage.success).toHaveBeenCalledWith("Share revoked")
    expect(closeShareConfirm).toHaveBeenCalledTimes(1)

    fireEvent.click(
      screen.getByRole("button", { name: "Revoke share link tok_abc" })
    )
    const revokeTokenConfig = mockModalConfirm.mock.calls.at(-1)?.[0] as {
      onOk?: (close?: () => void) => Promise<void>
    }
    const closeTokenConfirm = vi.fn()
    await revokeTokenConfig.onOk?.(closeTokenConfirm)

    expect(mockRevokeToken).toHaveBeenCalledWith(99)
    expect(mockMessage.success).toHaveBeenCalledWith("Share link revoked")
    expect(closeTokenConfirm).toHaveBeenCalledTimes(1)
    expect(mockStaticModalConfirm).not.toHaveBeenCalled()
    expect(mockStaticMessage.success).not.toHaveBeenCalled()
    expect(mockStaticMessage.error).not.toHaveBeenCalled()
  })

  it("updates active team or org share access and clone permission", async () => {
    sharingState.shares = [
      {
        id: 42,
        workspace_id: "workspace-alpha",
        owner_user_id: 1,
        share_scope_type: "team",
        share_scope_id: 7,
        access_level: "view_chat",
        allow_clone: false,
        created_by: 1,
        is_revoked: false
      }
    ]
    mockUpdateShare.mockResolvedValue({
      id: 42,
      workspace_id: "workspace-alpha",
      owner_user_id: 1,
      share_scope_type: "team",
      share_scope_id: 7,
      access_level: "view_chat_add",
      allow_clone: true,
      created_by: 1,
      is_revoked: false
    })

    render(
      <ShareDialog
        workspaceId="workspace-alpha"
        open={true}
        onClose={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("tab", { name: "Active Shares" }))

    fireEvent.change(screen.getByLabelText("Access level for Team #7"), {
      target: { value: "view_chat_add" }
    })
    fireEvent.click(screen.getByRole("checkbox", { name: "Allow cloning for Team #7" }))

    await waitFor(() => {
      expect(mockUpdateShare).toHaveBeenCalledWith({
        shareId: 42,
        access_level: "view_chat_add"
      })
      expect(mockUpdateShare).toHaveBeenCalledWith({
        shareId: 42,
        allow_clone: true
      })
      expect(mockMessage.success).toHaveBeenCalledWith("Share updated")
    })
  })

  it("keeps revoke confirmations open and shows errors when revocation fails", async () => {
    sharingState.shares = [
      {
        id: 42,
        workspace_id: "workspace-alpha",
        owner_user_id: 1,
        share_scope_type: "team",
        share_scope_id: 7,
        access_level: "full_edit",
        allow_clone: false,
        created_by: 1,
        is_revoked: false
      }
    ]
    sharingState.tokens = [
      {
        id: 99,
        token_prefix: "tok_abc",
        raw_token: "do-not-render-secret",
        resource_type: "workspace",
        resource_id: "workspace-alpha",
        access_level: "view_chat_add",
        allow_clone: true,
        is_password_protected: true,
        max_uses: 5,
        use_count: 2,
        expires_at: "2026-07-01T00:00:00.000Z",
        is_revoked: false
      }
    ]
    mockRevokeShare.mockRejectedValue(new Error("Team revoke failed"))
    mockRevokeToken.mockRejectedValue(new Error("Token revoke failed"))

    render(
      <ShareDialog
        workspaceId="workspace-alpha"
        open={true}
        onClose={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("tab", { name: "Active Shares" }))

    fireEvent.click(
      screen.getByRole("button", { name: "Revoke team share Team #7" })
    )
    const revokeShareConfig = mockModalConfirm.mock.calls.at(-1)?.[0] as {
      onOk?: (close?: () => void) => Promise<void>
    }
    const closeShareConfirm = vi.fn()
    await expect(revokeShareConfig.onOk?.(closeShareConfirm)).rejects.toThrow(
      "Team revoke failed"
    )

    expect(mockMessage.error).toHaveBeenCalledWith("Team revoke failed")
    expect(closeShareConfirm).not.toHaveBeenCalled()

    fireEvent.click(
      screen.getByRole("button", { name: "Revoke share link tok_abc" })
    )
    const revokeTokenConfig = mockModalConfirm.mock.calls.at(-1)?.[0] as {
      onOk?: (close?: () => void) => Promise<void>
    }
    const closeTokenConfirm = vi.fn()
    await expect(revokeTokenConfig.onOk?.(closeTokenConfirm)).rejects.toThrow(
      "Token revoke failed"
    )

    expect(mockMessage.error).toHaveBeenCalledWith("Token revoke failed")
    expect(closeTokenConfirm).not.toHaveBeenCalled()
  })
})
