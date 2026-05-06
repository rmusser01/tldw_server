import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PublicShare } from "../PublicShare"

const hookMocks = vi.hoisted(() => ({
  usePublicPreview: vi.fn(),
  useVerifySharePassword: vi.fn(),
  useImportFromToken: vi.fn()
}))

const routerMocks = vi.hoisted(() => ({
  navigate: vi.fn()
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useNavigate: () => routerMocks.navigate
  }
})

vi.mock("@/hooks/useSharing", () => ({
  usePublicPreview: (...args: unknown[]) => hookMocks.usePublicPreview(...args),
  useVerifySharePassword: (...args: unknown[]) =>
    hookMocks.useVerifySharePassword(...args),
  useImportFromToken: (...args: unknown[]) => hookMocks.useImportFromToken(...args)
}))

describe("PublicShare", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    hookMocks.useVerifySharePassword.mockReturnValue({
      isPending: false,
      mutateAsync: vi.fn()
    })
    hookMocks.useImportFromToken.mockReturnValue({
      isPending: false,
      mutateAsync: vi.fn()
    })
  })

  it("routes prototype workspace tokens into the prototype collaboration flow", () => {
    hookMocks.usePublicPreview.mockReturnValue({
      data: {
        resource_type: "prototype_workspace",
        resource_name: "Sales dashboard",
        resource_description: "Stakeholder prototype",
        is_password_protected: false,
        access_level: "full_edit",
        allow_clone: false
      },
      isLoading: false,
      error: null
    })

    render(<PublicShare token="prototype-token" />)

    fireEvent.click(
      screen.getByRole("button", { name: "Open Prototype Collaboration" })
    )

    expect(routerMocks.navigate).toHaveBeenCalledWith(
      "/prototype-workspaces?share_token=prototype-token"
    )
  })

  it("passes a verified prototype share password through navigation state", async () => {
    const verifyPassword = vi.fn().mockResolvedValue({ verified: true })
    hookMocks.usePublicPreview.mockReturnValue({
      data: {
        resource_type: "prototype_workspace",
        resource_name: "Sales dashboard",
        resource_description: "Stakeholder prototype",
        is_password_protected: true,
        access_level: "full_edit",
        allow_clone: false
      },
      isLoading: false,
      error: null
    })
    hookMocks.useVerifySharePassword.mockReturnValue({
      isPending: false,
      mutateAsync: verifyPassword
    })

    render(<PublicShare token="prototype-token" />)

    fireEvent.change(screen.getByPlaceholderText("Enter password"), {
      target: { value: "demo-pass" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Verify Password" }))

    await waitFor(() => {
      expect(verifyPassword).toHaveBeenCalledWith({
        token: "prototype-token",
        password: "demo-pass"
      })
    })

    fireEvent.click(
      screen.getByRole("button", { name: "Open Prototype Collaboration" })
    )

    expect(routerMocks.navigate).toHaveBeenCalledWith(
      "/prototype-workspaces?share_token=prototype-token",
      {
        state: {
          prototypeSharePassword: "demo-pass"
        }
      }
    )
  })
})
