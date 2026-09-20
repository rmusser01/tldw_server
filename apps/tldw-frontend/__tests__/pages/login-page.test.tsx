// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const routerMock = vi.hoisted(() => ({ push: vi.fn() }))
vi.mock("next/router", () => ({ useRouter: () => routerMock }))
vi.mock("next/dynamic", () => ({
  default: () => () => <div data-testid="hosted-settings" />
}))

const deploymentMock = vi.hoisted(() => ({ hosted: false }))
vi.mock("@/services/tldw/deployment-mode", () => ({
  isHostedTldwDeployment: () => deploymentMock.hosted
}))

const clientMock = vi.hoisted(() => ({ getConfig: vi.fn() }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: clientMock }))

const authMock = vi.hoisted(() => ({ login: vi.fn() }))
vi.mock("@/services/tldw/TldwAuth", () => ({ tldwAuth: authMock }))

vi.mock("@/services/auth-errors", () => ({
  mapMultiUserLoginErrorMessage: () => "Login failed. Check your credentials."
}))

vi.mock("@web/components/navigation/RouteRedirect", () => ({
  RouteRedirect: ({ title }: { title: string }) => (
    <div data-testid="route-redirect">{title}</div>
  )
}))

import LoginPage from "../../pages/login"

describe("LoginPage (#2919)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    deploymentMock.hosted = false
    clientMock.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8001",
      authMode: "multi-user"
    })
    authMock.login.mockResolvedValue(undefined)
  })

  it("renders a focused sign-in screen for configured multi-user servers", async () => {
    render(<LoginPage />)

    expect(
      await screen.findByRole("heading", { name: "Sign in to tldw" })
    ).toBeInTheDocument()
    expect(screen.getByText("http://127.0.0.1:8001", { exact: false })).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Change server" })).toHaveAttribute(
      "href",
      "/settings/tldw"
    )
  })

  it("signs in and navigates home", async () => {
    render(<LoginPage />)
    await screen.findByRole("heading", { name: "Sign in to tldw" })

    fireEvent.change(screen.getByLabelText("Username"), {
      target: { value: "audit-admin" }
    })
    fireEvent.change(screen.getByLabelText("Password"), {
      target: { value: "secret" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }))

    await waitFor(() => {
      expect(authMock.login).toHaveBeenCalledWith({
        username: "audit-admin",
        password: "secret"
      })
    })
    await waitFor(() => {
      expect(routerMock.push).toHaveBeenCalledWith("/")
    })
  })

  it("shows a friendly error on failed login and stays put", async () => {
    authMock.login.mockRejectedValue(new Error("401"))
    render(<LoginPage />)
    await screen.findByRole("heading", { name: "Sign in to tldw" })

    fireEvent.change(screen.getByLabelText("Username"), {
      target: { value: "audit-admin" }
    })
    fireEvent.change(screen.getByLabelText("Password"), {
      target: { value: "wrong" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }))

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Login failed. Check your credentials."
    )
    expect(routerMock.push).not.toHaveBeenCalled()
  })

  it("redirects to settings when no multi-user server is configured", async () => {
    clientMock.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user"
    })

    render(<LoginPage />)

    expect(await screen.findByTestId("route-redirect")).toHaveTextContent(
      "Connect a server first"
    )
  })
})
