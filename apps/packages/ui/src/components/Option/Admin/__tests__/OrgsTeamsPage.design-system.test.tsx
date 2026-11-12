// @vitest-environment jsdom

import React from "react"
import { ConfigProvider, Form } from "antd"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import OrgsTeamsPage from "../OrgsTeamsPage"

const apiMock = vi.hoisted(() => ({
  canonical: vi.fn(),
  listOrgMembers: vi.fn(), listTeams: vi.fn(), createOrg: vi.fn(),
  listOrgs: vi.fn()
}))

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({ useCanonicalConnectionConfig: apiMock.canonical }))
const testConfig = (user = 1, serverUrl = "https://admin-a.example.test") => ({
  serverUrl, authMode: "multi-user", accessToken: `fixture.${btoa(JSON.stringify({ sub: String(user) }))}.signature`
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock
}))

const expectDesignSystemAlertForTitle = async (titleText: string) => {
  const title = await screen.findByText(titleText)
  const alert = title.closest('[data-ds-component="Alert"]')

  expect(alert).not.toBeNull()
  const alertEl = alert as HTMLElement
  expect(alertEl).toHaveAttribute("role", "alert")
  return alertEl
}

describe("OrgsTeamsPage design-system states", () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.clearAllMocks()
    apiMock.canonical.mockReturnValue({ config: testConfig(), authorityLoading: false })
    apiMock.listOrgs.mockResolvedValue([])
    apiMock.listOrgMembers.mockResolvedValue([])
    apiMock.listTeams.mockResolvedValue([])
    apiMock.createOrg.mockResolvedValue({})
  })

  it("renders forbidden guard feedback through the design-system Alert primitive", async () => {
    apiMock.listOrgs.mockRejectedValueOnce({ status: 403 })

    render(<OrgsTeamsPage />)

    const alert = await expectDesignSystemAlertForTitle("Access Denied")
    expect(alert).toHaveTextContent(
      "You don't have permission to manage organizations."
    )
  })

  it("renders missing-endpoint guard feedback through the design-system Alert primitive", async () => {
    apiMock.listOrgs.mockRejectedValueOnce({ status: 404 })

    render(<OrgsTeamsPage />)

    const alert = await expectDesignSystemAlertForTitle("Not Available")
    expect(alert).toHaveTextContent(
      "Organization management is not available on this server."
    )
  })

  it("renders orgs from the items-envelope response (#2916)", async () => {
    apiMock.listOrgs.mockResolvedValue({
      items: [
        {
          id: 1,
          name: "alice Workspace",
          slug: null,
          owner_user_id: 2,
          is_active: true,
          created_at: "2026-09-06T02:50:34"
        }
      ],
      total: 1,
      limit: 100,
      offset: 0
    })

    render(<OrgsTeamsPage />)

    expect(await screen.findByText("alice Workspace")).toBeInTheDocument()
  })
  it.each(["server", "account", "unresolved"])("retires privileged data when %s authority changes", async (change) => {
    apiMock.listOrgs.mockResolvedValueOnce([{ id: 91, name: "Old privileged organization" }])
    const view = render(<OrgsTeamsPage />)
    await screen.findByText("Old privileged organization")
    apiMock.canonical.mockReturnValue({
      config: change === "server" ? testConfig(1, "https://admin-b.example.test") : change === "account" ? testConfig(2) : testConfig(),
      authorityLoading: change === "unresolved"
    })
    view.rerender(<OrgsTeamsPage />)
    expect(screen.queryByText("Old privileged organization")).not.toBeInTheDocument()
  })

  it("ignores a retired privileged response after authority switches", async () => {
    let finish!: (value: unknown) => void
    apiMock.listOrgs.mockReturnValueOnce(new Promise(resolve => { finish = resolve })).mockResolvedValue([])
    const view = render(<OrgsTeamsPage />)
    apiMock.canonical.mockReturnValue({ config: testConfig(2), authorityLoading: false })
    view.rerender(<OrgsTeamsPage />)
    await act(async () => finish([{ id: 91, name: "Old privileged organization" }]))
    expect(screen.queryByText("Old privileged organization")).not.toBeInTheDocument()
  })

  it("shows a current member-list failure and permits a scoped Retry", async () => {
    apiMock.listOrgs.mockResolvedValue([{ id: 91, name: "Owned organization" }])
    apiMock.listOrgMembers.mockRejectedValueOnce(new Error("member request failed")).mockResolvedValue([{ user_id: 11, username: "recovered member", role: "member" }])
    render(<OrgsTeamsPage />)
    await screen.findByText("Owned organization")
    fireEvent.click(screen.getByRole("button", { name: "Expand row" }))
    expect(await screen.findByText("Organization members unavailable")).toBeVisible()
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(await screen.findByText("recovered member")).toBeVisible()
    expect(screen.queryByText("Organization members unavailable")).not.toBeInTheDocument()
    expect(apiMock.listOrgMembers).toHaveBeenCalledTimes(2)
  })

  it("does not create an organization after authority changes during form validation", async () => {
    const originalUseForm = Form.useForm
    let form!: ReturnType<typeof Form.useForm>[0]
    vi.spyOn(Form, "useForm").mockImplementation((...args) => { const result = originalUseForm(...args); form = result[0]; return result })
    const view = render(<OrgsTeamsPage />)
    fireEvent.click(await screen.findByRole("button", { name: /Create Org$/ }))
    let finish!: (value: unknown) => void
    vi.spyOn(form, "validateFields").mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    fireEvent.click(screen.getByRole("button", { name: "OK", exact: true }))
    apiMock.canonical.mockReturnValue({ config: testConfig(2), authorityLoading: false })
    view.rerender(<OrgsTeamsPage />)
    await act(async () => finish({ name: "old organization" }))
    expect(apiMock.createOrg).not.toHaveBeenCalled()
  })

  it("keeps the create modal open and prevents Cancel while its validated write is pending", async () => {
    const originalUseForm = Form.useForm
    let form!: ReturnType<typeof Form.useForm>[0]
    vi.spyOn(Form, "useForm").mockImplementation((...args) => { const result = originalUseForm(...args); form = result[0]; return result })
    render(<ConfigProvider theme={{ token: { motion: false } }}><OrgsTeamsPage /></ConfigProvider>)
    fireEvent.click(await screen.findByRole("button", { name: /Create Org$/ }))
    await waitFor(() => expect(screen.getByRole("dialog")).toBeVisible())
    let finish!: (value: unknown) => void
    vi.spyOn(form, "validateFields").mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    try {
      fireEvent.click(screen.getByRole("button", { name: "OK", exact: true }))
      expect(screen.getByRole("button", { name: "Cancel", exact: true })).toBeDisabled()
      fireEvent.click(screen.getByRole("button", { name: "Cancel", exact: true }))
      expect(screen.getByRole("dialog")).toBeVisible()
    } finally {
      await act(async () => finish({ name: "reviewed organization" }))
    }
    await waitFor(() => expect(apiMock.createOrg).toHaveBeenCalledTimes(1))
  })

})
