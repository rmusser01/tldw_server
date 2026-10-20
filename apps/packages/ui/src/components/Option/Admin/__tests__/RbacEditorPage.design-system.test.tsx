// @vitest-environment jsdom

import React from "react"
import { ConfigProvider, Form } from "antd"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import RbacEditorPage from "../RbacEditorPage"

const apiMock = vi.hoisted(() => ({
  canonical: vi.fn(),
  listAdminUsers: vi.fn(), listAdminRoles: vi.fn(), listPermissions: vi.fn(),
  listUserRoles: vi.fn(), listUserOverrides: vi.fn(), getUserEffectivePermissions: vi.fn(),
  createAdminRole: vi.fn(), assignUserRole: vi.fn(),
  revokeRolePermission: vi.fn(), addUserOverride: vi.fn(),
  getRolePermissionMatrix: vi.fn(),
  listPermissionCategories: vi.fn()
}))

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({ useCanonicalConnectionConfig: apiMock.canonical }))
const testConfig = (user = 1, serverUrl = "https://admin-a.example.test") => ({
  serverUrl, authMode: "multi-user", accessToken: `fixture.${btoa(JSON.stringify({ sub: String(user) }))}.signature`
})

type SelectProps = { id?: string; placeholder?: string; disabled?: boolean; value?: number | string | null; options?: Array<{ value: number | string; label: React.ReactNode; disabled?: boolean }>; onSearch?: (value: string) => void; onChange: (value: number | string | undefined) => void }

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  const Select = (props: SelectProps) => props.placeholder === "Search for a user..." ? (
    <div><input aria-label="User search" onChange={e => props.onSearch?.(e.target.value)} />
    <select aria-label="Selected user" value={props.value ?? ""} onChange={e => props.onChange(Number(e.target.value))}>
      <option value="">Select</option>{props.options?.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select></div>
  ) : <select id={props.id} aria-label={props.placeholder} disabled={props.disabled} value={props.value ?? ""}
    onChange={e => props.onChange?.(props.options?.find((o) => String(o.value) === e.target.value)?.value)}>
    <option value="">Select</option>{props.options?.map((o) => <option key={o.value} value={o.value} disabled={o.disabled}>{o.label}</option>)}
  </select>
  return { ...actual, Select }
})

const selectUser = async (id: string) => {
  fireEvent.click(await screen.findByRole("tab", { name: /User Permissions$/ }))
  fireEvent.change(await screen.findByLabelText("User search"), { target: { value: "a" } })
  await screen.findByRole("option", { name: "alice" })
  fireEvent.change(screen.getByLabelText("Selected user"), { target: { value: id } })
}

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock
}))

describe("RbacEditorPage design-system states", () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.clearAllMocks()
    apiMock.canonical.mockReturnValue({ config: testConfig(), authorityLoading: false })
    apiMock.getRolePermissionMatrix.mockResolvedValue({
      roles: [],
      permissions: [],
      grants: [], total_roles: 0
    })
    apiMock.listPermissionCategories.mockResolvedValue([])
    apiMock.listAdminUsers.mockResolvedValue({ users: [{ id: 11, username: "alice" }, { id: 22, username: "bob" }] })
    apiMock.listAdminRoles.mockResolvedValue([])
    apiMock.listPermissions.mockResolvedValue([])
    apiMock.listUserRoles.mockResolvedValue([])
    apiMock.listUserOverrides.mockResolvedValue([])
    apiMock.getUserEffectivePermissions.mockResolvedValue([])
    apiMock.createAdminRole.mockResolvedValue({})
    apiMock.assignUserRole.mockResolvedValue({})
    apiMock.revokeRolePermission.mockResolvedValue({})
    apiMock.addUserOverride.mockResolvedValue({})
  })

  it("renders admin guard feedback through the design-system Alert primitive", async () => {
    apiMock.getRolePermissionMatrix.mockRejectedValueOnce({ status: 403 })

    render(<RbacEditorPage />)

    const title = await screen.findByText("Access Restricted")
    const alert = title.closest('[data-ds-component="Alert"]')

    expect(alert).not.toBeNull()
    const alertEl = alert as HTMLElement
    expect(alertEl).toHaveAttribute("role", "alert")
    expect(alertEl).toHaveTextContent("Access Restricted")
    expect(alertEl).toHaveTextContent("forbidden")
  })
  it.each(["server", "account", "unresolved"])("retires privileged data when %s authority changes", async (change) => {
    apiMock.getRolePermissionMatrix.mockResolvedValueOnce({ roles: [], permissions: [{ id: 91, name: "Old privileged permission" }], grants: [], total_roles: 0 })
    const view = render(<RbacEditorPage />)
    await screen.findByText("Old privileged permission")
    apiMock.canonical.mockReturnValue({
      config: change === "server" ? testConfig(1, "https://admin-b.example.test") : change === "account" ? testConfig(2) : testConfig(),
      authorityLoading: change === "unresolved"
    })
    view.rerender(<RbacEditorPage />)
    expect(screen.queryByText("Old privileged permission")).not.toBeInTheDocument()
  })

  it("ignores a retired privileged response after authority switches", async () => {
    let finish!: (value: unknown) => void
    apiMock.getRolePermissionMatrix.mockReturnValueOnce(new Promise(resolve => { finish = resolve })).mockResolvedValue({ roles: [], permissions: [], grants: [], total_roles: 0 })
    const view = render(<RbacEditorPage />)
    apiMock.canonical.mockReturnValue({ config: testConfig(2), authorityLoading: false })
    view.rerender(<RbacEditorPage />)
    await act(async () => finish({ roles: [], permissions: [{ id: 91, name: "Old privileged permission" }], grants: [], total_roles: 0 }))
    expect(screen.queryByText("Old privileged permission")).not.toBeInTheDocument()
  })

  it("shows a failed role pane as unavailable while preserving successful permission panes and permits Retry", async () => {
    apiMock.listUserRoles.mockRejectedValueOnce(new Error("Role request failed")).mockResolvedValue([{ id: 31, name: "Recovered role" }])
    apiMock.getUserEffectivePermissions.mockResolvedValue([{ permission_id: 91, permission_name: "notes.read" }])
    render(<RbacEditorPage />)
    await selectUser("11")
    expect(await screen.findByText("notes.read")).toBeVisible()
    expect(await screen.findByText("User permissions are incomplete")).toBeVisible()
    expect(screen.getByRole("button", { name: /Add Role$/ })).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(await screen.findByText("Recovered role")).toBeVisible()
    expect(screen.queryByText("User permissions are incomplete")).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Add Role$/ })).toBeEnabled()
    expect(apiMock.listUserRoles).toHaveBeenCalledTimes(2)
  })

  it("does not put the previous user's late role into the selected user's action table", async () => {
    let finish!: (value: unknown) => void
    apiMock.listUserRoles.mockImplementation((id) => id === 11 ? new Promise(resolve => { finish = resolve }) : Promise.resolve([{ id: 41, name: "Bob current role" }]))
    render(<RbacEditorPage />)
    await selectUser("11")
    await waitFor(() => expect(finish).toBeTypeOf("function"))
    fireEvent.change(screen.getByLabelText("Selected user"), { target: { value: "22" } })
    await screen.findByText("Bob current role")
    await act(async () => finish([{ id: 31, name: "Alice old role" }]))
    expect(screen.queryByText("Alice old role")).not.toBeInTheDocument()
    expect(screen.getByText("Bob current role")).toBeVisible()
  })

  it("does not create a role after authority changes during form validation", async () => {
    const originalUseForm = Form.useForm
    let form!: ReturnType<typeof Form.useForm>[0]
    vi.spyOn(Form, "useForm").mockImplementation((...args) => { const result = originalUseForm(...args); form = result[0]; return result })
    const view = render(<RbacEditorPage />)
    fireEvent.click(await screen.findByRole("tab", { name: /Roles$/ }))
    await screen.findByPlaceholderText("Role name")
    let finish!: (value: unknown) => void
    vi.spyOn(form, "validateFields").mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    fireEvent.click(screen.getByRole("button", { name: /Create$/ }))
    apiMock.canonical.mockReturnValue({ config: testConfig(2), authorityLoading: false })
    view.rerender(<RbacEditorPage />)
    await act(async () => finish({ name: "old role" }))
    expect(apiMock.createAdminRole).not.toHaveBeenCalled()
  })

  it("renders real ID-bearing matrix grants and revokes the exact permission ID", async () => {
    apiMock.getRolePermissionMatrix.mockResolvedValue({
      roles: [{ id: 41, name: "Scoped reviewer" }],
      permissions: [{ id: 91, name: "notes.graph.read", category: "notes" }],
      grants: [{ role_id: 41, permission_id: 91 }], total_roles: 1
    })
    render(<RbacEditorPage />)
    const checkbox = await screen.findByRole("checkbox")
    expect(checkbox).toBeChecked()
    fireEvent.click(checkbox)
    await waitFor(() => expect(apiMock.revokeRolePermission).toHaveBeenCalledWith(41, 91, expect.objectContaining({ configSnapshot: testConfig(), abortSignal: expect.any(AbortSignal) })))
  })

  it("displays the real named user-role, override and effective-permission envelopes", async () => {
    apiMock.listUserRoles.mockResolvedValue({ user_id: 11, roles: [{ id: 31, name: "Actual role" }] })
    apiMock.listUserOverrides.mockResolvedValue({ user_id: 11, overrides: [{ permission_id: 91, permission_name: "notes.graph.write", granted: false }] })
    apiMock.getUserEffectivePermissions.mockResolvedValue({ user_id: 11, permissions: ["media.search"] })
    render(<RbacEditorPage />)
    await selectUser("11")
    expect(await screen.findByText("Actual role")).toBeVisible()
    expect(screen.getByText("notes.graph.write")).toBeVisible()
    expect(screen.getByText("DENY")).toBeVisible()
    expect(screen.getByText("media.search")).toBeVisible()
  })

  it("writes the supported allow effect when the operator selects Grant", async () => {
    apiMock.listPermissions.mockResolvedValue([{ id: 91, name: "notes.graph.read" }])
    render(<RbacEditorPage />)
    await selectUser("11")
    const add = screen.getByRole("button", { name: /Add Override$/ })
    await waitFor(() => expect(add).toBeEnabled())
    fireEvent.click(add)
    fireEvent.change(screen.getByLabelText("Select permission"), { target: { value: "91" } })
    const grant = screen.getByRole("option", { name: "Grant", exact: true })
    fireEvent.change(screen.getByLabelText("Select effect"), { target: { value: grant.getAttribute("value") } })
    fireEvent.click(screen.getByRole("button", { name: "OK", exact: true }))
    await waitFor(() => expect(apiMock.addUserOverride).toHaveBeenCalledWith(11, { permission_id: 91, effect: "allow" }, expect.objectContaining({ configSnapshot: testConfig(), abortSignal: expect.any(AbortSignal) })))
  })

  it("treats a malformed matrix as unavailable instead of editable false grants", async () => {
    apiMock.getRolePermissionMatrix.mockResolvedValue({ roles: [{ id: 41, name: "Scoped reviewer" }], permissions: [{ id: 91, name: "notes.graph.read" }] })
    render(<RbacEditorPage />)
    expect(await screen.findByText("Unable to load the permission matrix")).toBeVisible()
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument()
  })

  it("retains saved roles after a missing assignment target and permits a fresh selection", async () => {
    const saved = { id: 31, name: "Saved role" }, replacement = { id: 42, name: "Replacement role" }
    apiMock.listAdminRoles.mockResolvedValue([saved, { id: 41, name: "Deleted role" }, replacement])
    apiMock.listUserRoles.mockResolvedValue({ user_id: 11, roles: [saved] })
    apiMock.getUserEffectivePermissions.mockResolvedValue({ user_id: 11, permissions: ["media.search"] })
    apiMock.assignUserRole.mockRejectedValueOnce(Object.assign(new Error("Role not found"), { status: 404 }))
    render(<ConfigProvider theme={{ token: { motion: false } }}><RbacEditorPage /></ConfigProvider>)
    await selectUser("11")
    await screen.findByRole("cell", { name: "Saved role", exact: true })
    const add = screen.getByRole("button", { name: /Add Role$/ })
    await waitFor(() => expect(add).toBeEnabled())
    fireEvent.click(add)
    let modal = screen.getByRole("dialog", { name: "Assign Role" })
    await waitFor(() => expect(modal).toBeVisible())
    fireEvent.change(within(modal).getByLabelText("Select a role"), { target: { value: "41" } })
    fireEvent.click(within(modal).getByRole("button", { name: "OK", exact: true }))
    expect(await screen.findByText("Role not found")).toBeVisible()
    expect(screen.queryByText("Access Restricted")).not.toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole("dialog", { name: "Assign Role" })).toBeVisible())
    modal = screen.getByRole("dialog", { name: "Assign Role" })
    expect(screen.getByRole("cell", { name: "Saved role", exact: true })).toBeVisible()
    expect(screen.getByText("media.search")).toBeVisible()
    apiMock.listUserRoles.mockResolvedValue({ user_id: 11, roles: [saved, replacement] })
    await waitFor(() => expect(within(modal).getByRole("button", { name: "OK", exact: true })).toBeEnabled())
    fireEvent.change(within(modal).getByLabelText("Select a role"), { target: { value: "42" } })
    fireEvent.click(within(modal).getByRole("button", { name: "OK", exact: true }))
    expect(await screen.findByRole("cell", { name: "Replacement role", exact: true })).toBeVisible()
    expect(screen.queryByRole("dialog", { name: "Assign Role" })).not.toBeInTheDocument()
    expect(apiMock.assignUserRole.mock.calls.map(call => call.slice(0, 2))).toEqual([[11, 41], [11, 42]])
  })

  it("keeps genuine forbidden assignment failures behind the Admin guard", async () => {
    apiMock.listAdminRoles.mockResolvedValue([{ id: 41, name: "Scoped role" }])
    apiMock.assignUserRole.mockRejectedValueOnce(Object.assign(new Error("Forbidden"), { status: 403 }))
    render(<ConfigProvider theme={{ token: { motion: false } }}><RbacEditorPage /></ConfigProvider>)
    await selectUser("11")
    const add = screen.getByRole("button", { name: /Add Role$/ })
    await waitFor(() => expect(add).toBeEnabled())
    fireEvent.click(add)
    const modal = screen.getByRole("dialog", { name: "Assign Role" })
    fireEvent.change(within(modal).getByLabelText("Select a role"), { target: { value: "41" } })
    fireEvent.click(within(modal).getByRole("button", { name: "OK", exact: true }))
    expect(await screen.findByText("Access Restricted")).toBeVisible()
    expect(screen.getByText("forbidden")).toBeVisible()
    expect(screen.queryByRole("dialog", { name: "Assign Role" })).not.toBeInTheDocument()
    expect(apiMock.assignUserRole).toHaveBeenCalledTimes(1)
  })

  it("prevents dismissing an override modal while validation is pending", async () => {
    const originalUseForm = Form.useForm
    let form!: ReturnType<typeof Form.useForm>[0]
    vi.spyOn(Form, "useForm").mockImplementation((...args) => { const result = originalUseForm(...args); form = result[0]; return result })
    render(<RbacEditorPage />)
    await selectUser("11")
    const add = screen.getByRole("button", { name: /Add Override$/ })
    await waitFor(() => expect(add).toBeEnabled())
    fireEvent.click(add)
    let finish!: (value: unknown) => void
    vi.spyOn(form, "validateFields").mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    fireEvent.click(screen.getByRole("button", { name: "OK", exact: true }))
    expect(screen.getByRole("button", { name: "Cancel", exact: true })).toBeDisabled()
    await act(async () => finish({ permission_id: 91, effect: "allow" }))
    await waitFor(() => expect(apiMock.addUserOverride).toHaveBeenCalledTimes(1))
  })

})
