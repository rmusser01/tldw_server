import React from "react"
import { QueryClientProvider, type QueryClient } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import RbacEditorPage from "../RbacEditorPage"
import { createAdminQueryClient } from "../AdminQueryProvider"

/**
 * Stage C-S3 (F16): the User Permissions tab must resolve override names and
 * role-option disabled state through O(1) keyed lookups (Map/Set built in
 * useMemo) instead of per-row `Array.prototype.find` / `.some` scans.
 *
 * Roles and permissions arrive through the shared admin react-query hooks
 * (B-S4), so each test renders the page under a fresh admin query client and
 * keeps a reference to the exact permission array instance it serves: the
 * spy on that instance proves whether row rendering linear-scans the catalog.
 */
const adminQueryClients: QueryClient[] = []
const renderPage = (ui: React.ReactElement) => {
  const client = createAdminQueryClient()
  adminQueryClients.push(client)
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>
  )
}

afterEach(() => {
  cleanup()
  adminQueryClients.splice(0).forEach((client) => client.clear())
})

const apiMock = vi.hoisted(() => ({
  getRolePermissionMatrix: vi.fn(),
  listPermissionCategories: vi.fn(),
  listAdminRoles: vi.fn(),
  listPermissions: vi.fn(),
  listAdminUsers: vi.fn(),
  listUserRoles: vi.fn(),
  listUserOverrides: vi.fn(),
  getUserEffectivePermissions: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock
}))

const PERMISSION_COUNT = 300
const OVERRIDE_COUNT = 100

const permissions = Array.from({ length: PERMISSION_COUNT }, (_, index) => ({
  id: index + 1,
  name: `permission-${index + 1}`,
  category: index % 2 === 0 ? "media" : "chat"
}))

const roles = [
  { id: 1, name: "Administrators", is_system: true },
  { id: 2, name: "Editors" },
  { id: 3, name: "Viewers" }
]

const users = [{ id: 10, username: "alice" }]

/** Overrides reference permission ids without carrying permission_name, so
 *  every override row must resolve its display name from the catalog. */
const overrides = Array.from({ length: OVERRIDE_COUNT }, (_, index) => ({
  permission_id: ((index * 7) % PERMISSION_COUNT) + 1,
  effect: index % 2 === 0 ? "grant" : "deny"
}))

const assignedRoles = [{ id: 1, name: "Administrators" }]

/** Switch to the lazily-mounted User Permissions tab and select a user. */
const openUserPermissions = async (userRoles: unknown[]) => {
  apiMock.listAdminUsers.mockResolvedValue(users)
  apiMock.listUserRoles.mockResolvedValue(userRoles)
  apiMock.listUserOverrides.mockResolvedValue(overrides)
  apiMock.getUserEffectivePermissions.mockResolvedValue([])

  fireEvent.click(screen.getByText("User Permissions"))

  const searchInput = await screen.findByRole("combobox")
  fireEvent.mouseDown(searchInput.closest(".ant-select") as HTMLElement)
  fireEvent.change(searchInput, { target: { value: "a" } })
  fireEvent.click(await screen.findByText("alice"))

  await waitFor(() => {
    expect(apiMock.listUserOverrides).toHaveBeenCalledWith(10)
  })
}

describe("RbacEditorPage lookup performance (C-S3)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMock.getRolePermissionMatrix.mockResolvedValue({
      roles: [],
      permissions: [],
      grid: {}
    })
    apiMock.listPermissionCategories.mockResolvedValue([])
    apiMock.listAdminRoles.mockResolvedValue(roles)
    apiMock.listPermissions.mockResolvedValue(permissions)
  })

  it("test_override_lookup_is_o1", async () => {
    const permissionFindSpy = vi.spyOn(permissions, "find")

    try {
      renderPage(<RbacEditorPage />)
      await openUserPermissions(assignedRoles)

      // Every override row resolved its catalog name (behavior parity with
      // the linear scan it replaces): id (0*7 % 300) + 1 = 1, (1*7 % 300) + 1 = 8.
      expect(await screen.findByText("permission-1")).toBeTruthy()
      expect(await screen.findByText("permission-8")).toBeTruthy()
      // ...and a late id from the 100-override set: (99*7 % 300) + 1 = 93.
      expect(await screen.findByText("permission-93")).toBeTruthy()

      // The permission catalog was never linear-scanned during rendering.
      expect(permissionFindSpy).not.toHaveBeenCalled()
    } finally {
      permissionFindSpy.mockRestore()
    }
  })

  it(
    "test_role_options_disabled_via_set",
    async () => {
      renderPage(<RbacEditorPage />)
      await openUserPermissions(assignedRoles)

      // The assigned role renders in the Assigned Roles table.
      expect(await screen.findByText("Administrators")).toBeTruthy()

      fireEvent.click(screen.getByRole("button", { name: /add role/i }))

      const modal = await waitFor(() => {
        const modalEl = document.querySelector(".ant-modal")
        expect(modalEl).toBeTruthy()
        return modalEl as HTMLElement
      })
      const searchInput = within(modal).getByRole("combobox")
      fireEvent.mouseDown(searchInput.closest(".ant-select") as HTMLElement)

      const assignedOption = await screen.findByText("Administrators", {
        selector: ".ant-select-item-option-content"
      })
      expect(assignedOption.closest(".ant-select-item-option")).toHaveClass(
        "ant-select-item-option-disabled"
      )

      const unassignedOption = await screen.findByText("Editors", {
        selector: ".ant-select-item-option-content"
      })
      expect(unassignedOption.closest(".ant-select-item-option")).not.toHaveClass(
        "ant-select-item-option-disabled"
      )
    },
    // Opening the modal re-renders the full 300-permission/100-override tab
    // in jsdom; that DOM diff alone takes several seconds.
    20_000
  )
})
