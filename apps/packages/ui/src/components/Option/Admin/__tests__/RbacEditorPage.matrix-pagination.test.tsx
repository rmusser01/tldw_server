import React from "react"
import i18next from "i18next"
import { initReactI18next } from "react-i18next"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import RbacEditorPage from "../RbacEditorPage"

void i18next.use(initReactI18next).init({ lng: "en", resources: {} })

/**
 * Stage C-S4 (F16): the permission matrix table must paginate its rows
 * (pageSize 50) instead of mounting the whole permissions x roles checkbox
 * grid, and the role-column array must keep a stable identity across
 * re-renders while the matrix state is unchanged.
 */
const apiMock = vi.hoisted(() => ({
  getRolePermissionMatrix: vi.fn(),
  listPermissionCategories: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock
}))

const MATRIX_PAGE_SIZE = 50

const buildMatrix = (
  permissionCount: number,
  roles: Array<{ id: number; name: string }>
) => ({
  roles,
  permissions: Array.from({ length: permissionCount }, (_, index) => ({
    id: index + 1,
    name: `permission-${index + 1}`,
    category: index % 2 === 0 ? "media" : "chat"
  })),
  grid: {}
})

/** Data rows currently mounted in the matrix table (excludes rc-table's
 *  internal measure/placeholder rows). */
const matrixRows = () =>
  Array.from(document.querySelectorAll(".ant-table-tbody tr.ant-table-row"))

describe("RbacEditorPage matrix pagination (C-S4)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMock.listPermissionCategories.mockResolvedValue([])
  })

  it("test_matrix_paginates_permissions", async () => {
    apiMock.getRolePermissionMatrix.mockResolvedValue(
      buildMatrix(120, [
        { id: 1, name: "Admins" },
        { id: 2, name: "Editors" }
      ])
    )

    render(<RbacEditorPage />)

    // Page 1 mounts only the first 50 of the 120 permission rows.
    expect(await screen.findByText("permission-1")).toBeTruthy()
    await waitFor(() => {
      expect(matrixRows()).toHaveLength(MATRIX_PAGE_SIZE)
    })
    expect(screen.getByText("permission-50")).toBeTruthy()
    expect(screen.queryByText("permission-51")).toBeNull()

    // Page 2 mounts the next 50 rows and drops page 1's rows.
    fireEvent.click(screen.getByTitle("2"))

    expect(await screen.findByText("permission-51")).toBeTruthy()
    await waitFor(() => {
      expect(matrixRows()).toHaveLength(MATRIX_PAGE_SIZE)
    })
    expect(screen.getByText("permission-100")).toBeTruthy()
    expect(screen.queryByText("permission-101")).toBeNull()
    expect(screen.queryByText("permission-1")).toBeNull()
  })

  it("test_matrix_columns_stable", async () => {
    const roles = [
      { id: 1, name: "Admins" },
      { id: 2, name: "Editors" },
      { id: 3, name: "Viewers" }
    ]
    const roleMapSpy = vi.spyOn(roles, "map")

    apiMock.getRolePermissionMatrix.mockResolvedValue(buildMatrix(60, roles))

    const { rerender } = render(<RbacEditorPage />)

    expect(await screen.findByText("permission-1")).toBeTruthy()
    // The role-column array was derived exactly once for the initial render.
    expect(roleMapSpy).toHaveBeenCalledTimes(1)

    try {
      // Re-renders with the same matrix state (same roles/grid instances)
      // must reuse the memoized role-column array instead of rebuilding it.
      rerender(<RbacEditorPage />)
      rerender(<RbacEditorPage />)

      expect(screen.getAllByText("Admins").length).toBeGreaterThan(0)
      expect(roleMapSpy).toHaveBeenCalledTimes(1)
    } finally {
      roleMapSpy.mockRestore()
    }
  })
})
