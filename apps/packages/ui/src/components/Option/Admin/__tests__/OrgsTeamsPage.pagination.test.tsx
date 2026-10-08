// @vitest-environment jsdom
//
// Stage B-S3: the org members/teams sub-tables must page on the server
// (limit/offset) instead of assuming the full list arrives in one response.

import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import OrgsTeamsPage from "../OrgsTeamsPage"

const apiMock = vi.hoisted(() => ({
  listOrgs: vi.fn(),
  listOrgMembers: vi.fn(),
  listTeams: vi.fn(),
  listTeamMembers: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock
}))

// No i18n instance exists in tests; hand back a STABLE t that honors default
// values and interpolates {{vars}} (the oversight-page test does the same).
const stableT = vi.hoisted(
  () =>
    (
      key: string,
      fallbackOrOptions?: string | Record<string, unknown>,
      maybeOptions?: Record<string, unknown>
    ) => {
      let text =
        typeof fallbackOrOptions === "string" ? fallbackOrOptions : key
      const options =
        typeof fallbackOrOptions === "string" ? maybeOptions : fallbackOrOptions
      if (options && typeof options === "object") {
        for (const [name, value] of Object.entries(options)) {
          text = text.replace(new RegExp(`{{\\s*${name}\\s*}}`, "g"), String(value))
        }
      }
      return text
    }
)

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: stableT })
}))

const TOTAL_MEMBERS = 250

describe("OrgsTeamsPage server-side pagination (admin perf B-S3)", () => {
  beforeEach(() => {
    vi.clearAllMocks()

    if (!window.matchMedia) {
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: false,
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn()
        }))
      })
    }

    if (!(window as any).ResizeObserver) {
      ;(window as any).ResizeObserver = class {
        observe() {}
        unobserve() {}
        disconnect() {}
      }
    }

    apiMock.listOrgs.mockResolvedValue({
      items: [{ id: 1, name: "Alpha Org", slug: "alpha", member_count: TOTAL_MEMBERS, created_at: "2026-01-01T00:00:00Z" }],
      total: 1,
      limit: 100,
      offset: 0
    })

    // Server-side slice of a 250-member roster; mirrors the {items, total}
    // envelope the backend repo layer produces.
    apiMock.listOrgMembers.mockImplementation(
      async (_orgId: number, params?: { limit?: number; offset?: number }) => {
        const limit = params?.limit ?? 100
        const offset = params?.offset ?? 0
        const all = Array.from({ length: TOTAL_MEMBERS }, (_, i) => ({
          user_id: i + 1,
          username: `user_${i + 1}`,
          role: "member",
          joined_at: "2026-01-01T00:00:00Z"
        }))
        return { items: all.slice(offset, offset + limit), total: TOTAL_MEMBERS }
      }
    )

    apiMock.listTeams.mockResolvedValue({ items: [], total: 0 })
    apiMock.listTeamMembers.mockResolvedValue([])
  })

  const expandFirstOrg = async (container: HTMLElement) => {
    const expandButton = await waitFor(() => {
      const button = container.querySelector(
        "button.ant-table-row-expand-icon"
      ) as HTMLElement | null
      expect(button).not.toBeNull()
      return button as HTMLElement
    })
    fireEvent.click(expandButton)
    await screen.findByText("Organization Members")
  }

  it("test_org_members_server_pagination: page 2 fetches offset=20 and the footer reports the server total", async () => {
    const { container } = render(<OrgsTeamsPage />)

    await screen.findByText("Alpha Org")
    await expandFirstOrg(container)

    // Initial page: first 20 rows, offset 0.
    await waitFor(() => {
      expect(apiMock.listOrgMembers).toHaveBeenCalledWith(1, {
        limit: 20,
        offset: 0
      })
    })

    // The table footer surfaces the server-reported total (250 members).
    expect(await screen.findByText("Total 250 items")).toBeInTheDocument()

    // Jump to page 2: the next fetch must carry offset=20, not refetch page 1.
    fireEvent.click(screen.getByTitle("2"))

    await waitFor(() => {
      expect(apiMock.listOrgMembers).toHaveBeenLastCalledWith(1, {
        limit: 20,
        offset: 20
      })
    })
    // Page 2 rows render (users 21..40), not a re-slice of page 1.
    expect(await screen.findByText("user_21")).toBeInTheDocument()
    expect(screen.queryByText("user_1")).not.toBeInTheDocument()
  })
})
