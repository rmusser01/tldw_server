// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"

const mocks = vi.hoisted(() => ({
  listAcpProfiles: vi.fn(),
  createAcpProfile: vi.fn()
}))

vi.mock("@/services/tldw/mcp-hub", () => ({
  listAcpProfiles: (...args: unknown[]) => mocks.listAcpProfiles(...args),
  createAcpProfile: (...args: unknown[]) => mocks.createAcpProfile(...args)
}))

import { AcpProfilesTab } from "../AcpProfilesTab"

describe("AcpProfilesTab", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.listAcpProfiles.mockResolvedValue([])
    mocks.createAcpProfile.mockResolvedValue({
      id: 1,
      name: "default-dev",
      owner_scope_type: "global",
      profile: {},
      is_active: true
    })
  })

  it("renders ACP profile list and can open create form", async () => {
    const user = userEvent.setup()
    render(<AcpProfilesTab />)

    expect(await screen.findByRole("button", { name: /create profile/i })).toBeTruthy()
    await user.click(screen.getByRole("button", { name: /create profile/i }))
    expect(screen.getByLabelText(/profile name/i)).toBeTruthy()
  })

  it("renders load failures through the shared Alert primitive", async () => {
    mocks.listAcpProfiles.mockRejectedValueOnce(new Error("offline"))

    const { container } = render(<AcpProfilesTab />)

    expect(await screen.findByText("Failed to load ACP profiles.")).toBeTruthy()
    expect(container.querySelector('[data-ds-component="Alert"]')).toBeTruthy()
  })
})
