import React from "react"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { DictionaryVersionHistoryModal } from "../DictionaryVersionHistoryModal"
import { tldwClient } from "@/services/tldw/TldwApiClient"

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    dictionaryVersions: vi.fn(),
    dictionaryVersionSnapshot: vi.fn(),
    revertDictionaryVersion: vi.fn()
  }
}))

const tldwClientMock = vi.mocked(tldwClient)

const dictionary = {
  id: 7,
  name: "Core Terms"
}

describe("DictionaryVersionHistoryModal design-system alerts", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("renders version-history load failures through the design-system Alert primitive", async () => {
    tldwClientMock.dictionaryVersions.mockRejectedValueOnce(
      new Error("Could not reach dictionary history")
    )

    render(
      <DictionaryVersionHistoryModal
        open
        dictionary={dictionary}
        onClose={vi.fn()}
      />
    )

    const title = await screen.findByText("Version history failed")
    const alert = title.closest('[data-ds-component="Alert"]')

    expect(alert).not.toBeNull()
    const alertEl = alert as HTMLElement

    expect(alertEl).toHaveTextContent("Could not reach dictionary history")
  })

  it("renders revision-restored messages through the design-system Alert primitive", async () => {
    const user = userEvent.setup()
    tldwClientMock.dictionaryVersions.mockResolvedValue({
      versions: [
        {
          revision: 3,
          created_at: "2026-05-29T12:00:00Z",
          change_type: "update",
          entry_count: 2
        }
      ]
    })
    tldwClientMock.dictionaryVersionSnapshot.mockResolvedValue({
      revision: 3,
      change_type: "update",
      created_at: "2026-05-29T12:00:00Z",
      dictionary: {
        name: "Core Terms",
        is_active: true,
        category: "General"
      },
      entries: [{ term: "alpha" }, { term: "beta" }]
    })
    tldwClientMock.revertDictionaryVersion.mockResolvedValue({
      message: "Revision 3 restored."
    })

    render(
      <DictionaryVersionHistoryModal
        open
        dictionary={dictionary}
        onClose={vi.fn()}
      />
    )

    await screen.findByText("r3")
    await user.click(screen.getByRole("button", { name: "Revert to revision 3" }))

    const title = await screen.findByText("Revision restored")
    const alert = title.closest('[data-ds-component="Alert"]')

    expect(alert).not.toBeNull()
    const alertEl = alert as HTMLElement

    expect(alertEl).toHaveTextContent("Revision 3 restored.")
    expect(within(alertEl).getByText("Revision 3 restored.")).toBeInTheDocument()
  })
})
