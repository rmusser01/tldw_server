import React from "react"
import { act, cleanup, render, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import initHeadManager from "next/dist/client/head-manager"
import { HeadManagerContext } from "next/dist/shared/lib/head-manager-context.shared-runtime"
import { publishOpenNoteTitle } from "@/routes/route-document-title"

vi.mock("next/dynamic", () => ({
  default: (_load: unknown, options: { ssr?: boolean }) => {
    if (options.ssr !== false) throw new Error("Shared routes require browser-only loading")
    return () => <main>Shared route</main>
  }
}))

import Notes from "@web/pages/notes"

afterEach(() => {
  cleanup()
  publishOpenNoteTitle(null)
  document.head.innerHTML = ""
})

describe("Notes page title (XP-05)", () => {
  it("names the open note, and goes back to the page name when none is open", async () => {
    const manager = initHeadManager()
    render(
      <HeadManagerContext.Provider value={manager}>
        <Notes />
      </HeadManagerContext.Provider>
    )
    await waitFor(() => expect(document.title).toBe("Notes | tldw"))

    act(() => publishOpenNoteTitle("Thesis chapter draft"))
    await waitFor(() => expect(document.title).toBe("Thesis chapter draft · Notes | tldw"))

    act(() => publishOpenNoteTitle(null))
    await waitFor(() => expect(document.title).toBe("Notes | tldw"))
  })
})
