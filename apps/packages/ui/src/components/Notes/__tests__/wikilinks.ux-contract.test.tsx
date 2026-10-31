/**
 * UX review 2026-10 contract reproduction NE-02 (#3110, decision D2: accept
 * both `[[Title]]` and `[[id:UUID]]`).
 *
 * Renders the note preview through the real Markdown component (stage7 mocks
 * MarkdownPreview, which hid this defect). It was an `it.fails` reproduction
 * until NE-02 was fixed; it now guards the fix as a plain `it(...)`.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import Markdown from "@/components/Common/Markdown"
import { buildWikilinkIndex, renderContentWithResolvedWikilinks } from "../wikilinks"

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))

describe("Notes wikilink preview UX contract reproductions (#3110)", { timeout: 60_000 }, () => {
  // NE-02 (#3110): wikilinks.ts used to emit [raw](note://id), and Markdown.tsx transformMarkdownUrl blanks unknown schemes, so the link had no target.
  it("NE-02 (#3110): a [[Title]] wikilink in the note preview links to that note", async () => {
    const index = buildWikilinkIndex([{ id: "note-42", title: "Target Note" }])
    const previewContent = renderContentWithResolvedWikilinks("See [[Target Note]] for details.", index)

    render(<Markdown message={previewContent} />)

    const label = await screen.findByText("[[Target Note]]")
    const anchor = label.closest("a")
    expect(anchor?.getAttribute("href") ?? "").toContain("note-42")
  })
})
