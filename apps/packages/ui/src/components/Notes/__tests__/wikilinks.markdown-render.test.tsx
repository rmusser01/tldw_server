/**
 * NE-02 (#3110, decision D2): both `[[Title]]` and `[[id:UUID]]` render as
 * links in the note preview, and an unresolved title offers to create the note.
 *
 * These render through the real Markdown component, because a mocked preview
 * hid the original defect (the link lost its target).
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import Markdown from "@/components/Common/Markdown"
import {
  buildWikilinkIndex,
  parseWikilinkHref,
  parseWikilinkResolutions,
  renderContentWithResolvedWikilinks
} from "../wikilinks"

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))

const TARGET_ID = "22222222-2222-4222-8222-222222222222"

const anchorFor = async (label: string): Promise<HTMLAnchorElement> => {
  const element = await screen.findByText(label)
  const anchor = element.closest("a")
  expect(anchor).not.toBeNull()
  return anchor as HTMLAnchorElement
}

describe("Notes wikilink preview rendering", { timeout: 60_000 }, () => {
  it("renders an [[id:UUID]] link as a link to that note, labelled with its title", async () => {
    const resolutions = parseWikilinkResolutions({
      titles: [],
      ids: [{ id: TARGET_ID, note_id: TARGET_ID, note_title: "Target Note" }]
    })
    const preview = renderContentWithResolvedWikilinks(`See [[id:${TARGET_ID}]] for details.`, new Map(), {
      resolutions
    })

    render(<Markdown message={preview} />)

    const anchor = await anchorFor("[[Target Note]]")
    expect(parseWikilinkHref(anchor.getAttribute("href") ?? "")).toEqual({ kind: "note", noteId: TARGET_ID })
  })

  it("renders an unresolved [[Title]] as a create-note link", async () => {
    const resolutions = parseWikilinkResolutions({
      titles: [{ title: "Plain Title Test", note_id: null, note_title: null, candidate_count: 0 }],
      ids: []
    })
    const preview = renderContentWithResolvedWikilinks("Later: [[Plain Title Test]].", new Map(), {
      resolutions,
      labels: { createNote: (title) => `Create note "${title}"` }
    })

    render(<Markdown message={preview} />)

    const anchor = await anchorFor("[[Plain Title Test]]")
    expect(parseWikilinkHref(anchor.getAttribute("href") ?? "")).toEqual({
      kind: "create",
      title: "Plain Title Test"
    })
    expect(anchor.getAttribute("title")).toBe('Create note "Plain Title Test"')
  })

  it("keeps Markdown- and LaTeX-significant characters in a title literal", async () => {
    const title = "[Draft] $5 *plan* \\(x\\) & `code`"
    const index = buildWikilinkIndex([{ id: "note-7", title }])
    const preview = renderContentWithResolvedWikilinks(`Read [[${title}]] next.`, index)

    render(<Markdown message={preview} />)

    const anchor = await anchorFor(`[[${title}]]`)
    expect(parseWikilinkHref(anchor.getAttribute("href") ?? "")).toEqual({ kind: "note", noteId: "note-7" })
    expect(anchor.querySelector("em, code, .katex")).toBeNull()
  })

  it("keeps the link target in the SillyTavern-compatible rich text mode", async () => {
    // That mode renders through a separate HTML sanitizer with its own URL allow-list.
    const index = buildWikilinkIndex([{ id: "note-42", title: "Target Note" }])
    const preview = renderContentWithResolvedWikilinks("See [[Target Note]] for details.", index)

    render(<Markdown message={preview} richTextModeOverride="st_compat" />)

    const anchor = await anchorFor("[[Target Note]]")
    expect(parseWikilinkHref(anchor.getAttribute("href") ?? "")).toEqual({ kind: "note", noteId: "note-42" })
  })

  it("leaves ordinary Markdown links and text untouched", async () => {
    const index = buildWikilinkIndex([{ id: "note-42", title: "Target Note" }])
    const preview = renderContentWithResolvedWikilinks(
      "Visit [the docs](https://example.com/docs) and [[Target Note]].",
      index
    )

    render(<Markdown message={preview} />)

    const external = await anchorFor("the docs")
    expect(external.getAttribute("href")).toBe("https://example.com/docs")
    expect(parseWikilinkHref(external.getAttribute("href") ?? "")).toBeNull()
  })
})
