import { describe, expect, it } from "vitest"
import {
  MISSING_WIKILINK_PREVIEW_CLASSES,
  buildWikilinkCreateHref,
  buildWikilinkIndex,
  buildWikilinkNoteHref,
  collectWikilinkTargets,
  getActiveWikilinkQuery,
  insertWikilinkAtCursor,
  parseWikilinkHref,
  parseWikilinkResolutions,
  renderContentWithResolvedWikilinks,
  resolveWikilinkTitle,
  tokenizeWikilinks
} from "../wikilinks"

const TARGET_ID = "22222222-2222-4222-8222-222222222222"
const OTHER_ID = "33333333-3333-4333-8333-333333333333"

describe("notes wikilink utilities", () => {
  it("tokenizes wikilinks with positions", () => {
    const content = "See [[Alpha Note]] and [[Beta Note]]."
    const tokens = tokenizeWikilinks(content)
    expect(tokens).toEqual([
      { raw: "[[Alpha Note]]", title: "Alpha Note", start: 4, end: 18 },
      { raw: "[[Beta Note]]", title: "Beta Note", start: 23, end: 36 }
    ])
  })

  it("tokenizes [[id:UUID]] links with the target note id", () => {
    const tokens = tokenizeWikilinks(`Go to [[id:${TARGET_ID.toUpperCase()}]] now`)

    expect(tokens).toHaveLength(1)
    expect(tokens[0].noteId).toBe(TARGET_ID)
    expect(tokens[0].raw).toBe(`[[id:${TARGET_ID.toUpperCase()}]]`)
  })

  it("ignores malformed id links instead of treating them as titles", () => {
    expect(tokenizeWikilinks("[[id:short]] and [[id:]]")).toEqual([])
  })

  it("tokenizes titles that contain single brackets", () => {
    const tokens = tokenizeWikilinks("Read [[[Draft] Proposal]] and [[Q3 [final]]].")

    expect(tokens.map((token) => token.title)).toEqual(["[Draft] Proposal", "Q3 [final]"])
  })

  it("links the innermost title when brackets nest, and never spans lines", () => {
    expect(tokenizeWikilinks("[[outer [[Inner]] tail]]").map((token) => token.title)).toEqual(["Inner"])
    expect(tokenizeWikilinks("[[Broken\nTitle]] [[   ]]")).toEqual([])
  })

  it("resolves ambiguous titles with deterministic id fallback", () => {
    const index = buildWikilinkIndex([
      { id: "note-3", title: "Shared" },
      { id: "note-1", title: "Shared" },
      { id: "note-2", title: "Shared" }
    ])
    expect(resolveWikilinkTitle("Shared", index)).toBe("note-1")
  })

  it("prefers an exact-case title over a case variant", () => {
    const index = buildWikilinkIndex([
      { id: "note-1", title: "shared" },
      { id: "note-2", title: "Shared" }
    ])
    expect(resolveWikilinkTitle("Shared", index)).toBe("note-2")
    expect(resolveWikilinkTitle("  shared ", index)).toBe("note-1")
  })

  it("renders resolved wikilinks as in-page note anchors", () => {
    const index = buildWikilinkIndex([
      { id: "note-a", title: "Alpha Note" },
      { id: "note-b", title: "Beta Note" }
    ])
    const content = "Link [[Alpha Note]] and keep [[Missing Note]] plain."
    const rendered = renderContentWithResolvedWikilinks(content, index)
    // The label is entity-encoded so brackets survive Markdown and LaTeX preprocessing.
    expect(rendered).toContain("[&#91;&#91;Alpha Note&#93;&#93;](#note:note-a)")
    // Without a server answer an unknown title stays plain text.
    expect(rendered).toContain("keep [[Missing Note]] plain.")
  })

  it("uses the server resolution instead of the loaded page", () => {
    const index = buildWikilinkIndex([{ id: "note-on-page", title: "Shared" }])
    const resolutions = parseWikilinkResolutions({
      titles: [
        { title: "Shared", note_id: "note-oldest", note_title: "Shared", candidate_count: 2 },
        { title: "Beyond Page", note_id: "note-far", note_title: "Beyond page", candidate_count: 1 }
      ],
      ids: []
    })

    const rendered = renderContentWithResolvedWikilinks("[[Shared]] and [[Beyond   Page]]", index, {
      resolutions
    })

    expect(rendered).toContain("](#note:note-oldest)")
    expect(rendered).toContain("](#note:note-far)")
    expect(rendered).not.toContain("note-on-page")
  })

  it("renders a server-confirmed missing title as a create-note link", () => {
    const resolutions = parseWikilinkResolutions({
      titles: [{ title: "Missing Note", note_id: null, note_title: null, candidate_count: 0 }],
      ids: []
    })

    const rendered = renderContentWithResolvedWikilinks("Plan [[Missing Note]].", new Map(), {
      resolutions
    })

    expect(rendered).toContain("[&#91;&#91;Missing Note&#93;&#93;](#note-new:Missing%20Note \"")
    expect(parseWikilinkHref("#note-new:Missing%20Note")).toEqual({ kind: "create", title: "Missing Note" })
  })

  it("renders id links with the note title as the label", () => {
    const resolutions = parseWikilinkResolutions({
      titles: [],
      ids: [{ id: TARGET_ID, note_id: TARGET_ID, note_title: "Target Note" }]
    })

    const rendered = renderContentWithResolvedWikilinks(`See [[id:${TARGET_ID}]].`, new Map(), {
      resolutions
    })

    expect(rendered).toBe(`See [&#91;&#91;Target Note&#93;&#93;](#note:${TARGET_ID}).`)
  })

  it("links an id before the server answers, and leaves a confirmed-missing id plain", () => {
    const content = `See [[id:${OTHER_ID}]].`

    expect(renderContentWithResolvedWikilinks(content, new Map())).toContain(`](#note:${OTHER_ID})`)

    const missing = parseWikilinkResolutions({
      titles: [],
      ids: [{ id: OTHER_ID, note_id: null, note_title: null }]
    })
    expect(renderContentWithResolvedWikilinks(content, new Map(), { resolutions: missing })).toBe(content)
  })

  it("round-trips note and create hrefs, including Markdown-significant characters", () => {
    const title = 'A (b) "c" [d] 100% & more!'

    expect(parseWikilinkHref(buildWikilinkNoteHref("note 1/2"))).toEqual({ kind: "note", noteId: "note 1/2" })
    expect(parseWikilinkHref(buildWikilinkCreateHref(title))).toEqual({ kind: "create", title })
    expect(buildWikilinkCreateHref(title)).not.toMatch(/[\s()"]/)
    expect(parseWikilinkHref("https://example.com/#note:1")).toBeNull()
    expect(parseWikilinkHref("#heading-anchor")).toBeNull()
  })

  it("styles exactly the create-note hrefs as missing links", () => {
    // Tailwind needs the href prefix as a literal, so guard it against drift.
    expect(MISSING_WIKILINK_PREVIEW_CLASSES).toContain(`a[href^='${buildWikilinkCreateHref("")}']`)
    expect(buildWikilinkNoteHref("x").startsWith(buildWikilinkCreateHref(""))).toBe(false)
  })

  it("collects unique link targets for the resolve request", () => {
    const content = `[[Alpha]] [[ alpha ]] [[Beta   Note]] [[id:${TARGET_ID}]] [[id:${TARGET_ID.toUpperCase()}]] [[id:bad]]`

    expect(collectWikilinkTargets(content)).toEqual({
      titles: ["Alpha", "Beta Note"],
      ids: [TARGET_ID]
    })
  })

  it("tolerates a malformed resolve response", () => {
    const resolutions = parseWikilinkResolutions({ titles: "nope", ids: [null, { id: 7 }] })

    expect(resolutions.titles.size).toBe(0)
    expect(resolutions.ids.size).toBe(0)
    expect(parseWikilinkResolutions(null)).toBeNull()
  })

  it("detects active wikilink query and inserts selected title", () => {
    const content = "Research [[Al"
    const query = getActiveWikilinkQuery(content, content.length)
    expect(query).toEqual({
      start: 9,
      end: 13,
      query: "Al"
    })
    const inserted = insertWikilinkAtCursor(content, query!, "Alpha Note")
    expect(inserted.content).toBe("Research [[Alpha Note]]")
    expect(inserted.cursor).toBe(23)
  })

  it("inserts an id link when the picked title is ambiguous", () => {
    const content = "Research [[Sha"
    const query = getActiveWikilinkQuery(content, content.length)

    const inserted = insertWikilinkAtCursor(content, query!, "Shared", TARGET_ID)

    expect(inserted.content).toBe(`Research [[id:${TARGET_ID}]]`)
    // A non-UUID id can't be linked by id, so the title form is kept.
    expect(insertWikilinkAtCursor(content, query!, "Shared", "note-1").content).toBe("Research [[Shared]]")
  })
})
