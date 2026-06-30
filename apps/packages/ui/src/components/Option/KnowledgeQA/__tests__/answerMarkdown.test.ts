import { describe, expect, it } from "vitest"
import {
  buildKnowledgeExportTrustLines,
  remarkCitationLinks,
  splitCitationSegments,
} from "../answerMarkdown"

describe("answerMarkdown citation transform", () => {
  it("splits text into citation and non-citation segments", () => {
    expect(splitCitationSegments("Alpha [1] beta [2].")).toEqual([
      { type: "text", value: "Alpha " },
      { type: "citation", index: 1 },
      { type: "text", value: " beta " },
      { type: "citation", index: 2 },
      { type: "text", value: "." },
    ])
  })

  it("ignores malformed citation tokens", () => {
    expect(splitCitationSegments("Alpha [x] [] [01]")).toEqual([
      { type: "text", value: "Alpha [x] [] " },
      { type: "citation", index: 1 },
    ])
  })

  it("converts text citations to markdown link nodes but preserves code nodes", () => {
    const tree: any = {
      type: "root",
      children: [
        {
          type: "paragraph",
          children: [{ type: "text", value: "Claim [3] and [4]." }],
        },
        {
          type: "code",
          lang: "js",
          value: "const x = '[7]'",
        },
      ],
    }

    const transform = remarkCitationLinks()
    transform(tree)

    const paragraphChildren = tree.children[0].children
    expect(paragraphChildren).toEqual([
      { type: "text", value: "Claim " },
      {
        type: "link",
        url: "citation://3",
        children: [{ type: "text", value: "[3]" }],
      },
      { type: "text", value: " and " },
      {
        type: "link",
        url: "citation://4",
        children: [{ type: "text", value: "[4]" }],
      },
      { type: "text", value: "." },
    ])
    expect(tree.children[1].value).toBe("const x = '[7]'")
  })

  it("labels unsupported draft exports with trust and evidence origin metadata", () => {
    expect(
      buildKnowledgeExportTrustLines({
        trustState: "uncited_degraded_answer",
        evidenceOrigin: "local_library",
      })
    ).toEqual([
      "Trust: unsupported draft",
      "Answer status: Uncited answer",
      "Evidence origin: local library",
    ])
  })

  it("labels cited exports as supported answer content", () => {
    expect(
      buildKnowledgeExportTrustLines({
        trustState: "cited_answer",
        evidenceOrigin: "mixed",
      })
    ).toEqual([
      "Trust: cited answer",
      "Answer status: Cited answer",
      "Evidence origin: mixed sources",
    ])
  })
})
