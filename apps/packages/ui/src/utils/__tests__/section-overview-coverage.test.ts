import { describe, expect, it } from "vitest"
import { getSectionOverviewCoverage } from "../section-overview-coverage"

const sections = () => [
  { title: "1. Agency A", status: "completed", batches: [{ evidence: [{ id: "E9.8.7", quote: "A proposes reform." }] }] },
  { title: "27. Financial Regulatory Agencies", status: "completed", batches: [{ evidence: [{ id: "E2.1.1", quote: "B proposes oversight." }] }] }
]
const coverage = (text: string) => getSectionOverviewCoverage(text, sections())

// Keep the literal parser cases aligned with test_section_overview_coverage.py.
describe("structural overview chapter coverage", () => {
  it("uses actual evidence ownership without inferring chapters from ID prefixes", () => {
    expect(coverage("Agency A proposes reform [E9.8.7].")).toEqual({
      selected_sections: 2, completed_sections: 2, cited_sections: 1,
      missing_sections: [{ index: 1, title: "27. Financial Regulatory Agencies" }], complete: false
    })
  })

  it.each([
    "Both propose reforms [E9.8.7] [E2.1.1].",
    "Both propose reforms [E9.8.7], [E2.1.1].",
    "Both propose reforms [E9.8.7]; [E2.1.1].",
    "Both propose reforms [E9.8.7]\n[E2.1.1].",
    "Agency A proposes reform [E9.8.7].\n\nFinancial regulators propose oversight [E2.1.1].",
    "- Agency A proposes reform [E9.8.7].\n- Financial regulators propose oversight [E2.1.1]."
  ])("counts literal citations attached to prose: %s", text => {
    expect(coverage(text).complete).toBe(true)
  })

  it.each([
    "[E9.8.7] [E2.1.1]",
    "# Agency A [E9.8.7]\n\n## Financial Regulatory Agencies [E2.1.1]",
    "Agency A [E9.8.7]\n===\n\nFinancial Regulatory Agencies [E2.1.1]\n---",
    "1. Agency A [E9.8.7].\n27. Financial Regulatory Agencies [E2.1.1].",
    "**Agency A** [E9.8.7].\n\n**Financial Regulatory Agencies** [E2.1.1].",
    "**Chapter 1: Agency A** [E9.8.7].\n\n**Section 27: Financial Regulatory Agencies** [E2.1.1].",
    "```markdown\nA proposes reform [E9.8.7].\nB proposes oversight [E2.1.1].\n```",
    "~~~\nA proposes reform [E9.8.7].\nB proposes oversight [E2.1.1].\n~~~",
    "> ```\n> Agency A proposes reform [E9.8.7].\n> Financial regulators propose oversight [E2.1.1].\n> ```",
    "- ```\n  Agency A proposes reform [E9.8.7].\n  Financial regulators propose oversight [E2.1.1].\n  ```",
    "    A proposes reform [E9.8.7].\n    B proposes oversight [E2.1.1].",
    "`A proposes reform [E9.8.7].`\n\n`B proposes oversight [E2.1.1].`",
    "[E9.8.7](https://example.test/a)\n\n![E2.1.1](https://example.test/b)",
    "[E9.8.7]: https://example.test/a\n[E2.1.1]: https://example.test/b",
    "References: [E9.8.7] [E2.1.1].",
    "Evidence IDs: [E9.8.7, E2.1.1].",
    "Citation IDs: [E9.8.7, E2.1.1].",
    "# References\n\nA proposes reform [E9.8.7].\n\n## Detail\nB proposes oversight [E2.1.1].",
    "<!-- A proposes reform [E9.8.7].\nB proposes oversight [E2.1.1]. -->",
    "A proposes reform E9.8.7. B proposes oversight E2.1.1.",
    "Both propose reform [E9.8.7–E2.1.1].",
    "Both propose reform [E9.8.7, E2.1.1–E2.1.5].",
    "A proposes reform [E1.1.1]. B proposes oversight [E27.1.1].",
    "1. [E9.8.7]\n2. [E2.1.1]",
    "Source section: Agency A [E9.8.7].\n\nChapter: Financial Regulatory Agencies [E2.1.1].",
    "- **Source section:** Agency A [E9.8.7].\n- Section: 27. Financial Regulatory Agencies [E2.1.1]."
  ])("ignores headings, inventories, code and unregistered citations: %s", text => {
    expect(coverage(text).cited_sections).toBe(0)
  })

  it.each([". ", "! ", "? ", "\n\n"])("does not borrow prose across a boundary %j", separator => {
    expect(coverage(`Agency A proposes reform [E9.8.7]${separator}[E2.1.1]`).cited_sections).toBe(1)
  })

  it.each([
    "10. ```\n    sample [E2.1.1]\n    ```\n\nAgency A proposes reform [E9.8.7].",
    "> ```\n> sample [E2.1.1]\n\nAgency A proposes reform [E9.8.7]."
  ])("reads prose following a closed code container: %s", text => {
    expect(coverage(text).missing_sections).toEqual([{ index: 1, title: "27. Financial Regulatory Agencies" }])
  })

  it("ends the reference inventory at the next same-level heading", () => {
    const text = "## References\nA proposes reform [E9.8.7].\n\n### Details\nB proposes oversight [E2.1.1].\n\n## Proposals\nAgency A proposes reform [E9.8.7]."
    expect(coverage(text).missing_sections).toEqual([{ index: 1, title: "27. Financial Regulatory Agencies" }])
  })

  it.each(["\n", "\r\n", "\r"])("recognizes reference definitions with line ending %j", newline => {
    expect(coverage(`A proposes reform [E9.8.7].${newline}[E9.8.7]: https://example.test`).cited_sections).toBe(0)
  })

  it.each(["\n", "\r\n", "\r"])("recognizes inventory headings with line ending %j", newline => {
    expect(coverage(["## References", "A proposes reform [E9.8.7].", "## Proposals", "B proposes oversight [E2.1.1]."].join(newline)).missing_sections)
      .toEqual([{ index: 0, title: "1. Agency A" }])
  })

  it("does not let an ambiguous evidence ID cover either owner", () => {
    const input = sections()
    input[1].batches[0].evidence.push({ id: "E9.8.7", quote: "Different owned passage." })
    expect(getSectionOverviewCoverage("A proposes reform [E9.8.7].", input).cited_sections).toBe(0)
  })

  it("keeps chapters with duplicate titles distinct", () => {
    const input = sections()
    input[1].title = input[0].title
    expect(getSectionOverviewCoverage("A proposes reform [E9.8.7].", input).missing_sections)
      .toEqual([{ index: 1, title: "1. Agency A" }])
  })

  it("counts only completed chapters in the overview denominator", () => {
    const input = sections()
    input[1].status = "failed"
    input.push({ title: "Unread", status: "unread", batches: [] })
    expect(getSectionOverviewCoverage("A proposes reform [E9.8.7].", input)).toEqual({
      selected_sections: 3, completed_sections: 1, cited_sections: 1, missing_sections: [], complete: true
    })
  })

  it.each([{ input: [] }, { input: [{ title: "Unread", status: "unread", batches: [] }] }])("does not claim complete without completed chapters: $input", ({ input }) => {
    expect(getSectionOverviewCoverage("A proposes reform [E9.8.7].", input).complete).toBe(false)
  })

  it("does not use Claims verdicts or mutate saved evidence", () => {
    const input = sections().map(section => ({ ...section, claim_verification: { verdict: "unsupported" } }))
    const before = JSON.stringify(input)
    expect(getSectionOverviewCoverage("A proposes reform [E9.8.7]. B proposes oversight [E2.1.1].", input).complete).toBe(true)
    expect(JSON.stringify(input)).toBe(before)
  })

  it.each([["E9.8.7", "E2.1.1"], ["E2.1.1", "E9.8.7"]])("handles either registered ID order %s %s", (first, second) => {
    expect(coverage(`Both propose reform [${first}] [${second}]`).cited_sections).toBe(2)
  })
})
