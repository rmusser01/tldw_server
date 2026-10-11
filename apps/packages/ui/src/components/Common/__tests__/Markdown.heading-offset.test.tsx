import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import Markdown from "../Markdown"

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) => [defaultValue]
}))

const headings = "# Title *emphasis*\n\nSubtitle\n---\n\n### Three\n\n#### Four\n\n##### Five\n\n###### Six"

describe.each(["safe_markdown", "st_compat"] as const)("Markdown %s heading offset", mode => {
  it("keeps default heading levels on other surfaces", () => {
    render(<Markdown message={headings} richTextModeOverride={mode} />)
    expect(screen.getAllByRole("heading").map(node => node.tagName)).toEqual([
      "H1", "H2", "H3", "H4", "H5", "H6"
    ])
  })

  it("offsets parsed headings and clamps at H6 without changing content", () => {
    const { container } = render(
      <Markdown message={headings} richTextModeOverride={mode} headingOffset={1} />
    )
    expect(screen.getAllByRole("heading").map(node => node.tagName)).toEqual([
      "H2", "H3", "H4", "H5", "H6", "H6"
    ])
    expect(container.querySelector("h2 em")).toHaveTextContent("emphasis")
  })

  it("does not rewrite heading-looking text in code or links", () => {
    const { container } = render(
      <Markdown
        message={'```text\n# literal\n<h1>code</h1>\n```\n\n[Link](https://example.test/#heading)'}
        richTextModeOverride={mode}
        headingOffset={1}
        codeBlockVariant="plain"
      />
    )
    expect(screen.queryByRole("heading")).not.toBeInTheDocument()
    expect(container).toHaveTextContent("# literal")
    expect(container).toHaveTextContent("<h1>code</h1>")
    expect(screen.getByRole("link", { name: "Link" })).toHaveAttribute("href", "https://example.test/#heading")
  })
})

it("preserves ReactMarkdown anchors and highlighting after the offset", () => {
  const { container } = render(
    <Markdown message="# Title" headingOffset={1} headingAnchorIds={["section-title"]} searchQuery="Title" />
  )
  expect(screen.getByRole("heading", { level: 2 })).toHaveAttribute("id", "section-title")
  expect(container.querySelector("h2 mark")).toHaveTextContent("Title")
})

it("offsets sanitized raw headings in ST mode without weakening sanitization", () => {
  const { container } = render(
    <Markdown
      message={'<h1 id="raw-title" onclick="bad()">Raw <em>title</em></h1><script>bad()</script>'}
      richTextModeOverride="st_compat"
      headingOffset={1}
    />
  )
  expect(screen.getByRole("heading", { level: 2 })).toHaveAttribute("id", "raw-title")
  expect(container.querySelector("[onclick],script")).toBeNull()
})

it("uses the offset level for accessible ST headings with an explicit aria-level", () => {
  render(
    <>
      <h1>Chat Workspace</h1>
      <Markdown
        message={'<h1 role="heading" aria-level="1">Raw title</h1>'}
        richTextModeOverride="st_compat"
        headingOffset={1}
      />
    </>
  )
  expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1)
  expect(screen.getByRole("heading", { name: "Raw title", level: 2 })).toHaveProperty("tagName", "H2")
})
