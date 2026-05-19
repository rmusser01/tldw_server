import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { FlashcardMarkdownSnippet } from "../FlashcardMarkdownSnippet"

describe("FlashcardMarkdownSnippet", () => {
  it("renders inline markdown formatting for snippet previews", () => {
    render(<FlashcardMarkdownSnippet content={"**Important** concept"} />)

    expect(screen.getByText("Important").tagName).toBe("STRONG")
    expect(screen.getByText("concept")).toBeInTheDocument()
  })

  it("prevents markdown link clicks from bubbling to the parent row", () => {
    const onRowClick = vi.fn()

    render(
      <div onClick={onRowClick}>
        <FlashcardMarkdownSnippet content={"[Reference](https://example.com)"} />
      </div>
    )

    fireEvent.click(screen.getByRole("link", { name: "Reference" }))

    expect(onRowClick).not.toHaveBeenCalled()
  })
})
