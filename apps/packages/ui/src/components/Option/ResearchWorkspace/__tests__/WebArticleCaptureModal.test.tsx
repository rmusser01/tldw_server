import { fireEvent, render, screen } from "@testing-library/react"
import { expect, it, vi } from "vitest"
import { WebArticleCaptureModal } from "../SourcesPane/WebArticleCaptureModal"
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_key: string, fallback: string) => fallback })
}))
it("discloses the extra Note and offers full text expansion and explicit save/cancel", () => {
  const capture = {
    source: { title: "Article" },
    preview: {
      text: "entire accepted text\n<script>plain text</script>",
      title: "Article",
      capturedAt: "2026-10-07T00:00:00Z"
    },
    pending: null,
    busy: false,
    error: null,
    notice: null,
    open: vi.fn(),
    extract: vi.fn(),
    save: vi.fn(),
    cancel: vi.fn()
  }
  render(<WebArticleCaptureModal capture={capture as never} />)
  expect(screen.getByText(/additional capture Note/i)).toBeInTheDocument()
  fireEvent.click(screen.getByRole("button", { name: "Expand text" }))
  expect(screen.getByText(/entire accepted text/)).toBeInTheDocument()
  expect(document.querySelector("script")).toBeNull()
  fireEvent.click(screen.getByRole("button", { name: "Save capture" }))
  expect(capture.save).toHaveBeenCalledOnce()
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }))
  expect(capture.cancel).toHaveBeenCalledOnce()
})
