import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ChatAttachmentImage } from "../ChatAttachmentImage"

vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }))
const errors = vi.hoisted(() => new Map<string, () => void>())
vi.mock("antd", () => ({ Image: (props: React.ImgHTMLAttributes<HTMLImageElement>) => {
  errors.set(props.src!, props.onError as unknown as () => void)
  // eslint-disable-next-line @next/next/no-img-element -- Native img delivers the browser error event in this component test.
  return <img {...props} />
} }))

describe("chat image retrieval recovery", () => {
  it("keeps the valid attachment source and preview attributes", () => {
    render(<ChatAttachmentImage src="data:image/png;base64,original" className="rounded-md" />)
    expect(screen.getByRole("img", { name: "Uploaded Image" })).toHaveAttribute("src", "data:image/png;base64,original")
    expect(screen.queryByRole("status")).toBeNull()
  })
  it("names a failed retrieval without exposing its URL", () => {
    render(<ChatAttachmentImage src="http://chat.test/private-missing.png" />)
    fireEvent.error(screen.getByRole("img"))
    expect(screen.getByRole("status")).toHaveTextContent("Image unavailable. Attach this image again before continuing.")
    expect(screen.queryByRole("img")).toBeNull()
    expect(screen.queryByText(/private-missing/)).toBeNull()
  })
  it("allows a replacement source after a failed image", () => {
    const view = render(<ChatAttachmentImage src="missing" />)
    fireEvent.error(screen.getByRole("img"))
    view.rerender(<ChatAttachmentImage src="replacement" />)
    expect(screen.getByRole("img")).toHaveAttribute("src", "replacement")
    expect(screen.queryByRole("status")).toBeNull()
  })
})

it("ignores a retired image error after the current source also fails", () => {
  const view = render(<ChatAttachmentImage src="old-source" />)
  view.rerender(<ChatAttachmentImage src="new-source" />)
  fireEvent.error(screen.getByRole("img"))
  act(() => errors.get("old-source")!())
  expect(screen.getByRole("status")).toHaveTextContent("Image unavailable.")
  expect(screen.queryByRole("img")).toBeNull()
})

it("retries an earlier source after a distinct replacement", () => {
  const view = render(<ChatAttachmentImage src="source-A" />)
  fireEvent.error(screen.getByRole("img"))
  view.rerender(<ChatAttachmentImage src="source-B" />)
  view.rerender(<ChatAttachmentImage src="source-A" />)
  expect(screen.getByRole("img")).toHaveAttribute("src", "source-A")
  expect(screen.queryByRole("status")).toBeNull()
})

it("ignores a first-incarnation error when returning to the same URL", () => {
  const view = render(<ChatAttachmentImage src="same-A" />)
  const retiredError = errors.get("same-A")!
  view.rerender(<ChatAttachmentImage src="other-B" />)
  view.rerender(<ChatAttachmentImage src="same-A" />)
  act(() => retiredError())
  expect(screen.getByRole("img")).toHaveAttribute("src", "same-A")
  expect(screen.queryByRole("status")).toBeNull()
})
