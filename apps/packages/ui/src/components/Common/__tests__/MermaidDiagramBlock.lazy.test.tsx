import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { expect, it, vi } from "vitest"
import { MermaidDiagramBlock } from "../MermaidDiagramBlock"
import type { MermaidRenderState } from "../Mermaid"

const previewLoad = vi.hoisted(() => {
  let release!: () => void
  const ready = new Promise<void>((resolve) => {
    release = resolve
  })
  return { requests: 0, blocked: false, ready, release }
})

vi.mock("../MermaidPreviewDialog", async (importOriginal) => {
  previewLoad.requests += 1
  if (previewLoad.blocked) await previewLoad.ready
  return importOriginal()
})

vi.mock("../Mermaid", async () => {
  const ReactModule = await import("react")
  const Mermaid = ({
    code,
    onRenderStateChange
  }: {
    code: string
    onRenderStateChange: (state: MermaidRenderState) => void
  }) => {
    ReactModule.useEffect(() => {
      onRenderStateChange({
        status: "success",
        svg: `<svg><text>${code}</text></svg>`
      })
    }, [code, onRenderStateChange])
    return (
      <div role="img" aria-label="Mermaid diagram">
        {code}
      </div>
    )
  }
  return { default: Mermaid }
})

vi.mock("antd", () => ({
  Modal: ({
    children,
    open,
    title
  }: {
    children: React.ReactNode
    open: boolean
    title: string
  }) =>
    open ? (
      <div role="dialog" aria-label={title}>
        {children}
      </div>
    ) : null,
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))
vi.mock("@/store/artifacts", () => ({
  useArtifactsStore: () => ({ openArtifact: vi.fn() })
}))

it("loads preview only on first use and retires a pending open when the source changes", async () => {
  previewLoad.blocked = true
  const view = render(<MermaidDiagramBlock source="original diagram" />)
  expect(
    screen.getByRole("img", { name: "Mermaid diagram" })
  ).toHaveTextContent("original diagram")
  expect(previewLoad.requests).toBe(0)

  fireEvent.click(screen.getByRole("button", { name: "Open Mermaid preview" }))
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  view.rerender(<MermaidDiagramBlock source="replacement diagram" />)
  await act(async () => {
    previewLoad.release()
    await previewLoad.ready
  })
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument()

  fireEvent.click(screen.getByRole("button", { name: "Open Mermaid preview" }))
  expect(
    await screen.findByRole("dialog", { name: "Mermaid diagram preview" })
  ).toBeInTheDocument()
  expect(screen.getByTestId("mermaid-preview-canvas")).toHaveTextContent(
    "replacement diagram"
  )
  expect(screen.getByTestId("mermaid-preview-canvas")).not.toHaveTextContent(
    "original diagram"
  )
  fireEvent.click(screen.getByRole("button", { name: "Zoom in" }))
  fireEvent.click(screen.getByRole("button", { name: "Close Mermaid preview" }))
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole("button", { name: "Open Mermaid preview" }))
  expect(
    await screen.findByLabelText("Mermaid preview zoom level")
  ).toHaveTextContent("100%")
  expect(previewLoad.requests).toBe(1)
})
