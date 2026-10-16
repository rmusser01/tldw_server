import { createRequire } from "node:module"
import React from "react"
import { message, Modal, notification } from "antd"
import { act, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

const require = createRequire(import.meta.url)
const antdRequire = createRequire(require.resolve("antd"))
const externalRenderer = antdRequire("@rc-component/util/lib/React/render") as {
  render: (node: React.ReactNode, container: HTMLElement) => void
}

let effectUnmounts = 0
let externalContainer: HTMLDivElement

function ExternalEffectProbe({ label }: { label: string }) {
  React.useEffect(() => () => { effectUnmounts++ }, [])
  return <span>{label}</span>
}

const renderExternalProbe = async (label: string) => {
  externalContainer = document.createElement("div")
  document.body.append(externalContainer)
  await act(async () => {
    externalRenderer.render(<ExternalEffectProbe label={label} />, externalContainer)
  })
  expect(screen.getByText(label)).toBeVisible()
}

// These serial cases exercise real static roots outside RTL's render registry.
// Persistent notices and a queued confirmation must be retired by shared teardown.
describe.sequential("real AntD static roots between tests", () => {
  it("opens persistent notices and queues a confirmation outside RTL", async () => {
    await renderExternalProbe("First external effect")
    message.open({ content: "First persistent static message", duration: 0 })
    notification.open({ title: "First persistent static notification", duration: 0 })
    expect(await screen.findByText("First persistent static message")).toBeInTheDocument()
    expect(await screen.findByText("First persistent static notification")).toBeInTheDocument()
    expect(effectUnmounts).toBe(0)

    // Modal.confirm schedules its renderer on a zero-delay timer. Intentionally
    // finish before that task runs so teardown must handle queued static work.
    Modal.confirm({ title: "First queued static confirmation" })
  })

  it("unmounts old effects and can open fresh static APIs", async () => {
    // Give a leaked confirmation's scheduled render a deterministic opportunity
    // to run before checking isolation; this is a task boundary, not a sleep.
    await act(async () => {
      await new Promise<void>(resolve => setTimeout(resolve, 0))
    })
    expect(screen.queryByText("First persistent static message")).not.toBeInTheDocument()
    expect(screen.queryByText("First persistent static notification")).not.toBeInTheDocument()
    expect(screen.queryByText("First queued static confirmation")).not.toBeInTheDocument()
    expect(effectUnmounts).toBe(1)
    expect(externalContainer).toBeEmptyDOMElement()

    await renderExternalProbe("Second external effect")
    message.open({ content: "Second persistent static message", duration: 0 })
    notification.open({ title: "Second persistent static notification", duration: 0 })
    Modal.confirm({ title: "Second static confirmation" })
    expect(await screen.findByText("Second persistent static message")).toBeInTheDocument()
    expect(await screen.findByText("Second persistent static notification")).toBeInTheDocument()
    expect(await screen.findByRole("dialog", { name: "Second static confirmation" })).toBeInTheDocument()
  })

  it("retires the fresh roots rather than only removing their DOM containers", async () => {
    expect(effectUnmounts).toBe(2)
    expect(externalContainer).toBeEmptyDOMElement()
    expect(screen.queryByText("Second external effect")).not.toBeInTheDocument()
    expect(screen.queryByText("Second persistent static message")).not.toBeInTheDocument()
    expect(screen.queryByText("Second persistent static notification")).not.toBeInTheDocument()
    expect(screen.queryByText("Second static confirmation")).not.toBeInTheDocument()
  })

  it("queues notices without awaiting their holder readiness", () => {
    message.open({ content: "Unawaited previous static message", duration: 0 })
    notification.open({ title: "Unawaited previous static notification", duration: 0 })
  })

  it("does not replay pending notices when fresh static holders open", async () => {
    message.open({ content: "Fresh post-pending static message", duration: 0 })
    notification.open({ title: "Fresh post-pending static notification", duration: 0 })
    expect(await screen.findByText("Fresh post-pending static message")).toBeInTheDocument()
    expect(await screen.findByText("Fresh post-pending static notification")).toBeInTheDocument()
    expect(screen.queryByText("Unawaited previous static message")).not.toBeInTheDocument()
    expect(screen.queryByText("Unawaited previous static notification")).not.toBeInTheDocument()
  })
})
