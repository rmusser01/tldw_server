import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import {
  VirtualChatTimeline,
  type ChatTimelineNavigation
} from "../VirtualChatTimeline"
import { EditMessageForm } from "@/components/Common/Playground/EditMessageForm"

const scrollToOffset = vi.hoisted(() => vi.fn())
const getOffsetForIndex = vi.hoisted(() =>
  vi.fn((index: number): [number, string] | undefined => [
    index * 160,
    "center"
  ])
)
const windowRange = vi.hoisted(() => ({ start: 0, end: 3 }))
vi.mock("@tanstack/react-virtual", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@tanstack/react-virtual")>()
  return {
    ...actual,
    useVirtualizer: (options: {
      count: number
      rangeExtractor: (range: {
        startIndex: number
        endIndex: number
        overscan: number
        count: number
      }) => number[]
    }) => ({
      getVirtualItems: () =>
        options
          .rangeExtractor({
            startIndex: windowRange.start,
            endIndex: windowRange.end,
            overscan: 0,
            count: options.count
          })
          .map((index) => ({ index, start: index * 160 })),
      getTotalSize: () => options.count * 160,
      measureElement: () => {},
      getOffsetForIndex,
      scrollToOffset
    })
  }
})
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key })
}))

const save = vi.fn()
function Message({ id }: { id: string }) {
  const [editing, setEditing] = React.useState(false)
  return (
    <div data-testid={id}>
      {editing ? (
        <EditMessageForm
          value={`Original ${id}`}
          isBot={false}
          onClose={() => setEditing(false)}
          onSumbit={(value) => save(id, value)}
        />
      ) : (
        <button onClick={() => setEditing(true)}>Edit {id}</button>
      )}
    </div>
  )
}
const blocks = Array.from({ length: 201 }, (_, index) => `message-${index}`)
const parent = { current: document.createElement("div") }
const messageBlocks = new Map(blocks.map((_, index) => [index, index]))
const timeline = (owner = "owner-A", rows = blocks) => (
  <VirtualChatTimeline
    key={owner}
    blocks={rows}
    getKey={(index) => rows[index]}
    messageBlocks={messageBlocks}
    scrollParentRef={parent}
    renderBlock={(id) => <Message id={id} />}
  />
)
beforeEach(() => {
  windowRange.start = 0
  windowRange.end = 3
  save.mockClear()
  scrollToOffset.mockClear()
  getOffsetForIndex
    .mockReset()
    .mockImplementation((index: number) => [index * 160, "center"])
})

it.each(["save", "cancel"])(
  "retains the editor and draft outside the visible range until explicit %s",
  (action) => {
    const view = render(timeline())
    fireEvent.click(screen.getByText("Edit message-1"))
    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "Unsaved draft" }
    })
    windowRange.start = 190
    windowRange.end = 193
    view.rerender(timeline())
    expect(screen.getByRole("textbox")).toHaveValue("Unsaved draft")
    expect(
      view.container.querySelectorAll("[data-timeline-block]")
    ).toHaveLength(5)
    windowRange.start = 0
    windowRange.end = 3
    view.rerender(timeline())
    fireEvent.click(screen.getByRole("button", { name: action, exact: true }))
    expect(save.mock.calls).toEqual(
      action === "save" ? [["message-1", "Unsaved draft"]] : []
    )
    windowRange.start = 190
    windowRange.end = 193
    view.rerender(timeline())
    expect(screen.queryByTestId("message-1")).toBeNull()
    expect(
      view.container.querySelectorAll("[data-timeline-block]")
    ).toHaveLength(4)
  }
)

it("resolves retained keys against the current block map and drops removed identities", () => {
  const view = render(timeline())
  fireEvent.click(screen.getByText("Edit message-1"))
  fireEvent.change(screen.getByRole("textbox"), {
    target: { value: "Stable draft" }
  })
  windowRange.start = 190
  windowRange.end = 193
  view.rerender(timeline("owner-A", [...blocks.slice(2), blocks[0], blocks[1]]))
  expect(screen.getByRole("textbox")).toHaveValue("Stable draft")
  view.rerender(
    timeline(
      "owner-A",
      blocks.filter((id) => id !== "message-1")
    )
  )
  expect(screen.queryByRole("textbox")).toBeNull()
})

it("does not carry an old owner's editor into another timeline with matching message IDs", () => {
  const view = render(timeline())
  fireEvent.click(screen.getByText("Edit message-1"))
  fireEvent.change(screen.getByRole("textbox"), {
    target: { value: "Owner A private draft" }
  })
  view.rerender(timeline("owner-B"))
  expect(screen.queryByRole("textbox")).toBeNull()
  fireEvent.click(screen.getByText("Edit message-1"))
  expect(screen.getByRole("textbox")).toHaveValue("Original message-1")
})

it("uses one coarse offset for an already-mounted retained editor", async () => {
  const navigationRef = { current: null as ChatTimelineNavigation | null }
  const renderTimeline = () => (
    <VirtualChatTimeline
      blocks={blocks}
      getKey={(index) => blocks[index]}
      messageBlocks={messageBlocks}
      scrollParentRef={parent}
      navigationRef={navigationRef}
      renderBlock={(id) => <Message id={id} />}
    />
  )
  const view = render(renderTimeline())
  fireEvent.click(screen.getByText("Edit message-1"))
  windowRange.start = 190
  windowRange.end = 193
  view.rerender(renderTimeline())
  expect(screen.getByRole("textbox")).toBeInTheDocument()
  await act(async () => {
    expect(await navigationRef.current!.reveal(1)).toBe(true)
  })
  expect(scrollToOffset).toHaveBeenCalledWith(160, { align: "start" })
})

it("rejects unavailable offset without starting a scroll", async () => {
  const navigationRef = { current: null as ChatTimelineNavigation | null }
  render(
    <VirtualChatTimeline
      blocks={blocks}
      getKey={(index) => blocks[index]}
      messageBlocks={messageBlocks}
      scrollParentRef={parent}
      navigationRef={navigationRef}
      renderBlock={(id) => <Message id={id} />}
    />
  )
  getOffsetForIndex.mockReturnValueOnce(undefined)
  expect(await navigationRef.current!.reveal(1)).toBe(false)
  expect(scrollToOffset).not.toHaveBeenCalled()
})
