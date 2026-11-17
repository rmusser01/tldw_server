import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ComposerTextarea } from "../ComposerTextarea"
import userEvent from "@testing-library/user-event"
import { useActionBarVisibility } from "@/hooks/playground/useActionBarVisibility"
import useDynamicTextareaSize from "@/hooks/useDynamicTextareaSize"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

const createProps = (overrides: Record<string, unknown> = {}) => ({
  textareaRef: { current: null } as React.RefObject<HTMLTextAreaElement | null>,
  value: "",
  displayValue: "",
  onChange: vi.fn(),
  onKeyDown: vi.fn(),
  onPaste: vi.fn(),
  onFocus: vi.fn(),
  onSelect: vi.fn(),
  onCompositionStart: vi.fn(),
  onCompositionEnd: vi.fn(),
  onMouseDown: vi.fn(),
  onMouseUp: vi.fn(),
  placeholder: "Type a message...",
  isProMode: true,
  isMobile: false,
  isConnectionReady: true,
  isCollapsed: false,
  ariaExpanded: true,
  formInputProps: {},
  showSlashMenu: false,
  slashCommands: [],
  slashActiveIndex: 0,
  onSlashSelect: vi.fn(),
  onSlashActiveIndexChange: vi.fn(),
  slashEmptyLabel: "No results found",
  showMentions: false,
  filteredTabs: [],
  mentionPosition: null,
  onMentionSelect: vi.fn(),
  onMentionsClose: vi.fn(),
  onMentionRefetch: vi.fn(async () => undefined),
  onMentionsOpen: vi.fn(async () => undefined),
  draftSaved: false,
  ...overrides
})

describe("ComposerTextarea idle collapse", () => {
  it("uses a compact min-height when idle-collapsed", () => {
    render(<ComposerTextarea {...(createProps({ compactWhenInactive: true }) as any)} />)

    const textarea = screen.getByTestId("chat-input")
    expect(textarea).toHaveStyle({ minHeight: "44px" })
  })
})


describe("ComposerTextarea native semantics", () => {
  it("keeps multiline input and composition handlers without claiming an expandable textbox", () => {
    const draft = "  Exact opening\nKeep trailing spaces  "
    const props = createProps({ value: draft, displayValue: draft, ariaExpanded: true })
    render(<ComposerTextarea {...props} />)
    const textarea = screen.getByRole("textbox", { name: "Message input" })
    expect(textarea.tagName).toBe("TEXTAREA")
    expect(textarea).not.toHaveAttribute("aria-expanded")
    expect(textarea).toHaveValue(draft)
    fireEvent.compositionStart(textarea)
    fireEvent.compositionEnd(textarea)
    expect(props.onCompositionStart).toHaveBeenCalledTimes(1)
    expect(props.onCompositionEnd).toHaveBeenCalledTimes(1)
  })
})


it("forwards the sizing callback to the native textarea and retains the focus object", () => {
  const objectRef = React.createRef<HTMLTextAreaElement>()
  const frames: FrameRequestCallback[] = []
  const frame = vi
    .spyOn(window, "requestAnimationFrame")
    .mockImplementation((callback) => {
      frames.push(callback)
      return frames.length
    })
  const cancel = vi
    .spyOn(window, "cancelAnimationFrame")
    .mockImplementation(() => {})
  function SizedTextarea() {
    const attachTextarea = useDynamicTextareaSize(
      objectRef,
      "visible draft",
      120
    )
    return (
      <ComposerTextarea
        {...createProps({
          textareaRef: objectRef,
          attachTextarea,
          value: "visible draft",
          displayValue: "visible draft"
        })}
      />
    )
  }
  const view = render(<SizedTextarea />)
  const node = screen.getByRole("textbox", {
    name: "Message input"
  }) as HTMLTextAreaElement
  Object.defineProperty(node, "scrollHeight", {
    configurable: true,
    get: () => 64
  })
  act(() => frames.splice(0).forEach((callback) => callback(0)))
  expect(node.style.height).toBe("64px")
  expect(objectRef.current).toBe(node)
  objectRef.current!.focus()
  expect(document.activeElement).toBe(node)
  view.unmount()
  expect(objectRef.current).toBeNull()
  frame.mockRestore()
  cancel.mockRestore()
})

function FocusGeometryComposer() {
  const visibility = useActionBarVisibility({ externalPinSources: false })
  return <><div onFocusCapture={visibility.handlers.onFocusCapture} onBlurCapture={visibility.handlers.onBlurCapture}>
    <ComposerTextarea {...createProps({ compactWhenInactive: !visibility.composerTextareaExpanded })} />
    <button type="button">Tools geometry control</button>
  </div><button type="button">Outside composer</button></>
}

describe("composer action focus preserves input geometry", () => {
  it("keeps an idle input compact when a toolbar action takes focus", async () => {
    const user = userEvent.setup()
    const view = render(<FocusGeometryComposer />)
    try {
      await user.click(screen.getByRole("button", { name: "Tools geometry control" }))
      expect(screen.getByTestId("chat-input")).toHaveStyle({ minHeight: "44px" })
    } finally { view.unmount() }
  })
  it("expands on input entry and keeps that height through internal action focus", async () => {
    const user = userEvent.setup()
    const view = render(<FocusGeometryComposer />)
    try {
      await user.click(screen.getByTestId("chat-input"))
      expect(screen.getByTestId("chat-input")).toHaveStyle({ minHeight: "60px" })
      await user.click(screen.getByRole("button", { name: "Tools geometry control" }))
      expect(screen.getByTestId("chat-input")).toHaveStyle({ minHeight: "60px" })
      await user.click(screen.getByRole("button", { name: "Outside composer" }))
      expect(screen.getByTestId("chat-input")).toHaveStyle({ minHeight: "44px" })
    } finally { view.unmount() }
  })
})
