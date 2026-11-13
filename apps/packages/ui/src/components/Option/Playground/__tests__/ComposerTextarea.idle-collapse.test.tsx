import React from "react"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ComposerTextarea } from "../ComposerTextarea"
import userEvent from "@testing-library/user-event"
import { useActionBarVisibility } from "@/hooks/playground/useActionBarVisibility"
import useDynamicTextareaSize from "@/hooks/useDynamicTextareaSize"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string; count?: number }) =>
      typeof fallback === "string"
        ? fallback
        : (fallback?.defaultValue ?? _key).replace("{{count}}", String(fallback?.count ?? ""))
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

// AX-12: the slash menu is a listbox the textarea (a textbox, not a
// combobox) controls, so a screen reader hears each command while arrowing
// and how many there are.
describe("ComposerTextarea slash suggestions", () => {
  const commands = [
    { id: "slash-search", command: "search", label: "Search", description: "Search your knowledge" },
    { id: "slash-web", command: "web", label: "Web search" }
  ]

  it("points the textbox into the slash menu's listbox of options", async () => {
    const view = render(
      <ComposerTextarea {...(createProps({ showSlashMenu: true, slashCommands: commands, slashActiveIndex: 0 }) as any)} />
    )
    const textarea = screen.getByRole("textbox", { name: "Message input" })
    expect(textarea).not.toHaveAttribute("role")
    expect(textarea).not.toHaveAttribute("aria-expanded")
    expect(textarea).toHaveAttribute("aria-autocomplete", "list")
    const listbox = screen.getByRole("listbox")
    expect(textarea).toHaveAttribute("aria-controls", listbox.id)
    const options = within(listbox).getAllByRole("option")
    expect(options).toHaveLength(2)
    expect(options.map((option) => option.id).every(Boolean)).toBe(true)
    expect(textarea).toHaveAttribute("aria-activedescendant", options[0].id)
    expect(options[0]).toHaveAttribute("aria-selected", "true")
    await waitFor(() => expect(screen.getByTestId("suggestion-count-status")).toHaveTextContent("2 commands"))

    view.rerender(
      <ComposerTextarea {...(createProps({ showSlashMenu: true, slashCommands: commands, slashActiveIndex: 1 }) as any)} />
    )
    expect(textarea).toHaveAttribute("aria-activedescendant", within(screen.getByRole("listbox")).getAllByRole("option")[1].id)
  })

  it("controls nothing while the menu is closed", () => {
    render(<ComposerTextarea {...(createProps() as any)} />)
    const textarea = screen.getByRole("textbox", { name: "Message input" })
    expect(textarea).toHaveAttribute("aria-autocomplete", "list")
    expect(textarea).not.toHaveAttribute("aria-controls")
    expect(textarea).not.toHaveAttribute("aria-activedescendant")
  })

  it("picks a command without taking focus from the textarea", () => {
    const onSlashSelect = vi.fn()
    render(
      <ComposerTextarea {...(createProps({ showSlashMenu: true, slashCommands: commands, onSlashSelect }) as any)} />
    )
    const option = screen.getAllByRole("option")[1]
    expect(option).not.toHaveAttribute("tabindex")
    expect(fireEvent.mouseDown(option)).toBe(false)
    fireEvent.click(option)
    expect(onSlashSelect).toHaveBeenCalledWith(commands[1])
  })

  it("announces when no command matches", async () => {
    render(<ComposerTextarea {...(createProps({ showSlashMenu: true, slashCommands: [] }) as any)} />)
    expect(screen.getByRole("textbox", { name: "Message input" })).not.toHaveAttribute("aria-controls")
    expect(screen.queryByRole("listbox")).toBeNull()
    await waitFor(() => expect(screen.getByTestId("suggestion-count-status")).toHaveTextContent("No results found"))
  })
})
