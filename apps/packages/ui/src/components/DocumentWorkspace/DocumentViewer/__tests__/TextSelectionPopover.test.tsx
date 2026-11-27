import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { createPortal } from "react-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { TextSelectionPopover } from "../TextSelectionPopover"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"

const { rafState, messageApi, tooltip, translate, touch } = vi.hoisted(() => ({
  rafState: {
    callbacks: new Map<number, FrameRequestCallback>(),
    nextId: 1
  },
  messageApi: {
    success: vi.fn(),
    error: vi.fn()
  },
  tooltip: vi.fn(),
  translate: vi.fn(),
  touch: { coarse: false }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: string) => defaultValue || _key
  })
}))

vi.mock("@/hooks/document-workspace/useTranslate", () => ({
  useTranslate: () => ({
    mutateAsync: translate,
    isPending: false
  })
}))

vi.mock("@/hooks/document-workspace/useDocumentTTS", () => ({
  useDocumentTTS: () => ({
    speak: vi.fn(),
    stop: vi.fn(),
    state: {
      isPlaying: false,
      currentText: "",
      isLoading: false
    }
  })
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => false,
  useMediaQuery: () => touch.coarse
}))

vi.mock("antd", () => ({
  Button: ({ children, icon, className, loading: _loading, ...props }: any) => (
    <button className={className} {...props}>
      {icon}
      {children}
    </button>
  ),
  Dropdown: ({ children, menu, classNames, open: controlled, onOpenChange }: any) => {
    const [uncontrolled, setOpen] = React.useState(false)
    const open = controlled ?? uncontrolled
    const options = <div className={classNames?.root}>{menu.items.map((item: any) => (
      <button key={item.key} onClick={item.onClick}>{item.label}</button>
    ))}</div>
    return <>
      {React.cloneElement(children, { onClick: () => {
        setOpen(!open)
        onOpenChange?.(!open)
      } })}
      {open && createPortal(options, document.body)}
    </>
  },
  Tooltip: ({ children, title, trigger }: any) => {
    tooltip(title, trigger)
    return children
  },
  Modal: ({ open, title, children, footer }: any) => (
    open ? <div role="dialog" aria-label={title}>{children}{footer}</div> : null
  ),
  Spin: () => <div>Loading</div>,
  Select: ({ value, onChange, options = [] }: any) => (
    <select value={value} onChange={(event) => onChange?.(event.target.value)}>
      {options.map((option: any) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  ),
  message: messageApi
}))

function flushAnimationFrames() {
  const callbacks = [...rafState.callbacks.values()]
  rafState.callbacks.clear()
  callbacks.forEach((callback) => callback(performance.now()))
}

describe("TextSelectionPopover", () => {
  beforeEach(() => {
    rafState.callbacks.clear()
    rafState.nextId = 1
    messageApi.success.mockReset()
    messageApi.error.mockReset()
    tooltip.mockReset()
    translate.mockReset()
    touch.coarse = false
    useDocumentWorkspaceStore.getState().reset()

    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
      const id = rafState.nextId++
      rafState.callbacks.set(id, callback)
      return id
    })
    vi.stubGlobal("cancelAnimationFrame", (id: number) => {
      rafState.callbacks.delete(id)
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("hides the desktop popover again until the next position calculation is ready", () => {
    const onClose = vi.fn()
    const { container, rerender } = render(
      <TextSelectionPopover
        documentId={81}
        text="Selected text"
        position={{ x: 120, y: 80 }}
        onClose={onClose}
      />
    )

    const popover = container.querySelector("[data-selection-popover]") as HTMLDivElement

    act(() => {
      flushAnimationFrames()
    })

    expect(popover.style.opacity).toBe("1")
    expect(popover.style.pointerEvents).toBe("auto")

    rerender(
      <TextSelectionPopover
        documentId={81}
        text="Selected text"
        position={{ x: 260, y: 140 }}
        onClose={onClose}
      />
    )

    expect(popover.style.opacity).toBe("0")
    expect(popover.style.pointerEvents).toBe("none")

    act(() => {
      flushAnimationFrames()
    })

    expect(popover.style.opacity).toBe("1")
    expect(popover.style.pointerEvents).toBe("auto")
    expect(popover.style.left).toBe("260px")
    expect(popover.style.top).toBe("140px")
  })

  it("preserves the source in the portal and queues Ask AI for the document before Chat mounts", () => {
    const onClose = vi.fn()
    useDocumentWorkspaceStore.setState({
      workspaceAuthority: "fixture-owner", activeDocumentId: 81, activeRightTab: "annotations"
    })
    render(<TextSelectionPopover documentId={81} text="Selected passage" position={{ x: 100, y: 50 }} onClose={onClose} />)
    fireEvent.click(screen.getByRole("button", { name: "Ask AI" }))
    const explain = screen.getByRole("button", { name: "Explain" })
    expect(explain.closest(".document-selection-menu")).toBeTruthy()
    fireEvent.pointerDown(explain)
    expect(onClose).not.toHaveBeenCalled()
    fireEvent.click(explain)
    expect(useDocumentWorkspaceStore.getState()).toMatchObject({
      activeRightTab: "chat",
      pendingSelectionPrompt: { documentId: 81, prompt: 'Explain this passage: "Selected passage"' }
    })
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it("labels all five controls on both mouse hover and keyboard focus", () => {
    render(<TextSelectionPopover documentId={81} text="Selected text" position={{ x: 120, y: 80 }} onClose={vi.fn()} />)
    act(flushAnimationFrames)
    for (const label of ["Copy", "Highlight", "Translate", "Ask AI", "Listen"]) {
      expect(screen.getByRole("button", { name: label })).toBeVisible()
      expect(tooltip).toHaveBeenCalledWith(label, ["hover", "focus"])
    }
  })

  it("keeps action labels visible on a coarse pointer device", () => {
    touch.coarse = true
    render(<TextSelectionPopover documentId={81} text="Selected text" position={{ x: 120, y: 80 }} onClose={vi.fn()} />)
    act(flushAnimationFrames)
    expect(screen.getByText("Highlight", { exact: true })).toBeVisible()
    expect(screen.getByText("Ask AI", { exact: true })).toBeVisible()
  })

  it.each([
    { location: 4, source: { pdfPage: 4 } },
    { location: "epubcfi(/6/2!/4/1:0)", source: { epubCfi: "epubcfi(/6/2!/4/1:0)" } },
  ])("saves the selected location $location from the highlight menu", ({ location, source }) => {
    useDocumentWorkspaceStore.setState({ workspaceAuthority: "fixture-owner", activeDocumentId: 81, currentPage: 1 })
    render(<TextSelectionPopover documentId={81} text="Selected passage" position={{ x: 120, y: 80 }} onClose={vi.fn()} {...source} />)
    fireEvent.click(screen.getByRole("button", { name: "Highlight" }))
    fireEvent.click(screen.getByRole("button", { name: "Yellow" }))
    expect(useDocumentWorkspaceStore.getState().pendingAnnotations).toEqual([
      expect.objectContaining({ documentId: 81, location, text: "Selected passage", color: "yellow" })
    ])
  })

  it("closes the menu before dismissing the toolbar and restores keyboard focus", () => {
    const onClose = vi.fn()
    render(<TextSelectionPopover documentId={81} text="Selected text" position={{ x: 120, y: 80 }} onClose={onClose} />)
    const ask = screen.getByRole("button", { name: "Ask AI" })
    fireEvent.click(ask)
    fireEvent.keyDown(screen.getByRole("button", { name: "Explain" }), { key: "Escape" })
    expect(ask).toHaveFocus()
    expect(onClose).not.toHaveBeenCalled()
    fireEvent.keyDown(ask, { key: "Escape" })
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it("keeps the most recent translation and preserves the selected passage", async () => {
    useDocumentWorkspaceStore.setState({ workspaceAuthority: "fixture-owner", activeDocumentId: 81 })
    let finishEnglish!: (value: { translated_text: string }) => void
    translate.mockImplementationOnce(() => new Promise(resolve => { finishEnglish = resolve }))
      .mockResolvedValueOnce({ translated_text: "Dernière traduction" })
    const onClose = vi.fn()
    render(<TextSelectionPopover documentId={81} text="Selected passage" position={{ x: 120, y: 80 }} onClose={onClose} />)
    fireEvent.click(screen.getByRole("button", { name: "Translate" }))
    const dialog = screen.getByRole("dialog", { name: "Translation" })
    fireEvent.pointerDown(dialog)
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "French" } })
    expect(translate.mock.calls[0]?.[0].signal.aborted).toBe(true)
    expect(await screen.findByText("Dernière traduction")).toBeVisible()
    await act(async () => finishEnglish({ translated_text: "Stale English answer" }))
    expect(screen.queryByText("Stale English answer")).not.toBeInTheDocument()
    expect(screen.getByText("Selected passage")).toBeVisible()
    expect(onClose).not.toHaveBeenCalled()
  })

  it("drops a selection action if its captured account changed", () => {
    useDocumentWorkspaceStore.setState({ workspaceAuthority: "alice", activeDocumentId: 81, activeRightTab: "annotations" })
    render(<TextSelectionPopover documentId={81} text="Selected text" position={{ x: 120, y: 80 }} onClose={vi.fn()} />)
    fireEvent.click(screen.getByRole("button", { name: "Ask AI" }))
    act(() => useDocumentWorkspaceStore.setState({ workspaceAuthority: "bob" }))
    fireEvent.click(screen.getByRole("button", { name: "Explain" }))
    expect(useDocumentWorkspaceStore.getState().pendingSelectionPrompt).toBeNull()
  })
})
