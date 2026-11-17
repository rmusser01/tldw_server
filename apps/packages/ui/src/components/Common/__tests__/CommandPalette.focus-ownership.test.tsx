import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { MemoryRouter, useNavigate } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { CommandPalette, type CommandPaletteProps } from "../CommandPalette"
import { CommandPaletteHost } from "../CommandPaletteHost"
import { useStoreMessageOption } from "@/store/option"
import {
  HistorySelectionContext,
  type HistorySelectionController
} from "@/hooks/chat/useHistorySelection"

// Real Host, lazy boundary, Palette, source store and router; H1/account predicate
// doubles and manually held chunk/timer delivery qualify their stated boundaries.
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_key: string, fallback: string) => fallback })
}))
vi.mock("@/hooks/keyboard/useShortcutConfig", () => ({
  useShortcutConfig: () => ({ shortcuts: {} })
}))
vi.mock("@/components/Option/Prompt/usePromptPaletteCommands", () => ({
  usePromptPaletteCommands: () => []
}))
beforeEach(() => {
  vi.useFakeTimers()
  vi.spyOn(document, "visibilityState", "get").mockReturnValue("visible")
  HTMLElement.prototype.scrollIntoView = vi.fn()
  useStoreMessageOption.getState().startNewChatDraft()
})
afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})
const mountPalette = (host: boolean) => {
  let epoch = 1,
    lease = true
  const selection = {
    owner: { kind: "local", validate_lease: () => lease },
    fence: () => {
      const original = epoch
      return () => original === epoch
    }
  } as unknown as HistorySelectionController
  let reject!: (error: Error) => void
  const reloadPage = vi.fn()
  let release!: (value: {
    CommandPalette: React.ComponentType<CommandPaletteProps>
  }) => void
  const chunk = new Promise<{
    CommandPalette: React.ComponentType<CommandPaletteProps>
  }>((resolve, fail) => {
    release = resolve
    reject = fail
  })
  const loadPalette = vi.fn(() => chunk)
  function Fixture() {
    const navigate = useNavigate()
    return (
      <>
        <textarea
          aria-label="Original composer"
          defaultValue={"Exact palette draft.  \nSecond line."}
        />
        <input aria-label="New interaction" />
        <button onClick={() => navigate("/notes")}>Navigate</button>
        {host ? (
          <CommandPaletteHost
            loadPalette={loadPalette}
            reloadPage={reloadPage}
          />
        ) : (
          <CommandPalette />
        )}
      </>
    )
  }
  const mounted = render(
    <MemoryRouter initialEntries={["/chat"]}>
      <HistorySelectionContext.Provider value={selection}>
        <Fixture />
      </HistorySelectionContext.Provider>
    </MemoryRouter>
  )
  const open = () => {
    const opener = screen.getByRole("textbox", { name: "Original composer" })
    opener.focus()
    act(() =>
      window.dispatchEvent(new CustomEvent("tldw:open-command-palette"))
    )
    return opener
  }
  return {
    mounted,
    open,
    loadPalette,
    reloadPage,
    reject: async () => {
      await act(async () => reject(new Error("Offline chunk")))
    },
    release: async () => {
      await act(async () => release({ CommandPalette }))
    },
    retire: () => {
      epoch++
    },
    expire: () => {
      lease = false
    }
  }
}
const flush = () =>
  act(() => {
    vi.advanceTimersByTime(0)
  })
const close = () => fireEvent.keyDown(document, { key: "Escape" })

describe("palette explicit focus ownership", () => {
  it("the current loading Dismiss remains usable after outside focus without returning focus", () => {
    const probe = mountPalette(true),
      opener = probe.open()
    screen.getByRole("textbox", { name: "New interaction" }).focus()
    const focus = vi.spyOn(opener, "focus")
    const dismiss = screen.getByRole("button", { name: "Dismiss" })
    dismiss.focus()
    fireEvent.click(dismiss)
    flush()
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
    expect(focus).not.toHaveBeenCalled()
  })

  it("the current failed Reload remains usable after outside focus under the original source", async () => {
    const probe = mountPalette(true)
    probe.open()
    await probe.reject()
    screen.getByRole("textbox", { name: "New interaction" }).focus()
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true)
    const reload = screen.getByRole("button", { name: "Reload page" })
    reload.focus()
    fireEvent.click(reload)
    expect(confirm).toHaveBeenCalledOnce()
    expect(probe.reloadPage).toHaveBeenCalledOnce()
  })

  it.each(["New", "H1", "account", "route"])(
    "a current failed Reload still refuses retired %s source",
    async (boundary) => {
      const probe = mountPalette(true)
      probe.open()
      await probe.reject()
      const reload = screen.getByRole("button", { name: "Reload page" })
      const key = Object.keys(reload).find((name) =>
        name.startsWith("__reactProps$")
      )!
      const original = (
        reload as unknown as Record<string, { onClick: () => void }>
      )[key].onClick
      if (boundary === "New")
        act(() => useStoreMessageOption.getState().startNewChatDraft())
      else if (boundary === "H1") probe.retire()
      else if (boundary === "account") probe.expire()
      else fireEvent.click(screen.getByRole("button", { name: "Navigate" }))
      const confirm = vi.spyOn(window, "confirm").mockReturnValue(true)
      act(original)
      expect(confirm).not.toHaveBeenCalled()
      expect(probe.reloadPage).not.toHaveBeenCalled()
    }
  )

  it.each(["resolved", "reopened", "New"])(
    "a retained loading Dismiss cannot close %s commands",
    async (boundary) => {
      const probe = mountPalette(true)
      probe.open()
      const dismiss = screen.getByRole("button", { name: "Dismiss" })
      const key = Object.keys(dismiss).find((name) =>
        name.startsWith("__reactProps$")
      )!
      const original = (
        dismiss as unknown as Record<string, { onClick: () => void }>
      )[key].onClick
      if (boundary === "resolved") {
        await probe.release()
        flush()
      } else {
        if (boundary === "reopened") {
          dismiss.focus()
          fireEvent.click(dismiss)
          flush()
        } else act(() => useStoreMessageOption.getState().startNewChatDraft())
        probe.open()
      }
      act(original)
      expect(
        boundary === "resolved"
          ? screen.getByRole("dialog")
          : screen.getByRole("status")
      ).toBeInTheDocument()
      if (boundary !== "resolved") {
        await probe.release()
        flush()
        expect(screen.getByRole("dialog")).toBeInTheDocument()
      }
    }
  )

  it("a retained failed-load Reload cannot act on a reopened request", async () => {
    const probe = mountPalette(true)
    probe.open()
    await probe.reject()
    const reload = screen.getByRole("button", { name: "Reload page" })
    const key = Object.keys(reload).find((name) =>
      name.startsWith("__reactProps$")
    )!
    const original = (
      reload as unknown as Record<string, { onClick: () => void }>
    )[key].onClick
    const dismiss = screen.getByRole("button", { name: "Dismiss" })
    dismiss.focus()
    fireEvent.click(dismiss)
    flush()
    probe.open()
    await act(async () => Promise.resolve())
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true)
    act(original)
    expect(confirm).not.toHaveBeenCalled()
    expect(probe.reloadPage).not.toHaveBeenCalled()
    expect(screen.getByRole("alert")).toBeInTheDocument()
  })

  it("shows cold loading, returns an owned Dismiss and ignores late chunk delivery", async () => {
    const probe = mountPalette(true)
    expect(probe.loadPalette).not.toHaveBeenCalled()
    const opener = probe.open()
    expect(screen.getByRole("status")).toHaveTextContent("Loading commands")
    const dismiss = screen.getByRole("button", { name: "Dismiss" })
    dismiss.focus()
    fireEvent.click(dismiss)
    flush()
    expect(opener).toHaveFocus()
    await probe.release()
    flush()
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
    expect(probe.loadPalette).toHaveBeenCalledOnce()
  })

  it("retains the exact draft on chunk failure and requires confirmed reload", async () => {
    const probe = mountPalette(true),
      opener = probe.open()
    await probe.reject()
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Commands could not be loaded"
    )
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false)
    fireEvent.click(screen.getByRole("button", { name: "Reload page" }))
    expect(confirm).toHaveBeenCalledWith(
      expect.stringContaining("Unsaved edits may be lost")
    )
    expect(probe.reloadPage).not.toHaveBeenCalled()
    expect(opener).toHaveValue("Exact palette draft.  \nSecond line.")
    confirm.mockReturnValue(true)
    fireEvent.click(screen.getByRole("button", { name: "Reload page" }))
    expect(probe.reloadPage).toHaveBeenCalledOnce()
    expect(probe.loadPalette).toHaveBeenCalledOnce()
    const dismiss = screen.getByRole("button", { name: "Dismiss" })
    dismiss.focus()
    fireEvent.click(dismiss)
    flush()
    expect(opener).toHaveFocus()
  })

  it.each([false, true])(
    "current %s open focuses the palette and Escape returns exact original input",
    async (host) => {
      const probe = mountPalette(host),
        opener = probe.open()
      if (host) await probe.release()
      flush()
      expect(screen.getByRole("dialog")).toContainElement(
        document.activeElement as HTMLElement
      )
      close()
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
      flush()
      expect(opener).toHaveFocus()
      expect(opener).toHaveValue("Exact palette draft.  \nSecond line.")
    }
  )

  it.each([
    "focus ABA",
    "pointer",
    "input",
    "New",
    "H1",
    "account",
    "route",
    "unmount"
  ])(
    "cold chunk resolution cannot adopt a replacement %s",
    async (boundary) => {
      const probe = mountPalette(true),
        opener = probe.open()
      expect(probe.loadPalette).toHaveBeenCalledOnce()
      const other = screen.getByRole("textbox", { name: "New interaction" })
      if (boundary === "focus ABA") {
        other.focus()
        other.blur()
      } else if (boundary === "pointer") fireEvent.pointerDown(other)
      else if (boundary === "input")
        fireEvent.input(opener, { target: { value: "New input before load" } })
      else if (boundary === "New")
        useStoreMessageOption.getState().startNewChatDraft()
      else if (boundary === "H1") probe.retire()
      else if (boundary === "account") probe.expire()
      else if (boundary === "route")
        fireEvent.click(screen.getByRole("button", { name: "Navigate" }))
      else probe.mounted.unmount()
      const focus = vi.spyOn(opener, "focus")
      await probe.release()
      flush()
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
      expect(focus).not.toHaveBeenCalled()
    }
  )

  it.each([
    "focus ABA",
    "pointer",
    "input",
    "New",
    "H1",
    "account",
    "hidden",
    "disabled",
    "inert"
  ])("%s cancels a queued close return", async (boundary) => {
    const probe = mountPalette(false),
      opener = probe.open()
    flush()
    close()
    const other = screen.getByRole("textbox", { name: "New interaction" })
    const focus = vi.spyOn(opener, "focus")
    if (boundary === "focus ABA") {
      other.focus()
      other.blur()
    } else if (boundary === "pointer") fireEvent.pointerDown(other)
    else if (boundary === "input")
      fireEvent.input(other, { target: { value: "New input" } })
    else if (boundary === "New")
      useStoreMessageOption.getState().startNewChatDraft()
    else if (boundary === "H1") probe.retire()
    else if (boundary === "account") probe.expire()
    else if (boundary === "hidden") opener.style.display = "none"
    else if (boundary === "disabled") opener.setAttribute("disabled", "")
    else opener.setAttribute("inert", "")
    flush()
    expect(focus).not.toHaveBeenCalled()
  })

  it("an old opening input timer cannot take focus from a newer interaction", () => {
    const probe = mountPalette(false)
    probe.open()
    const other = screen.getByRole("textbox", { name: "New interaction" })
    other.focus()
    flush()
    expect(other).toHaveFocus()
  })
})
