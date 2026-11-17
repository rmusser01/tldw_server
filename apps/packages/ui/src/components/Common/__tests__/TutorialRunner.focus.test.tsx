import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { MemoryRouter, useNavigate } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { CallBackProps } from "react-joyride"
import { TutorialRunner } from "../TutorialRunner"
import { useStartTutorial } from "@/hooks/useStartTutorial"
import { useComposerEvents } from "@/hooks/useComposerEvents"
import { useStoreMessageOption } from "@/store/option"
import { useTutorialStore } from "@/store/tutorials"
import {
  HistorySelectionContext,
  type HistorySelectionController
} from "@/hooks/chat/useHistorySelection"

// Real runtime store, launch/source hook and composer registry; Joyride control
// rendering and H1/account leases are declared doubles in these ownership tests.
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_key: string, fallback: string) => fallback })
}))
vi.mock("react-joyride", () => ({
  default: function Controls({
    callback
  }: {
    callback: (data: CallBackProps) => void
  }) {
    const close = React.useRef<HTMLButtonElement>(null)
    React.useLayoutEffect(() => {
      close.current?.focus()
    }, [])
    const action = (kind: string) =>
      callback({
        action: kind === "close" ? "close" : "next",
        status:
          kind === "finish"
            ? "finished"
            : kind === "skip"
              ? "skipped"
              : "running",
        type: "step:after",
        index: 0
      } as CallBackProps)
    return (
      <div
        id="react-joyride-portal"
        role="dialog"
        aria-label="Declared tour controls">
        <button ref={close} onClick={() => action("close")}>
          Close
        </button>
        <button onClick={() => action("skip")}>Skip tour</button>
        <button onClick={() => action("finish")}>Finish</button>
      </div>
    )
  },
  STATUS: { FINISHED: "finished", SKIPPED: "skipped" },
  EVENTS: { STEP_AFTER: "step:after", TARGET_NOT_FOUND: "target:not_found" },
  ACTIONS: { CLOSE: "close", NEXT: "next", PREV: "prev" }
}))
let frames: Map<number, FrameRequestCallback>
beforeEach(() => {
  frames = new Map()
  let id = 0
  vi.spyOn(globalThis, "requestAnimationFrame").mockImplementation(
    (callback) => {
      frames.set(++id, callback)
      return id
    }
  )
  vi.spyOn(globalThis, "cancelAnimationFrame").mockImplementation((key) => {
    frames.delete(key)
  })
  vi.spyOn(document, "visibilityState", "get").mockReturnValue("visible")
  useTutorialStore.getState().resetProgress()
  useStoreMessageOption.getState().startNewChatDraft()
})
afterEach(() => {
  useTutorialStore.getState().resetProgress()
  vi.restoreAllMocks()
})
const finishFrames = async () => {
  const callbacks = [...frames.values()]
  frames.clear()
  await act(async () => callbacks.forEach((callback) => callback(0)))
}
const mountTour = () => {
  let epoch = 1,
    lease = true
  let capturedStart!: ReturnType<typeof useStartTutorial>
  const selection = {
    owner: { kind: "local", validate_lease: () => lease },
    fence: () => {
      const original = epoch
      return () => original === epoch
    }
  } as unknown as HistorySelectionController
  function Fixture() {
    const start = useStartTutorial(),
      navigate = useNavigate(),
      input = React.useRef<HTMLTextAreaElement>(null)
    React.useLayoutEffect(() => {
      capturedStart = start
    }, [start])
    useComposerEvents({ onFocusComposer: () => input.current?.focus() })
    return (
      <>
        <span>
          <button onClick={() => start("playground-basics")}>
            Launch tour
          </button>
        </span>
        <textarea
          ref={input}
          aria-label="Exact composer"
          defaultValue={"Keep exact draft.  \nSecond line."}
        />
        <button onClick={() => navigate("/notes")}>Navigate</button>
        <TutorialRunner />
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
  const start = () => {
    const opener = screen.getByRole("button", { name: "Launch tour" })
    opener.focus()
    fireEvent.click(opener)
    return opener
  }
  return {
    mounted,
    start,
    callback: () => capturedStart,
    retire: () => {
      epoch++
    },
    expire: () => {
      lease = false
    }
  }
}
describe("tutorial focus ownership", () => {
  it.each(["New", "H1", "replacement"])(
    "synchronous %s during publication cannot acknowledge retired Start",
    (boundary) => {
      const probe = mountTour()
      let replaced = false,
        replacement: ReturnType<typeof useTutorialStore.getState>["activeRun"] =
          null
      const unsubscribe = useTutorialStore.subscribe((state) => {
        if (replaced || !state.activeRun) return
        replaced = true
        if (boundary === "New")
          useStoreMessageOption.getState().startNewChatDraft()
        else if (boundary === "H1") probe.retire()
        else {
          useTutorialStore.getState().startTutorial("playground-basics")
          replacement = useTutorialStore.getState().activeRun
        }
      })
      try {
        let accepted: boolean | undefined
        act(() => {
          accepted = probe.callback()("playground-basics")
        })
        expect(accepted).toBe(false)
        expect(useTutorialStore.getState().activeRun).toBe(replacement)
      } finally {
        unsubscribe()
      }
    }
  )

  it("a rendered launcher refuses a committed route change while the fresh launcher works", () => {
    const probe = mountTour(),
      original = probe.callback()
    fireEvent.click(screen.getByRole("button", { name: "Navigate" }))
    act(() => expect(original("playground-basics")).toBe(false))
    expect(useTutorialStore.getState().activeRun).toBeNull()
    act(() => expect(probe.callback()("playground-basics")).toBe(true))
    expect(useTutorialStore.getState().activeRun).not.toBeNull()
  })

  it.each(["New", "H1", "account", "unmount"])(
    "a retained rendered Start cannot adopt %s before commit",
    (boundary) => {
      const probe = mountTour(),
        start = probe.callback()
      if (boundary === "New")
        useStoreMessageOption.getState().startNewChatDraft()
      else if (boundary === "H1") probe.retire()
      else if (boundary === "account") probe.expire()
      else probe.mounted.unmount()
      act(() => start("playground-basics"))
      expect(useTutorialStore.getState().activeRun).toBeNull()
    }
  )

  it.each(["Close", "Skip tour", "Finish"])(
    "%s returns the current removed control to its exact opener after commit",
    async (action) => {
      const probe = mountTour(),
        opener = probe.start()
      const button = screen.getByRole("button", { name: action })
      button.focus()
      fireEvent.click(button)
      expect(button.isConnected).toBe(false)
      await finishFrames()
      expect(opener).toHaveFocus()
      expect(
        screen.getByRole("textbox", { name: "Exact composer" })
      ).toHaveValue("Keep exact draft.  \nSecond line.")
      expect(useTutorialStore.getState().completedTutorials).toEqual(
        action === "Finish" ? ["playground-basics"] : []
      )
    }
  )

  it("returns to the registered composer when the original visible opener was removed", async () => {
    const probe = mountTour(),
      opener = probe.start()
    opener.remove()
    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    await finishFrames()
    expect(
      screen.getByRole("textbox", { name: "Exact composer" })
    ).toHaveFocus()
  })

  it.each([
    "focus ABA",
    "pointer",
    "input",
    "New",
    "H1",
    "account",
    "route",
    "restart",
    "dialog",
    "hidden",
    "unmount"
  ])("%s permanently cancels a queued return", async (boundary) => {
    const probe = mountTour(),
      opener = probe.start()
    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    const input = screen.getByRole("textbox", { name: "Exact composer" })
    const focus = vi.spyOn(opener, "focus")
    if (boundary === "focus ABA") {
      input.focus()
      input.blur()
    } else if (boundary === "pointer") fireEvent.pointerDown(input)
    else if (boundary === "input")
      fireEvent.input(input, { target: { value: "New typed input" } })
    else if (boundary === "New")
      useStoreMessageOption.getState().startNewChatDraft()
    else if (boundary === "H1") probe.retire()
    else if (boundary === "account") probe.expire()
    else if (boundary === "route")
      fireEvent.click(screen.getByRole("button", { name: "Navigate" }))
    else if (boundary === "restart")
      act(() => {
        useTutorialStore.getState().startTutorial("playground-basics")
        useTutorialStore.getState().endTutorial()
      })
    else if (boundary === "dialog") {
      const dialog = document.createElement("div")
      dialog.setAttribute("role", "dialog")
      dialog.textContent = "New dialog"
      document.body.append(dialog)
    } else if (boundary === "hidden")
      vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden")
    else probe.mounted.unmount()
    await finishFrames()
    expect(focus).not.toHaveBeenCalled()
    document.querySelectorAll('[role="dialog"]').forEach((dialog) => {
      if (dialog.textContent === "New dialog") dialog.remove()
    })
  })

  it("typing in a spotlight input cancels focus return while Finish still records completion", async () => {
    const probe = mountTour(),
      opener = probe.start()
    const input = screen.getByRole("textbox", { name: "Exact composer" })
    input.focus()
    fireEvent.input(input, { target: { value: "Tour exercise" } })
    const finish = screen.getByRole("button", { name: "Finish" })
    finish.focus()
    fireEvent.click(finish)
    const focus = vi.spyOn(opener, "focus")
    await finishFrames()
    expect(focus).not.toHaveBeenCalled()
    expect(useTutorialStore.getState().completedTutorials).toEqual([
      "playground-basics"
    ])
  })

  it("internal end without an explicit exit never restores focus", async () => {
    const probe = mountTour(),
      opener = probe.start()
    act(() => useTutorialStore.getState().endTutorial())
    const focus = vi.spyOn(opener, "focus")
    await finishFrames()
    expect(focus).not.toHaveBeenCalled()
  })
})
