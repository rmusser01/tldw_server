import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useExplicitStopFocus } from "../useExplicitStopFocus"
import { useComposerEvents } from "../useComposerEvents"
import { useStoreMessageOption } from "@/store/option"
import { captureChatDraftSource } from "@/hooks/chat/chat-draft-transition"
import { PageAssistContext } from "@/context"
import {
  HistorySelectionContext,
  type HistorySelectionController
} from "@/hooks/chat/useHistorySelection"
import { ChatComposer } from "@/components/Chat/composer/ChatComposer"
import { PlaygroundSendControl } from "@/components/Option/Playground/PlaygroundSendControl"

// Actual Stop wrapper, shared registry, global draft/activity store and rendered
// controls; H1/account predicates and abort settlement are declared boundaries.
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
  useStoreMessageOption.getState().startNewChatDraft()
  useStoreMessageOption.setState({
    isLoading: true,
    streaming: true,
    isProcessing: true
  })
})
afterEach(() => {
  useStoreMessageOption.setState({
    isLoading: false,
    streaming: false,
    isProcessing: false
  })
  vi.restoreAllMocks()
})

const mountStop = (
  kind:
    | "removed"
    | "retained"
    | "send-control"
    | "v1"
    | "v3"
    | "v5" = "removed",
  density: "desktop" | "compact" = "desktop"
) => {
  let epoch = 1,
    lease = true,
    inputRevision = 1
  let controller: AbortController | null = new AbortController()
  let currentStop!: ReturnType<typeof useExplicitStopFocus>
  const focusRequests = vi.fn()
  const idle = () => {
    controller = null
    useStoreMessageOption.setState({
      isLoading: false,
      streaming: false,
      isProcessing: false
    })
  }
  const rawStop = vi.fn(() => {
    controller?.abort()
    idle()
  })
  const owner = {
    kind: "local" as const,
    profile_id: "profile",
    owner_key: "owner",
    conversation_id: "chat",
    validate_lease: () => lease
  }
  const selection = {
    owner,
    fence: () => {
      const captured = epoch
      return () => captured === epoch
    }
  } as unknown as HistorySelectionController
  const context = {
    controller,
    getController: () => controller,
    setController: vi.fn(),
    embeddingController: null,
    setEmbeddingController: vi.fn()
  }
  function Fixture() {
    const busy = useStoreMessageOption(
      (state) => state.isLoading || state.streaming || state.isProcessing
    )
    const stop = useExplicitStopFocus(rawStop, () => {
      const revision = inputRevision,
        source = captureChatDraftSource()
      return () => source.isCurrent() && revision === inputRevision
    })
    React.useLayoutEffect(() => {
      currentStop = stop
    }, [stop])
    const input = React.useRef<HTMLTextAreaElement>(null)
    const onFocusComposer = React.useCallback(() => {
      focusRequests()
      input.current?.focus()
    }, [])
    useComposerEvents({ onFocusComposer })
    return (
      <>
        <textarea
          ref={input}
          aria-label="Stop composer"
          defaultValue={"Exact stopped draft.  \nSecond line."}
        />
        <button>Other focus</button>
        {kind.startsWith("v") ? (
          <ChatComposer
            variant={kind as "v1" | "v3" | "v5"}
            density={density}
            message=""
            onMessageChange={vi.fn()}
            onSend={vi.fn()}
            sending={busy}
            stopStreaming={stop}
          />
        ) : kind === "send-control" ? (
          <PlaygroundSendControl
            isProMode={false}
            isMobileViewport={density === "compact"}
            isSending={busy}
            isConnectionReady
            sendWhenEnter
            onSendWhenEnterChange={vi.fn()}
            sendLabel="Send"
            compareNeedsMoreModels={false}
            onStopStreaming={stop}
            onStopListening={vi.fn()}
            onSubmitForm={vi.fn()}
            sendMenuOpen={false}
            onSendMenuChange={vi.fn()}
            t={(_key: string, fallback?: string) => fallback ?? _key}
          />
        ) : kind === "retained" ? (
          <button onClick={stop}>{busy ? "Stop" : "Send"}</button>
        ) : busy ? (
          <button onClick={stop}>Stop</button>
        ) : null}
      </>
    )
  }
  const element = () => (
    <PageAssistContext.Provider value={context}>
      <HistorySelectionContext.Provider value={selection}>
        <Fixture />
      </HistorySelectionContext.Provider>
    </PageAssistContext.Provider>
  )
  const mounted = render(element())
  return {
    mounted,
    rawStop,
    focusRequests,
    idle,
    stop: () => currentStop,
    rerender: () => mounted.rerender(element()),
    retire: () => {
      epoch += 1
    },
    expire: () => {
      lease = false
    },
    edit: () => {
      inputRevision += 1
    },
    replaceController: () => {
      controller = new AbortController()
    },
    startAgain: () => {
      controller = new AbortController()
      useStoreMessageOption.setState({
        isLoading: true,
        streaming: true,
        isProcessing: true
      })
    }
  }
}
const finishFrames = async () => {
  const callbacks = [...frames.values()]
  frames.clear()
  await act(async () => {
    callbacks.forEach((callback) => callback(0))
  })
}
const stoppedButton = async (kind: string) =>
  kind === "send-control"
    ? screen.getByRole("button", { name: "tooltip.stopStreaming" })
    : kind.startsWith("v")
      ? screen.findByRole("button", { name: "Stop" })
      : screen.getByRole("button", { name: "Stop" })

describe("explicit Stop focus", () => {
  it.each(["removed", "send-control"] as const)(
    "returns a focused removed %s control to the exact composer once after idle",
    async (kind) => {
      const probe = mountStop(kind)
      const button = await stoppedButton(kind)
      button.focus()
      await act(async () => {
        fireEvent.click(button)
      })
      expect(probe.rawStop).toHaveBeenCalledOnce()
      expect(button.isConnected).toBe(false)
      await finishFrames()
      expect(
        screen.getByRole("textbox", { name: "Stop composer" })
      ).toHaveFocus()
      expect(
        screen.getByRole("textbox", { name: "Stop composer" })
      ).toHaveValue("Exact stopped draft.  \nSecond line.")
      expect(probe.focusRequests).toHaveBeenCalledOnce()
    }
  )

  it.each(["desktop", "compact"] as const)(
    "keeps focus on each actual Stop→Send variant in %s density",
    async (density) => {
      for (const kind of ["v1", "v3", "v5"] as const) {
        useStoreMessageOption.setState({
          isLoading: true,
          streaming: true,
          isProcessing: true
        })
        const probe = mountStop(kind, density)
        const button = await stoppedButton(kind)
        button.focus()
        await act(async () => {
          fireEvent.click(button)
        })
        await waitFor(() => expect(button).toHaveAccessibleName("Send"))
        await finishFrames()
        expect(button).toHaveFocus()
        expect(probe.rawStop).toHaveBeenCalledOnce()
        expect(probe.focusRequests).not.toHaveBeenCalled()
        probe.mounted.unmount()
      }
    }
  )

  it("keeps V1 Control+Enter Stop focus in its textarea", async () => {
    const probe = mountStop("v1")
    const input = await screen.findByRole("textbox", { name: "Message" })
    input.focus()
    await act(async () => {
      fireEvent.keyDown(input, { key: "Enter", ctrlKey: true })
    })
    await finishFrames()
    expect(probe.rawStop).toHaveBeenCalledOnce()
    expect(input).toHaveFocus()
    expect(probe.focusRequests).not.toHaveBeenCalled()
  })

  it("internal raw cancellation remains focus-neutral", async () => {
    const probe = mountStop()
    screen.getByRole("button", { name: "Stop" }).focus()
    await act(async () => {
      probe.rawStop()
    })
    await finishFrames()
    expect(probe.focusRequests).not.toHaveBeenCalled()
  })

  it.each(["New", "H1", "owner", "new controller"])(
    "does not let an old rendered Stop abort a replacement %s before a commit",
    async (boundary) => {
      const probe = mountStop()
      const old = probe.stop()
      const button = screen.getByRole("button", { name: "Stop" })
      if (boundary === "New")
        useStoreMessageOption.getState().startNewChatDraft()
      else if (boundary === "H1") probe.retire()
      else if (boundary === "owner") probe.expire()
      else probe.replaceController()
      act(() => {
        old({ currentTarget: button })
      })
      expect(probe.rawStop).not.toHaveBeenCalled()
      expect(probe.focusRequests).not.toHaveBeenCalled()
    }
  )

  it.each([
    "focus ABA",
    "pointer",
    "input",
    "programmatic input",
    "New",
    "H1",
    "owner",
    "new controller",
    "editor",
    "dialog",
    "hidden",
    "unmount"
  ])("does not steal newer focus after Stop (%s)", async (boundary) => {
    const probe = mountStop()
    const button = screen.getByRole("button", { name: "Stop" })
    const other = screen.getByRole("button", { name: "Other focus" })
    button.focus()
    await act(async () => {
      fireEvent.click(button)
    })
    let extra: HTMLElement | undefined
    try {
      if (boundary === "focus ABA") {
        other.focus()
        other.blur()
      } else if (boundary === "pointer") fireEvent.pointerDown(other)
      else if (boundary === "input")
        fireEvent.input(
          screen.getByRole("textbox", { name: "Stop composer" }),
          { target: { value: "Newer exact input.  " } }
        )
      else if (boundary === "programmatic input") probe.edit()
      else if (boundary === "New")
        useStoreMessageOption.getState().startNewChatDraft()
      else if (boundary === "H1") probe.retire()
      else if (boundary === "owner") probe.expire()
      else if (boundary === "new controller") probe.replaceController()
      else if (boundary === "editor") {
        extra = document.createElement("form")
        extra.dataset.chatMessageEditor = ""
        document.body.append(extra)
      } else if (boundary === "dialog") {
        extra = document.createElement("div")
        extra.setAttribute("role", "dialog")
        document.body.append(extra)
      } else if (boundary === "hidden")
        vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden")
      else probe.mounted.unmount()
      await finishFrames()
      expect(probe.focusRequests).not.toHaveBeenCalled()
    } finally {
      extra?.remove()
    }
  })

  it("an old canceled frame cannot focus or cancel a newer Stop attempt", async () => {
    const probe = mountStop()
    let button = screen.getByRole("button", { name: "Stop" })
    button.focus()
    await act(async () => {
      fireEvent.click(button)
    })
    const oldFrame = [...frames.values()][0]
    act(() => {
      probe.startAgain()
    })
    button = screen.getByRole("button", { name: "Stop" })
    button.focus()
    await act(async () => {
      fireEvent.click(button)
    })
    expect(oldFrame).toBeTypeOf("function")
    await act(async () => {
      oldFrame(0)
    })
    expect(probe.focusRequests).not.toHaveBeenCalled()
    await finishFrames()
    expect(probe.focusRequests).toHaveBeenCalledOnce()
  })
})
