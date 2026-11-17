// @vitest-environment jsdom
import type { ServerChatSummary } from "@/services/tldw/TldwApiClient"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { act, renderHook } from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  HistorySelectionContext,
  useHistorySelection
} from "../useHistorySelection"
import { useSelectServerChat } from "../useSelectServerChat"

const navigate = vi.hoisted(() => vi.fn())
vi.mock("react-router-dom", () => ({ useNavigate: () => navigate }))
vi.mock("@/services/chat-account-boundary", () => ({
  watchChatAccountChanges: () => () => undefined
}))
vi.mock("@/utils/settings-return", () => ({
  requestSettingsNavigation: () => true
}))
vi.mock("@/utils/cleanup-ant-overlays", () => ({
  cleanupAntOverlays: () => undefined
}))
vi.mock("@/utils/update-page-title", () => ({
  updatePageTitle: () => undefined
}))

const chat = {
  id: "saved",
  title: "Saved",
  scope_type: "global",
  version: 1
} as ServerChatSummary
beforeEach(() => {
  vi.clearAllMocks()
  useStoreMessageOption.getState().startNewChatDraft(true)
  useStoreMessageOption.setState({
    temporaryChat: true,
    historyId: "temp",
    serverChatId: null,
    messages: [{ role: "user", message: "Keep my draft" }]
  })
  vi.spyOn(window, "confirm").mockReturnValue(true)
})
describe("linked Open through actual stores and shared admission", () => {
  it.each(["New", "H1", "restore"])(
    "cannot adopt a %s published by the mode-change subscriber",
    (change) => {
      let selection!: ReturnType<typeof useHistorySelection>
      const Wrapper = ({ children }: React.PropsWithChildren) => {
        const current = useHistorySelection()
        React.useLayoutEffect(() => {
          selection = current
        }, [current])
        return (
          <HistorySelectionContext.Provider value={current}>
            {children}
          </HistorySelectionContext.Provider>
        )
      }
      const hook = renderHook(() => useSelectServerChat(), { wrapper: Wrapper })
      let retired = false
      const stop = useStoreMessageOption.subscribe((state) => {
        if (retired || state.temporaryChat) return
        retired = true
        if (change === "New") state.startNewChatDraft(false)
        else if (change === "H1") selection.beginLoad()
        else usePlaygroundSessionStore.getState().cancelPendingRestore()
      })
      try {
        act(() =>
          expect(hook.result.current(chat, { temporaryChat: false })).toBe(
            false
          )
        )
        expect(useStoreMessageOption.getState().messages[0]?.message).toBe(
          "Keep my draft"
        )
        expect(navigate).not.toHaveBeenCalled()
      } finally {
        stop()
      }
    }
  )
  it("accepts the admitted destination receipt after its source manager unmounts", () => {
    let managerCurrent = true
    const hook = renderHook(() => useSelectServerChat())
    act(() =>
      expect(
        hook.result.current(chat, {
          temporaryChat: false,
          isCurrent: () => managerCurrent
        })
      ).toBe(true)
    )
    const intent =
      usePlaygroundSessionStore.getState().serverChatSelectionIntent
    hook.unmount()
    managerCurrent = false
    act(() =>
      intent?.onLoaded?.({
        owner: {
          kind: "native",
          conversation_id: "saved",
          owner_key: "verified-account",
          request_scope: {} as never,
          validate_lease: () => true
        },
        view: {
          owner_key: "verified-account",
          conversation_id: "saved",
          view_session_id: "destination",
          selection_revision: 1,
          cursor: { kind: "empty" },
          interpretation: { kind: "parent_graph_v1" }
        }
      })
    )
    expect(useStoreMessageOption.getState().chatDraftAddress).toMatchObject({
      kind: "native",
      id: "saved",
      ownerKey: "verified-account"
    })
  })

  it("accepts the intentional Temporary-to-saved mode change and binds a load receipt", () => {
    const hook = renderHook(() => useSelectServerChat())
    act(() =>
      expect(hook.result.current(chat, { temporaryChat: false })).toBe(true)
    )
    expect(useStoreMessageOption.getState().temporaryChat).toBe(false)
    expect(useStoreMessageOption.getState().serverChatId).toBe("saved")
    expect(
      usePlaygroundSessionStore.getState().serverChatSelectionIntent
    ).toMatchObject({ chatId: "saved", onLoaded: expect.any(Function) })
  })
})


it("keeps character list metadata pending until the original scoped detail is loaded", () => {
  const hook = renderHook(() => useSelectServerChat())
  act(() =>
    expect(
      hook.result.current(
        {
          ...chat,
          assistant_kind: "character",
          assistant_id: "12",
          character_id: 12
        },
        { temporaryChat: false }
      )
    ).toBe(true)
  )
  expect(useStoreMessageOption.getState().serverChatMetaLoaded).toBe(false)
  hook.unmount()
})
