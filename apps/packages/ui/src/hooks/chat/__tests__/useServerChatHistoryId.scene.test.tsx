// @vitest-environment jsdom
import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import type { TFunction } from "i18next"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { createDefaultActorSettings } from "@/types/actor"
import { saveActorSettingsForChat } from "@/services/actor-settings"
import { captureNormalChatPreparation } from "@/hooks/chat-modes/normalChatMode"
import { captureChatDraftLoad } from "../chat-draft-transition"
import {
  captureActorSceneSend,
  captureCurrentActorSceneTarget
} from "../actor-scene-source"
import {
  HistorySelectionContext,
  type HistorySelectionController,
  type HistoryLoadReceipt
} from "../useHistorySelection"
import { useServerChatHistoryId } from "../useServerChatHistoryId"

const link = vi.hoisted(() =>
  vi.fn<typeof import("@/db/dexie/server-chat-mirror").linkServerChatMirror>()
)
vi.mock("@/db/dexie/server-chat-mirror", async (original) => ({
  ...(await original<typeof import("@/db/dexie/server-chat-mirror")>()),
  linkServerChatMirror: link
}))
const requestScope = {
  config: {
    serverUrl: "https://native-mirror-scene.example",
    authMode: "single-user" as const
  },
  userId: 1
}
const t = ((key: string) => key) as TFunction
const prepare = async () => {
  let accountCurrent = true,
    epoch = 0
  const owner = {
    kind: "native" as const,
    conversation_id: "native-source",
    owner_key: "original-native-owner",
    request_scope: requestScope,
    scope: { type: "global" as const },
    validate_lease: () => accountCurrent
  }
  const view = {
    view_session_id: "original-view",
    owner_key: owner.owner_key,
    conversation_id: owner.conversation_id,
    interpretation: { kind: "parent_graph_v1" as const },
    cursor: { kind: "empty" as const },
    selection_revision: 0
  }
  const receipt = { owner, view } as HistoryLoadReceipt
  expect(captureChatDraftLoad("switch")(receipt)).toBe(true)
  useStoreMessageOption.getState().setServerChatId(owner.conversation_id)
  const controller = {
    getCurrent: () => ({ status: "ready", ...receipt }),
    fence: () => {
      const captured = epoch
      return () => epoch === captured
    },
    draftAccount: { requestScope, isCurrent: () => accountCurrent }
  } as unknown as HistorySelectionController
  const target = captureCurrentActorSceneTarget(controller)!
  const settings = {
    ...createDefaultActorSettings(),
    notes: "Original native scene"
  }
  expect(
    await saveActorSettingsForChat({
      target,
      historyId: null,
      serverChatId: owner.conversation_id,
      expectedRecord: null,
      settings
    })
  ).toBe(true)
  const sceneSend = await captureActorSceneSend(controller)!.read()
  const preparation = captureNormalChatPreparation(
    {
      selectedModel: "fixture-model",
      historyId: null,
      serverChatId: owner.conversation_id
    },
    sceneSend.isCurrent
  )
  const hook = renderHook(
    () => {
      const historyId = useStoreMessageOption((state) => state.historyId)
      return useServerChatHistoryId({
        serverChatId: owner.conversation_id,
        historyId,
        setHistoryId: useStoreMessageOption.getState().setHistoryId,
        temporaryChat: false,
        t
      })
    },
    {
      wrapper: ({ children }) => (
        <HistorySelectionContext.Provider value={controller}>
          {children}
        </HistorySelectionContext.Provider>
      )
    }
  )
  const ensure = () =>
    hook.result.current.ensureServerChatHistoryId(
      owner.conversation_id,
      "Original title",
      undefined,
      undefined,
      sceneSend
    )
  return {
    ...hook,
    sceneSend,
    preparation,
    ensure,
    retireAccount: () => {
      accountCurrent = false
    },
    retireView: () => {
      epoch++
    }
  }
}
beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  Object.defineProperty(navigator, "locks", {
    configurable: true,
    value: {
      request: async (_key: string, work: () => Promise<unknown>) => work()
    }
  })
  useStoreMessageOption.getState().startNewChatDraft()
  useStoreMessageOption.setState({ historyId: null, serverChatId: null })
  link.mockResolvedValue("certified-native-mirror")
})

it("continues the original scene and normal preparation through the actual certified mirror publication", async () => {
  const prepared = await prepare()
  await act(async () => {
    expect(await prepared.ensure()).toBe("certified-native-mirror")
  })
  expect(useStoreMessageOption.getState().historyId).toBe(
    "certified-native-mirror"
  )
  expect(prepared.sceneSend.isCurrent()).toBe(true)
  expect(prepared.preparation.draftIsCurrent()).toBe(true)
  expect(prepared.sceneSend.scene.settings.notes).toBe("Original native scene")
})
it("accepts the identical concurrent mirror only after its original scoped link certifies it", async () => {
  let release!: (id: string) => void
  link.mockReturnValueOnce(
    new Promise((resolve) => {
      release = resolve
    })
  )
  const prepared = await prepare()
  const pending = prepared.ensure()
  await waitFor(() => expect(link).toHaveBeenCalledOnce())
  act(() =>
    useStoreMessageOption
      .getState()
      .setHistoryId("certified-native-mirror", { preserveServerChatId: true })
  )
  expect(prepared.sceneSend.isCurrent()).toBe(false)
  await act(async () => {
    release("certified-native-mirror")
    expect(await pending).toBe("certified-native-mirror")
  })
  expect(prepared.sceneSend.isCurrent()).toBe(true)
  expect(prepared.preparation.draftIsCurrent()).toBe(true)
})
it.each(["New", "same-ID restore", "H1 view", "account"])(
  "refuses delayed mirror publication after %s retires the original command",
  async (boundary) => {
    let release!: (id: string) => void
    link.mockReturnValueOnce(
      new Promise((resolve) => {
        release = resolve
      })
    )
    const prepared = await prepare()
    const pending = prepared.ensure().catch((error) => error)
    await waitFor(() => expect(link).toHaveBeenCalledOnce())
    act(() => {
      if (boundary === "New")
        useStoreMessageOption.getState().startNewChatDraft()
      if (boundary === "same-ID restore")
        usePlaygroundSessionStore.setState((state) => ({
          restoreRevision: state.restoreRevision + 1
        }))
      if (boundary === "H1 view") prepared.retireView()
      if (boundary === "account") prepared.retireAccount()
    })
    await act(async () => {
      release("certified-native-mirror")
    })
    expect(await pending).toMatchObject({ status: 412 })
    expect(useStoreMessageOption.getState().historyId).toBe(null)
    expect(prepared.sceneSend.isCurrent()).toBe(false)
  }
)
it("refuses a synchronous New during the actual mirror setter instead of returning an accepted link", async () => {
  const prepared = await prepare()
  const unsubscribe = useStoreMessageOption.subscribe((state) => {
    if (state.historyId === "certified-native-mirror") {
      unsubscribe()
      state.startNewChatDraft()
      useStoreMessageOption.setState({
        historyId: "replacement-history",
        serverChatId: null
      })
    }
  })
  let result: unknown
  await act(async () => {
    result = await prepared.ensure().catch((error) => error)
  })
  unsubscribe()
  expect(result).toMatchObject({ status: 412 })
  expect(useStoreMessageOption.getState().historyId).toBe("replacement-history")
  expect(prepared.sceneSend.isCurrent()).toBe(false)
})

it("never replaces a foreign alias with the original scoped mirror acknowledgement", async () => {
  let release!: (id: string) => void
  link.mockReturnValueOnce(
    new Promise((resolve) => {
      release = resolve
    })
  )
  const prepared = await prepare()
  const pending = prepared.ensure().catch((error) => error)
  await waitFor(() => expect(link).toHaveBeenCalledOnce())
  act(() =>
    useStoreMessageOption
      .getState()
      .setHistoryId("foreign-mirror", { preserveServerChatId: true })
  )
  await act(async () => {
    release("certified-native-mirror")
  })
  expect(await pending).toMatchObject({ status: 412 })
  expect(useStoreMessageOption.getState().historyId).toBe("foreign-mirror")
  expect(prepared.sceneSend.isCurrent()).toBe(false)
})

it("a rejected scoped mirror link grants no alias publication", async () => {
  const failure = new Error("Scoped mirror write rejected")
  link.mockRejectedValueOnce(failure)
  const prepared = await prepare()
  await expect(prepared.ensure()).rejects.toBe(failure)
  expect(useStoreMessageOption.getState().historyId).toBe(null)
  expect(prepared.sceneSend.isCurrent()).toBe(true)
  expect(prepared.preparation.draftIsCurrent()).toBe(true)
})
