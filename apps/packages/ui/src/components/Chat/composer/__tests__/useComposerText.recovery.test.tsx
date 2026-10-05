import { readFileSync } from "node:fs"
import path from "node:path"
import ts from "typescript"
import {
  isChatSubmitSuccess,
  normalizeChatSubmitResult
} from "@/hooks/chat/chat-action-utils"
import React from "react"
import { act, cleanup, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { Storage as PlasmoStorage } from "@plasmohq/storage"
import { useStoreMessageOption } from "@/store/option"
import {
  conversationDraftKey,
  type ChatDraftAddress
} from "@/utils/chat-draft-address"
import { createLocalRegistryBucket } from "@/services/settings/local-bucket"
import { chatDraftOwnerKey } from "@/hooks/useChatDraftOwner"
import { COMPOSER_CONSTANTS } from "@/config/ui-constants"

const authority = vi.hoisted(() => ({ user: "alice" }))
const scope = () => ({
  userId: authority.user,
  config: { serverUrl: "http://chat.test", authMode: "multi-user" as const }
})
vi.mock(
  "@plasmohq/storage",
  () =>
    import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage")
)
vi.mock("@/services/service-prompts", async (original) => ({
  ...(await original<typeof import("@/services/service-prompts")>()),
  resolveServicePromptScope: async () => scope()
}))
import { useComposerText } from "../hooks/useComposerText"
const compactFormAst = ts.createSourceFile(
  "form.tsx",
  readFileSync(
    path.resolve(__dirname, "../../../Sidepanel/Chat/form.tsx"),
    "utf8"
  ),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX
)
let afterSend: ts.Expression | undefined
const findAfterSend = (node: ts.Node) => {
  if (
    ts.isPropertyAssignment(node) &&
    node.name.getText(compactFormAst) === "afterSend" &&
    node.initializer
      .getText(compactFormAst)
      .includes("clearDraft(promptAssistAttemptId)")
  )
    afterSend = node.initializer
  ts.forEachChild(node, findAfterSend)
}
findAfterSend(compactFormAst)
if (!afterSend)
  throw new Error("Missing actual compact send acknowledgement callback")
const afterSendCode = ts.transpileModule(
  `const afterSend = ${afterSend.getText(compactFormAst)};`,
  {
    compilerOptions: { target: ts.ScriptTarget.ES2022 }
  }
).outputText
const deployedAfterSend = (
  attempt: number,
  composer: ReturnType<typeof useComposerText>
) =>
  new Function(
    "promptAssistAttemptId",
    "markPromptAssistAttemptSaved",
    "clearDraft",
    "isChatSubmitSuccess",
    "normalizeChatSubmitResult",
    "clearSelectedDocuments",
    "setContextFiles",
    "setKnowledgeMentionActive",
    `${afterSendCode}; return afterSend;`
  )(
    attempt,
    composer.markPromptAssistAttemptSaved,
    composer.clearDraft,
    isChatSubmitSuccess,
    normalizeChatSubmitResult,
    vi.fn(),
    vi.fn(),
    vi.fn()
  ) as (result: undefined) => void

type StoredDraft = {
  value: { content: string; metadata: Record<string, unknown> }
}
const root = "tldw:playgroundChatDraft"
const settle = async () => {
  await act(async () => {
    await vi.dynamicImportSettled()
  })
  await act(async () => {
    await vi.advanceTimersByTimeAsync(
      COMPOSER_CONSTANTS.DRAFT_SAVE_DEBOUNCE_MS + 1
    )
  })
}
const mount = (surface = root, tabs = false) =>
  renderHook(
    ({ surface }) => {
      const address = useStoreMessageOption((state) => state.chatDraftAddress)
      const transition = useStoreMessageOption(
        (state) => state.chatDraftTransition
      )
      return useComposerText({
        draftKey: address ? conversationDraftKey(surface, address) : surface,
        draftAddress: address ?? undefined,
        draftEnabled: address !== null,
        bindFromDraftKey:
          transition?.kind === "bind" && transition.sourceAddress
            ? conversationDraftKey(surface, transition.sourceAddress)
            : undefined,
        recoverySurface: tabs ? "tldw:sidepanelChatDraft" : root,
        recoverySurfaceTabs: tabs,
        tabScopedDraft: true,
        textareaRef: React.createRef<HTMLTextAreaElement>(),
        getDraftMetadata: () => ({
          collapsedRange: { start: 1, end: 5 },
          wasExpanded: false
        })
      })
    },
    { initialProps: { surface } }
  )
const fresh = (id: string, temporary = false): ChatDraftAddress => ({
  kind: "fresh",
  id,
  temporary,
  scope: { type: "global" }
})

describe("retained fresh composer recovery", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    localStorage.clear()
    sessionStorage.clear()
    authority.user = "alice"
    useStoreMessageOption.getState().startNewChatDraft(false)
  })
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it.each([false, true])(
    "makes departed nonempty fresh text reachable with its actual mode before debounce (Temporary %s)",
    async (temporary) => {
      useStoreMessageOption.getState().setTemporaryChat(temporary)
      const oldAddress = useStoreMessageOption.getState().chatDraftAddress!
      const view = mount()
      await settle()
      act(() => view.result.current.setMessageValue("UNSENT SOURCE"))
      act(() => useStoreMessageOption.getState().startNewChatDraft(false))
      await settle()
      const listing = await view.result.current.listRetainedDrafts()
      expect(listing.drafts).toHaveLength(1)
      expect(listing.drafts[0]).toMatchObject({
        content: "UNSENT SOURCE",
        handle: { address: oldAddress },
        surfaceKey: root
      })
      act(() =>
        useStoreMessageOption
          .getState()
          .setChatDraftAddress(listing.drafts[0].handle.address)
      )
      await settle()
      expect(view.result.current.form.values.message).toBe("UNSENT SOURCE")
    }
  )

  it("filters foreign owners, surfaces, scopes, saved identities, malformed handles, empty text and the active address", async () => {
    const view = mount()
    await settle()
    const bucket = createLocalRegistryBucket<unknown>({
      prefix: "registry:draft:"
    })
    const own = chatDraftOwnerKey(scope())
    const cases: [string, ChatDraftAddress, string, string, unknown][] = [
      [root, fresh("good", true), own, "GOOD", undefined],
      [root, fresh("foreign"), "a".repeat(64), "FOREIGN", undefined],
      [
        "tldw:sidepanelChatDraft:other",
        fresh("other-surface"),
        own,
        "OTHER",
        undefined
      ],
      [
        root,
        {
          ...fresh("other-scope"),
          scope: { type: "workspace", workspaceId: "elsewhere" }
        },
        own,
        "WORKSPACE",
        undefined
      ],
      [
        root,
        { kind: "local", id: "saved", profileId: "profile" },
        own,
        "SAVED",
        undefined
      ],
      [
        root,
        fresh("malformed"),
        own,
        "BAD",
        { ownerHash: own, address: { kind: "fresh" } }
      ],
      [root, fresh("empty"), own, "  ", undefined],
      [
        root,
        useStoreMessageOption.getState().chatDraftAddress!,
        own,
        "ACTIVE",
        undefined
      ]
    ]
    for (const [surface, address, owner, content, malformed] of cases)
      await bucket.set(
        `${conversationDraftKey(surface, address)}:owner:${owner}`,
        {
          content,
          metadata: {
            chatDraftHandle: malformed ?? { ownerHash: owner, address }
          }
        }
      )
    expect(
      (await view.result.current.listRetainedDrafts()).drafts.map(
        (draft) => draft.content
      )
    ).toEqual(["GOOD"])
  })

  it.each(["New", "account"])(
    "does not reveal delayed recovery results after %s",
    async (boundary) => {
      const view = mount()
      await settle()
      act(() => view.result.current.setMessageValue("OLD PRIVATE TEXT"))
      await settle()
      act(() => useStoreMessageOption.getState().startNewChatDraft())
      await settle()
      const getAll = PlasmoStorage.prototype.getAll
      let release!: () => void
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      vi.spyOn(PlasmoStorage.prototype, "getAll").mockImplementation(
        async function (this: PlasmoStorage) {
          const values = await getAll.call(this)
          await held
          return values
        }
      )
      const listing = view.result.current.listRetainedDrafts()
      act(() => {
        if (boundary === "New")
          useStoreMessageOption.getState().startNewChatDraft()
        else {
          authority.user = "bob"
          window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
        }
      })
      await act(async () => {
        release()
      })
      expect((await listing).drafts).toEqual([])
    }
  )

  it("retires the fresh recovery alias after first-ID binding while retaining its text", async () => {
    vi.stubGlobal(
      "navigator",
      Object.assign(Object.create(navigator), {
        locks: {
          request: (_name: string, operation: () => Promise<unknown>) =>
            operation()
        }
      })
    )
    const view = mount()
    await settle()
    const source = useStoreMessageOption.getState().chatDraftAddress!
    act(() => view.result.current.setMessageValue("NEXT INPUT"))
    await settle()
    act(() =>
      useStoreMessageOption.getState().setChatDraftAddress(
        { kind: "local", id: "saved", profileId: "profile" },
        {
          kind: "bind",
          sourceAddress: source,
          expectedRevision: useStoreMessageOption.getState().chatDraftRevision
        }
      )
    )
    await settle()
    expect((await view.result.current.listRetainedDrafts()).drafts).toEqual([])
    const record = await new PlasmoStorage().get<{
      value: { content: string; metadata: Record<string, unknown> }
    }>(
      `registry:draft:${conversationDraftKey(root, source)}:owner:${chatDraftOwnerKey(scope())}`
    )
    expect(record?.value.content).toBe("NEXT INPUT")
    expect(record?.value.metadata.chatDraftHandle).toBeUndefined()
  })

  it.each(["rejected", "volatile"])(
    "refuses recovery copying without a durable acknowledgement (%s)",
    async (outcome) => {
      const view = mount("tldw:sidepanelChatDraft:source", true)
      await settle()
      act(() => view.result.current.setMessageValue("RETAINED SOURCE"))
      await settle()
      act(() => useStoreMessageOption.getState().startNewChatDraft())
      view.rerender({ surface: "tldw:sidepanelChatDraft:active" })
      await settle()
      act(() => view.result.current.setMessageValue("CURRENT TEXT"))
      await settle()
      const candidate = (await view.result.current.listRetainedDrafts())
        .drafts[0]
      if (outcome === "rejected")
        vi.spyOn(PlasmoStorage.prototype, "set").mockRejectedValue(
          new Error("quota")
        )
      else
        vi.spyOn(
          PlasmoStorage.prototype,
          "hasPersistentBackend",
          "get"
        ).mockReturnValue(false)
      expect(
        await view.result.current.copyRetainedDraft(
          candidate,
          "tldw:sidepanelChatDraft:new"
        )
      ).toBe(false)
      expect(view.result.current.form.values.message).toBe("CURRENT TEXT")
      expect(
        (await view.result.current.listRetainedDrafts()).drafts.some(
          (draft) => draft.key === candidate.key
        )
      ).toBe(true)
    }
  )

  it("copies an occupied compact draft to another logical tab without deleting its source", async () => {
    const view = mount("tldw:sidepanelChatDraft:source", true)
    await settle()
    act(() => view.result.current.setMessageValue("CLOSED SOURCE"))
    await settle()
    act(() => useStoreMessageOption.getState().startNewChatDraft())
    view.rerender({ surface: "tldw:sidepanelChatDraft:active" })
    await settle()
    const candidate = (await view.result.current.listRetainedDrafts()).drafts[0]
    expect(
      await view.result.current.copyRetainedDraft(
        candidate,
        "tldw:sidepanelChatDraft:recovered"
      )
    ).toBe(true)
    expect(await view.result.current.copyRetainedDraft(candidate, root)).toBe(
      false
    )
    act(() =>
      useStoreMessageOption
        .getState()
        .setChatDraftAddress(candidate.handle.address)
    )
    view.rerender({ surface: "tldw:sidepanelChatDraft:recovered" })
    await settle()
    expect(view.result.current.form.values.message).toBe("CLOSED SOURCE")
    const listing = await view.result.current.listRetainedDrafts()
    expect(
      listing.drafts.some(
        (draft) => draft.surfaceKey === "tldw:sidepanelChatDraft:source"
      )
    ).toBe(true)
  })
  it("keeps the only durable fresh recovery source when saved-key writes fail through a cold remount", async () => {
    vi.stubGlobal(
      "navigator",
      Object.assign(Object.create(navigator), {
        locks: {
          request: (_name: string, operation: () => Promise<unknown>) =>
            operation()
        }
      })
    )
    const view = mount()
    await settle()
    const source = useStoreMessageOption.getState().chatDraftAddress!
    const own = chatDraftOwnerKey(scope())
    const saved = { kind: "local" as const, id: "saved", profileId: "profile" }
    const destination = `registry:draft:${conversationDraftKey(root, saved)}:owner:${own}`
    act(() => view.result.current.setMessageValue("ONLY DURABLE NEXT INPUT"))
    await settle()
    const set = PlasmoStorage.prototype.set
    vi.spyOn(PlasmoStorage.prototype, "set").mockImplementation(async function (
      this: PlasmoStorage,
      key: string,
      value: unknown
    ) {
      if (key === destination) throw new Error("quota")
      return set.call(this, key, value)
    })
    act(() =>
      useStoreMessageOption.getState().setChatDraftAddress(saved, {
        kind: "bind",
        sourceAddress: source,
        expectedRevision: useStoreMessageOption.getState().chatDraftRevision
      })
    )
    await settle()
    const acknowledged = view.result.current.draftSaved
    view.unmount()
    sessionStorage.clear()
    act(() => useStoreMessageOption.getState().startNewChatDraft())
    const reopened = mount()
    await settle()
    const drafts = (await reopened.result.current.listRetainedDrafts()).drafts
    expect(drafts).toHaveLength(1)
    expect(drafts[0]).toMatchObject({
      content: "ONLY DURABLE NEXT INPUT",
      handle: { address: source }
    })
    expect(acknowledged).toBe(false)
  })

  it("cold restores an independent compact copy with a matching workspace/Temporary key and marker", async () => {
    const source = {
      ...fresh("workspace-source", true),
      scope: { type: "workspace" as const, workspaceId: "work" }
    }
    act(() => useStoreMessageOption.getState().setChatDraftAddress(source))
    const view = mount("tldw:sidepanelChatDraft:source", true)
    await settle()
    act(() => view.result.current.setMessageValue("WORKSPACE COPY"))
    await settle()
    act(() =>
      useStoreMessageOption
        .getState()
        .setChatDraftAddress({ ...source, id: "active" })
    )
    view.rerender({ surface: "tldw:sidepanelChatDraft:active" })
    await settle()
    const candidate = (await view.result.current.listRetainedDrafts()).drafts[0]
    const destination = { ...candidate.handle.address, id: "independent-copy" }
    expect(
      await view.result.current.copyRetainedDraft(
        candidate,
        "tldw:sidepanelChatDraft:copy",
        destination
      )
    ).toBe(true)
    view.unmount()
    sessionStorage.clear()
    act(() => useStoreMessageOption.getState().setChatDraftAddress(destination))
    const cold = mount("tldw:sidepanelChatDraft:copy", true)
    await settle()
    expect(cold.result.current.form.values.message).toBe("WORKSPACE COPY")
    act(() =>
      useStoreMessageOption
        .getState()
        .setChatDraftAddress({ ...source, id: "another" })
    )
    cold.rerender({ surface: "tldw:sidepanelChatDraft:another" })
    await settle()
    expect(
      (await cold.result.current.listRetainedDrafts()).drafts.find(
        (draft) => draft.surfaceKey === "tldw:sidepanelChatDraft:copy"
      )?.handle.address
    ).toEqual(destination)
  })
  it.each([
    "delivered",
    "queue",
    "newer input",
    "newer attempt",
    "New",
    "account"
  ])(
    "retires an empty first-ID source only for its acknowledged delivery (%s)",
    async (boundary) => {
      vi.stubGlobal(
        "navigator",
        Object.assign(Object.create(navigator), {
          locks: {
            request: (_name: string, operation: () => Promise<unknown>) =>
              operation()
          }
        })
      )
      const view = mount()
      await settle()
      const source = useStoreMessageOption.getState().chatDraftAddress!
      const sourceKey = `registry:draft:${conversationDraftKey(root, source)}:owner:${chatDraftOwnerKey(scope())}`
      act(() => view.result.current.setMessageValue("SUBMITTED INPUT"))
      await settle()
      let attempt!: number
      act(() => {
        attempt = view.result.current.beginPromptAssistReset()
      })
      act(() =>
        useStoreMessageOption.getState().setChatDraftAddress(
          { kind: "local", id: "saved", profileId: "profile" },
          {
            kind: "bind",
            sourceAddress: source,
            expectedRevision: useStoreMessageOption.getState().chatDraftRevision
          }
        )
      )
      await settle()
      expect(
        (await new PlasmoStorage().get<StoredDraft>(sourceKey))?.value.metadata
          .chatDraftHandle
      ).toBeDefined()
      if (boundary === "newer input") {
        vi.spyOn(PlasmoStorage.prototype, "set").mockRejectedValue(
          new Error("quota")
        )
        act(() => view.result.current.setMessageValue("NEW UNSENT INPUT"))
      } else if (boundary === "newer attempt")
        act(() => view.result.current.beginPromptAssistReset())
      else if (boundary === "New")
        act(() => useStoreMessageOption.getState().startNewChatDraft())
      else if (boundary === "account")
        act(() => {
          authority.user = "bob"
          window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
        })
      act(() =>
        view.result.current.markPromptAssistAttemptSaved(
          attempt,
          boundary !== "queue"
        )
      )
      await settle()
      const record = await new PlasmoStorage().get<StoredDraft>(sourceKey)
      expect(record?.value.content).toBe("SUBMITTED INPUT")
      expect(Boolean(record?.value.metadata.chatDraftHandle)).toBe(
        boundary !== "delivered"
      )
    }
  )

  it.each(["New", "account"])(
    "does not retire the source after a saved-key acknowledgement arrives following %s",
    async (boundary) => {
      vi.stubGlobal(
        "navigator",
        Object.assign(Object.create(navigator), {
          locks: {
            request: (_name: string, operation: () => Promise<unknown>) =>
              operation()
          }
        })
      )
      const view = mount()
      await settle()
      const source = useStoreMessageOption.getState().chatDraftAddress!
      const own = chatDraftOwnerKey(scope())
      const sourceKey = `registry:draft:${conversationDraftKey(root, source)}:owner:${own}`
      act(() => view.result.current.setMessageValue("RETAIN UNTIL CURRENT ACK"))
      await settle()
      const saved = {
        kind: "local" as const,
        id: "held-save",
        profileId: "profile"
      }
      const destination = `registry:draft:${conversationDraftKey(root, saved)}:owner:${own}`
      const set = PlasmoStorage.prototype.set
      let release!: () => void
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      vi.spyOn(PlasmoStorage.prototype, "set").mockImplementation(
        async function (this: PlasmoStorage, key: string, value: unknown) {
          await set.call(this, key, value)
          if (key === destination) await held
        }
      )
      act(() =>
        useStoreMessageOption.getState().setChatDraftAddress(saved, {
          kind: "bind",
          sourceAddress: source,
          expectedRevision: useStoreMessageOption.getState().chatDraftRevision
        })
      )
      await settle()
      act(() => {
        if (boundary === "New")
          useStoreMessageOption.getState().startNewChatDraft()
        else {
          authority.user = "bob"
          window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
        }
      })
      await act(async () => {
        release()
      })
      await settle()
      expect(
        (await new PlasmoStorage().get<StoredDraft>(sourceKey))?.value.metadata
          .chatDraftHandle
      ).toBeDefined()
    }
  )
  it.each(["absent", "older"])(
    "retains immediate pre-debounce submission text after no acknowledgement and cold recovery (%s source)",
    async (initial) => {
      act(() => useStoreMessageOption.getState().setTemporaryChat(true))
      const source = useStoreMessageOption.getState().chatDraftAddress!
      const view = mount()
      await settle()
      if (initial === "older") {
        act(() => view.result.current.setMessageValue("OLDER TEXT"))
        await settle()
      }
      act(() => view.result.current.setMessageValue("LATEST IMMEDIATE SEND"))
      act(() => view.result.current.beginPromptAssistReset())
      // No successful submission/creation receipt: the request was rejected or
      // its result remains unknown, then the user leaves and opens Chat again.
      act(() => useStoreMessageOption.getState().startNewChatDraft())
      await settle()
      view.unmount()
      sessionStorage.clear()
      const cold = mount()
      await settle()
      expect(
        (await cold.result.current.listRetainedDrafts()).drafts
      ).toMatchObject([
        {
          content: "LATEST IMMEDIATE SEND",
          handle: { address: source },
          metadata: { collapsedRange: { start: 1, end: 5 }, wasExpanded: false }
        }
      ])
    }
  )
  it("keeps immediate-send text when the actual compact callback receives an unconfirmed pre-bind result", async () => {
    const view = mount("tldw:sidepanelChatDraft:source", true)
    await settle()
    const source = useStoreMessageOption.getState().chatDraftAddress!
    act(() => view.result.current.setMessageValue("LEGACY UNCONFIRMED INPUT"))
    let attempt!: number
    act(() => {
      attempt = view.result.current.beginPromptAssistReset()
    })
    await settle() // the legacy preflight/create error resolves asynchronously
    act(() => deployedAfterSend(attempt, view.result.current)(undefined))
    act(() => useStoreMessageOption.getState().startNewChatDraft())
    view.rerender({ surface: "tldw:sidepanelChatDraft:active" })
    await settle()
    view.unmount()
    sessionStorage.clear()
    const cold = mount("tldw:sidepanelChatDraft:active", true)
    await settle()
    expect(
      (await cold.result.current.listRetainedDrafts()).drafts
    ).toMatchObject([
      { content: "LEGACY UNCONFIRMED INPUT", handle: { address: source } }
    ])
  })
})
