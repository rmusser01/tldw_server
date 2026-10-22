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
import { useComposerSubmit } from "../hooks/useComposerSubmit"
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
    "restorePromptAssistAttempt",
    "isChatSubmitSuccess",
    "normalizeChatSubmitResult",
    "clearSubmittedDocuments",
    "selectedDocuments",
    "contextFiles",
    "intent",
    "knowledgeMentionRevision",
    "submittedKnowledgeMentionRevision",
    "setContextFiles",
    "setKnowledgeMentionActive",
    `${afterSendCode}; return afterSend;`
  )(
    attempt,
    composer.markPromptAssistAttemptSaved,
    composer.clearDraft,
    composer.restorePromptAssistAttempt,
    isChatSubmitSuccess,
    normalizeChatSubmitResult,
    vi.fn(),
    [],
    [],
    {isImageCommand:false},
    {current:0},
    0,
    vi.fn(),
    vi.fn()
  ) as (
    result:
      | import("@/hooks/chat/chat-action-utils").ChatSubmitResult
      | undefined
  ) => void

const webSubmitAst = ts.createSourceFile(
  "usePlaygroundSubmit.ts",
  readFileSync(
    path.resolve(
      __dirname,
      "../../../Option/Playground/hooks/usePlaygroundSubmit.ts"
    ),
    "utf8"
  ),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TS
)
let webAfterSend: ts.Expression | undefined
const findWebAfterSend = (node: ts.Node) => {
  if (
    ts.isPropertyAssignment(node) &&
    node.name.getText(webSubmitAst) === "afterSend" &&
    node.initializer
      .getText(webSubmitAst)
      .includes("markPromptAssistAttemptSaved")
  )
    webAfterSend = node.initializer
  ts.forEachChild(node, findWebAfterSend)
}
findWebAfterSend(webSubmitAst)
if (!webAfterSend) throw new Error("Missing actual WebUI send outcome callback")
const webAfterSendCode = ts.transpileModule(
  `const afterSend = ${webAfterSend.getText(webSubmitAst)};`,
  { compilerOptions: { target: ts.ScriptTarget.ES2022 } }
).outputText
const deployedWebAfterSend = (
  attempt: number,
  composer: ReturnType<typeof useComposerText>
) =>
  new Function(
    "promptAssistAttemptId",
    "markPromptAssistAttemptSaved",
    "restorePromptAssistAttempt",
    "isChatSubmitSuccess",
    "normalizeChatSubmitResult",
    "messageForModel",
    "clearImportedSidepanelContext",
    "draftOwnership",
    "pageContextRevisionRef",
    "pageContextRevision",
    `${webAfterSendCode}; return afterSend;`
  )(
    attempt,
    composer.markPromptAssistAttemptSaved,
    composer.restorePromptAssistAttempt,
    isChatSubmitSuccess,
    normalizeChatSubmitResult,
    undefined,
    vi.fn(),
    composer.captureDraftOwnership(),
    { current: { revision: 0 } },
    0
  ) as (
    result: import("@/hooks/chat/chat-action-utils").ChatSubmitResult
  ) => void

const deployedSendError = (
  surface: "compact" | "WebUI",
  attempt: number,
  composer: ReturnType<typeof useComposerText>
) => {
  const ast = surface === "compact" ? compactFormAst : webSubmitAst
  let handler: ts.Expression | undefined
  const find = (node: ts.Node) => {
    if (
      ts.isPropertyAssignment(node) &&
      node.name.getText(ast) === "onSendError" &&
      node.initializer.getText(ast).includes("restorePromptAssistAttempt")
    )
      handler = node.initializer
    ts.forEachChild(node, find)
  }
  find(ast)
  if (!handler)
    throw new Error(`Missing ${surface} dispatch rejection recovery callback`)
  const code = ts.transpileModule(
    `const onSendError = ${handler.getText(ast)};`,
    { compilerOptions: { target: ts.ScriptTarget.ES2022 } }
  ).outputText
  return new Function(
    "promptAssistAttemptId",
    "restorePromptAssistAttempt",
    `${code}; return onSendError;`
  )(attempt, composer.restorePromptAssistAttempt) as (error: unknown) => void
}

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
const mount = (
  surface = root,
  tabs = false,
  options?: Pick<
    Parameters<typeof useComposerText>[0],
    "getDraftMetadata" | "restoreWithMetadata"
  >
) =>
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
        getDraftMetadata:
          options?.getDraftMetadata ??
          (() => ({
            collapsedRange: { start: 1, end: 5 },
            wasExpanded: false
          })),
        restoreWithMetadata: options?.restoreWithMetadata
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

  it("cold lists a next message with its existing settled chat association", async () => {
    vi.stubGlobal("navigator", Object.assign(Object.create(navigator), { locks: { request: (_key: string, work: () => Promise<unknown>) => work() } }))
    const { createChatWithRecovery, markChatCreationBound, markChatCreationSettled } = await import("@/services/pending-chat-promotion")
    const source = fresh("quick-private")
    const handle = { address: source, ownerHash: chatDraftOwnerKey(scope()) }
    const snapshot = { requestScope: scope(), scopeSignal: new AbortController().signal,
      scopeInvalidatedSignal: new AbortController().signal } as unknown as import("@/services/service-prompts").ServicePromptSnapshot
    const created = await createChatWithRecovery({ snapshot, handle, surface: "playground", inputText: "FIRST SENT", setup: { characterId: 4 },
      isCurrent: () => true, create: async id => ({ id }) })
    await markChatCreationBound(handle, created.id)
    await markChatCreationSettled(handle, created.id)
    const bucket = createLocalRegistryBucket<{ content: string; metadata: Record<string, unknown> }>({ prefix: "registry:draft:" })
    const key = `${conversationDraftKey(root, source)}:owner:${handle.ownerHash}`
    await bucket.set(key, { content: "  NEXT UNSENT\nExact spaces  ", metadata: { chatDraftHandle: handle } })
    sessionStorage.clear()
    const view = mount()
    await settle()
    const retained = (await view.result.current.listRetainedDrafts()).drafts
    expect(retained).toHaveLength(1)
    expect(retained[0]).toMatchObject({ content: "  NEXT UNSENT\nExact spaces  ", creationAttempt: { state: "settled", chatId: created.id, inputText: "" } })
    expect((await bucket.get(key))?.value.content).toBe("  NEXT UNSENT\nExact spaces  ")
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
      expect(listing.unavailable).toBe(false)
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
  it.each(["fresh", "local", "native"] as const)(
    "restores the original untouched %s composer after an unsuccessful send",
    async (kind) => {
      if (kind !== "fresh")
        act(() =>
          useStoreMessageOption.getState().setChatDraftAddress(
            kind === "local"
              ? { kind: "local", id: "local-chat", profileId: "profile" }
              : {
                  kind: "native",
                  id: "native-chat",
                  ownerKey: "server-owner",
                  scope: { type: "global" }
                }
          )
        )
      const view = mount()
      await settle()
      act(() => {
        view.result.current.setMessageValue("  ORIGINAL COMPOSER TEXT  ")
        view.result.current.form.setFieldValue("image", "original-image")
      })
      let attempt!: number
      act(() => {
        attempt = view.result.current.beginPromptAssistReset()
      })
      await settle()
      act(() => view.result.current.restorePromptAssistAttempt(attempt))
      expect(view.result.current.form.values).toEqual({
        message: "  ORIGINAL COMPOSER TEXT  ",
        image: "original-image",
        imageFilename: ""
      })
      expect(view.result.current.promptAssistSavedAttemptId).toBeNull()
    }
  )

  it("restores a failure that resolves before the optimistic reset commits", async () => {
    const view = mount()
    await settle()
    act(() => view.result.current.setMessageValue("FAST FAILED TEXT"))
    await act(async () => {
      const attempt = view.result.current.beginPromptAssistReset()
      await Promise.resolve()
      view.result.current.restorePromptAssistAttempt(attempt)
    })
    expect(view.result.current.form.values.message).toBe("FAST FAILED TEXT")
  })

  it("restores original text after its one accepted first-ID bind", async () => {
    const view = mount()
    await settle()
    const source = useStoreMessageOption.getState().chatDraftAddress!
    act(() => view.result.current.setMessageValue("BOUND FAILED INPUT"))
    let attempt!: number
    act(() => {
      attempt = view.result.current.beginPromptAssistReset()
    })
    act(() =>
      useStoreMessageOption.getState().setChatDraftAddress(
        {
          kind: "native",
          id: "new-chat",
          ownerKey: "server-owner",
          scope: { type: "global" }
        },
        {
          kind: "bind",
          sourceAddress: source,
          expectedRevision: useStoreMessageOption.getState().chatDraftRevision
        }
      )
    )
    await settle()
    act(() => view.result.current.restorePromptAssistAttempt(attempt))
    expect(view.result.current.form.values.message).toBe("BOUND FAILED INPUT")
  })

  it.each([
    "text",
    "type then erase",
    "image",
    "image then erase",
    "new attempt",
    "New",
    "ABA",
    "account",
    "unmount"
  ])("does not restore an old attempt after %s", async (boundary) => {
    const view = mount()
    await settle()
    const source = useStoreMessageOption.getState().chatDraftAddress!
    act(() => view.result.current.setMessageValue("OLD FAILED INPUT"))
    let attempt!: number
    act(() => {
      attempt = view.result.current.beginPromptAssistReset()
    })
    await settle()
    const recover = () =>
      view.result.current.restorePromptAssistAttempt(attempt)
    act(() => {
      if (boundary === "text" || boundary === "type then erase")
        view.result.current.setMessageValue("NEW INPUT")
      else if (boundary === "image" || boundary === "image then erase")
        view.result.current.form.setFieldValue("image", "new-image")
      else if (boundary === "new attempt")
        view.result.current.beginPromptAssistReset()
      else if (boundary === "New" || boundary === "ABA")
        useStoreMessageOption.getState().startNewChatDraft()
      else if (boundary === "account") {
        authority.user = "bob"
        window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
      }
    })
    await settle()
    if (boundary === "type then erase")
      act(() => view.result.current.setMessageValue(""))
    if (boundary === "image then erase")
      act(() => view.result.current.form.setFieldValue("image", ""))
    if (boundary === "ABA") {
      act(() => useStoreMessageOption.getState().setChatDraftAddress(source))
      await settle()
      act(() => view.result.current.setMessageValue(""))
    }
    if (boundary === "unmount") view.unmount()
    const values = { ...view.result.current.form.values }
    act(recover)
    expect(view.result.current.form.values).toEqual(values)
  })

  it.each(["compact", "WebUI"])(
    "the actual %s callback restores explicit preflight refusals and skipped text",
    async (surface) => {
      for (const outcome of [
        { status: "failed", errorMessage: "preflight refused before admission", restoreDraft: true },
        { status: "skipped", reason: "choose a model" }
      ] as const) {
        const view = mount(
          surface === "compact" ? "tldw:sidepanelChatDraft:active" : root,
          surface === "compact"
        )
        await settle()
        act(() =>
          view.result.current.setMessageValue("ORIGINAL CALLBACK INPUT")
        )
        let attempt!: number
        act(() => {
          attempt = view.result.current.beginPromptAssistReset()
        })
        await settle()
        act(() =>
          (surface === "compact" ? deployedAfterSend : deployedWebAfterSend)(
            attempt,
            view.result.current
          )(outcome)
        )
        expect(view.result.current.form.values.message).toBe(
          "ORIGINAL CALLBACK INPUT"
        )
        expect(view.result.current.promptAssistSavedAttemptId).toBeNull()
        view.unmount()
      }
    }
  )

  it.each(["compact", "WebUI"] as const)(
    "the actual %s callback does not restore a failed send without a no-admission flag",
    async surface => {
      const view = mount(surface === "compact" ? "tldw:sidepanelChatDraft:active" : root, surface === "compact")
      await settle()
      act(() => view.result.current.setMessageValue("UNCONFIRMED DELIVERY"))
      let attempt!: number
      act(() => { attempt = view.result.current.beginPromptAssistReset() })
      await settle()
      act(() => (surface === "compact" ? deployedAfterSend : deployedWebAfterSend)(attempt, view.result.current)(
        { status: "failed", errorMessage: "provider failed after possible admission" }
      ))
      expect(view.result.current.form.values.message).toBe("")
      expect(view.result.current.promptAssistSavedAttemptId).toBeNull()
      view.unmount()
    }
  )

  it.each(["compact", "WebUI"] as const)(
    "the actual %s rejection hook restores original text on fast dispatch rejection",
    async (surface) => {
      const view = mount(
        surface === "compact" ? "tldw:sidepanelChatDraft:active" : root,
        surface === "compact"
      )
      await settle()
      act(() => view.result.current.setMessageValue("ORIGINAL REJECTED TEXT"))
      const error = new Error("pre-dispatch failure")
      const dispatcher = renderHook(() =>
        useComposerSubmit({
          sendMessage: async () => {
            throw error
          }
        })
      )
      await act(async () => {
        const attempt = view.result.current.beginPromptAssistReset()
        await expect(
          dispatcher.result.current.dispatch(
            { message: "TRANSFORMED REQUEST TEXT" },
            {
              onSendError: deployedSendError(
                surface,
                attempt,
                view.result.current
              )
            }
          )
        ).rejects.toBe(error)
      })
      expect(view.result.current.form.values.message).toBe(
        "ORIGINAL REJECTED TEXT"
      )
      expect(view.result.current.promptAssistSavedAttemptId).toBeNull()
    }
  )

  it("restores reset-time collapse metadata without reusing a later-mutated object", async () => {
    const metadata = { collapsedRange: { start: 2, end: 8 }, wasExpanded: true }
    const restore = vi.fn()
    const view = mount(root, false, {
      getDraftMetadata: () => metadata,
      restoreWithMetadata: restore
    })
    await settle()
    act(() => view.result.current.setMessageValue("COLLAPSED ORIGINAL TEXT"))
    let attempt!: number
    act(() => {
      attempt = view.result.current.beginPromptAssistReset()
    })
    metadata.collapsedRange.end = 200
    metadata.wasExpanded = false
    await settle()
    restore.mockClear()
    act(() => view.result.current.restorePromptAssistAttempt(attempt))
    expect(restore).toHaveBeenCalledWith("COLLAPSED ORIGINAL TEXT", {
      collapsedRange: { start: 2, end: 8 },
      wasExpanded: true
    })
  })
  it.each(["local", "native"] as const)(
    "keeps pending %s input reachable after held preparation and cold reopen",
    async (kind) => {
      const locks = new Map<string, Promise<unknown>>()
      vi.stubGlobal("navigator", {
        userAgent: navigator.userAgent,
        locks: {
          request: (name: string, operation: () => Promise<unknown>) => {
            const previous = locks.get(name) ?? Promise.resolve()
            const next = previous.catch(() => undefined).then(operation)
            locks.set(name, next)
            return next
          }
        }
      })
      const address: ChatDraftAddress =
        kind === "local"
          ? { kind: "local", id: "pending-local", profileId: "profile" }
          : {
              kind: "native",
              id: "pending-native",
              ownerKey: "server-owner",
              scope: { type: "global" }
            }
      act(() => useStoreMessageOption.getState().setChatDraftAddress(address))
      const view = mount()
      await settle()
      act(() => view.result.current.setMessageValue("PENDING ORIGINAL INPUT"))
      let attempt!: number
      act(() => {
        attempt = view.result.current.beginPromptAssistReset()
      })
      await settle() // model preparation is still pending; no owner admission exists
      const oldRecovery = view.result.current.restorePromptAssistAttempt
      act(() => useStoreMessageOption.getState().startNewChatDraft())
      await settle()
      view.unmount()
      oldRecovery(attempt) // the later failed/skipped result cannot restore another view
      sessionStorage.clear()
      act(() => useStoreMessageOption.getState().setChatDraftAddress(address))
      const cold = mount()
      await settle()
      expect(cold.result.current.form.values.message).toBe(
        "PENDING ORIGINAL INPUT"
      )
    }
  )
})
