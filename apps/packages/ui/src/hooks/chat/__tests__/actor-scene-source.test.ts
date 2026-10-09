import { beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"
import { captureChatDraftLoad } from "../chat-draft-transition"
import { captureCurrentActorSceneTarget } from "../actor-scene-source"
import type {
  HistorySelectionController,
  HistoryLoadReceipt
} from "../useHistorySelection"

const requestScope = {
  config: {
    serverUrl: "https://scene-source.example",
    authMode: "single-user" as const
  },
  userId: 1
}
const receipt = (kind: "native" | "local"): HistoryLoadReceipt => ({
  owner:
    kind === "native"
      ? {
          kind,
          conversation_id: "one",
          owner_key: "h1-owner",
          request_scope: requestScope,
          validate_lease: () => true
        }
      : {
          kind,
          conversation_id: "one",
          owner_key: "h1-owner",
          profile_id: "local-one"
        },
  view: {
    owner_key: "h1-owner",
    conversation_id: "one",
    selection_revision: 1
  } as HistoryLoadReceipt["view"]
})
const selection = (h1: HistoryLoadReceipt | null = null) => {
  let current = h1,
    accountCurrent = true
  const controller = {
    getCurrent: () => current ?? { owner: null, view: null },
    draftAccount: { requestScope, isCurrent: () => accountCurrent }
  } as unknown as HistorySelectionController
  return {
    controller,
    replace: (next: HistoryLoadReceipt | null) => {
      current = next
    },
    retireAccount: () => {
      accountCurrent = false
    }
  }
}
const install = (h1: HistoryLoadReceipt) => {
  expect(captureChatDraftLoad("switch")(h1)).toBe(true)
  useStoreMessageOption.setState({
    historyId: h1.owner.kind === "local" ? "one" : null,
    serverChatId: h1.owner.kind === "native" ? "one" : null
  })
}

beforeEach(() => {
  useStoreMessageOption.getState().startNewChatDraft()
  useStoreMessageOption.setState({ historyId: null, serverChatId: null })
})

describe("scene destination captured from actual store and declared accepted H1 receipt", () => {
  it("keeps unique fresh source current until New retires it", () => {
    const { controller } = selection()
    const original = captureCurrentActorSceneTarget(controller)
    expect(original?.isCurrent()).toBe(true)
    useStoreMessageOption.getState().startNewChatDraft()
    expect(original?.isCurrent()).toBe(false)
    expect(captureCurrentActorSceneTarget(controller)?.key).not.toBe(
      original?.key
    )
  })
  it("refuses a fresh source without verified account ownership", () => {
    const source = selection()
    source.retireAccount()
    expect(captureCurrentActorSceneTarget(source.controller)).toBeNull()
    expect(captureCurrentActorSceneTarget(null)).toBeNull()
  })
  it.each(["native", "local"] as const)(
    "retains %s ownership separately from cursor revision",
    (kind) => {
      const h1 = receipt(kind),
        source = selection(h1)
      install(h1)
      const target = captureCurrentActorSceneTarget(source.controller)
      expect(target?.isCurrent()).toBe(true)
      source.replace({
        owner: h1.owner,
        view: { ...h1.view, selection_revision: 2 }
      })
      expect(target?.isCurrent()).toBe(true)
    }
  )
  it.each(["native", "local"] as const)(
    "retires same-ID %s owner replacement",
    (kind) => {
      const h1 = receipt(kind),
        source = selection(h1)
      install(h1)
      const target = captureCurrentActorSceneTarget(source.controller)
      source.replace({ owner: { ...h1.owner }, view: h1.view })
      expect(target?.isCurrent()).toBe(false)
    }
  )
  it("uses the accepted native owner while an independent fresh-draft verifier is unavailable", () => {
    const h1 = receipt("native"),
      source = selection(h1)
    install(h1)
    source.retireAccount()
    expect(captureCurrentActorSceneTarget(source.controller)?.isCurrent()).toBe(
      true
    )
  })
  it("does not substitute an unrelated draft account for an accepted native owner", () => {
    const h1 = receipt("native"),
      source = selection(h1)
    install(h1)
    source.controller.draftAccount = {
      requestScope: { ...requestScope, userId: 2 },
      isCurrent: () => true
    }
    expect(captureCurrentActorSceneTarget(source.controller)?.isCurrent()).toBe(
      true
    )
  })
  it("keeps genuine-local scene current after unrelated backend verification retires", () => {
    const h1 = receipt("local"),
      source = selection(h1)
    install(h1)
    const target = captureCurrentActorSceneTarget(source.controller)
    source.retireAccount()
    expect(target?.isCurrent()).toBe(true)
    expect(target?.profileIsCurrent()).toBe(false)
  })
  it("refuses a local address not matched by its actual H1 profile", () => {
    const h1 = receipt("local"),
      source = selection(h1)
    install(h1)
    source.replace({
      ...h1,
      owner: {
        ...h1.owner,
        profile_id: "different"
      } as HistoryLoadReceipt["owner"]
    })
    expect(captureCurrentActorSceneTarget(source.controller)).toBeNull()
  })
  it("refuses inconsistent raw server ID versus draft address", () => {
    const h1 = receipt("native"),
      source = selection(h1)
    install(h1)
    useStoreMessageOption.setState({ serverChatId: "different" })
    expect(captureCurrentActorSceneTarget(source.controller)).toBeNull()
  })
  it("retires raw destination replacement without adopting it", () => {
    const h1 = receipt("native"),
      source = selection(h1)
    install(h1)
    const target = captureCurrentActorSceneTarget(source.controller)
    useStoreMessageOption.setState({ serverChatId: "different" })
    expect(target?.isCurrent()).toBe(false)
  })
})
