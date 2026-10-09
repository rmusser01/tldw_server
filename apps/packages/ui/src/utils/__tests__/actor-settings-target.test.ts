import { describe, expect, it } from "vitest"
import { chatDraftOwnerKey } from "@/hooks/useChatDraftOwner"
import {
  captureActorSettingsTarget,
  captureActorProfileTarget
} from "../actor-settings-target"
import type { HistoryLoadReceipt } from "@/hooks/chat/useHistorySelection"
import type { ChatDraftAddress } from "../chat-draft-address"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"

const scope: ServicePromptRequestScope = {
  config: { serverUrl: "https://scene.example", authMode: "single-user" },
  userId: 1
}
const fresh: ChatDraftAddress = {
  kind: "fresh",
  id: "fresh-one",
  scope: { type: "global" },
  temporary: false
}
const native: ChatDraftAddress = {
  kind: "native",
  id: "conversation-one",
  ownerKey: "h1-owner",
  scope: { type: "global" }
}
const local: ChatDraftAddress = {
  kind: "local",
  id: "conversation-one",
  profileId: "local-profile"
}
const receipt = (kind: "native" | "local"): HistoryLoadReceipt => ({
  owner:
    kind === "native"
      ? {
          kind,
          conversation_id: "conversation-one",
          owner_key: "h1-owner",
          request_scope: scope,
          validate_lease: () => true
        }
      : {
          kind,
          conversation_id: "conversation-one",
          owner_key: "h1-owner",
          profile_id: "local-profile"
        },
  view: {
    conversation_id: "conversation-one",
    owner_key: "h1-owner"
  } as HistoryLoadReceipt["view"]
})
const captured = (address: ChatDraftAddress = fresh) => ({
  address,
  requestScope: scope,
  accountIsCurrent: () => true,
  isCurrent: () => true
})

describe("scene target qualification from original address and H1 owner", () => {
  it("keeps unique fresh addresses separate within a verified account", () => {
    const a = captureActorSettingsTarget(captured()),
      b = captureActorSettingsTarget(captured({ ...fresh, id: "fresh-two" }))
    expect(a?.key).not.toBe(b?.key)
    expect(a?.accountKey).toBe(chatDraftOwnerKey(scope))
  })
  it.each([
    {
      label: "server",
      scope: {
        ...scope,
        config: { ...scope.config, serverUrl: "https://another.example" }
      }
    },
    {
      label: "auth mode",
      scope: {
        ...scope,
        config: { ...scope.config, authMode: "multi-user" as const }
      }
    },
    {
      label: "auth source",
      scope: {
        ...scope,
        config: { ...scope.config, authSource: "cookie-session" as const }
      }
    },
    {
      label: "organization",
      scope: { ...scope, config: { ...scope.config, orgId: "another-org" } }
    },
    { label: "user", scope: { ...scope, userId: 2 } },
    {
      label: "single-user credential scope",
      scope: {
        ...scope,
        config: {
          ...scope.config,
          expectedSingleUserApiKeyScope: "another-scope"
        }
      }
    }
  ])(
    "distinguishes captured $label without exposing credentials in the locator",
    ({ scope: other }) => {
      const a = captureActorSettingsTarget(captured()),
        b = captureActorSettingsTarget({ ...captured(), requestScope: other })
      expect(a?.key).not.toBe(b?.key)
      expect(b?.key).not.toContain(other.config.serverUrl)
    }
  )
  it("qualifies matching native H1 conversation and full account hash", () => {
    expect(
      captureActorSettingsTarget({ ...captured(native), ...receipt("native") })
        ?.key
    ).toBe(
      JSON.stringify([
        "native",
        chatDraftOwnerKey(scope),
        "global",
        null,
        "conversation-one"
      ])
    )
  })
  it.each([
    { label: "conversation", address: { ...native, id: "different" } },
    { label: "H1 owner", address: { ...native, ownerKey: "different" } },
    {
      label: "workspace",
      address: {
        ...native,
        scope: { type: "workspace" as const, workspaceId: "workspace-one" }
      }
    }
  ])("refuses a native $label mismatch", ({ address }) => {
    expect(
      captureActorSettingsTarget({ ...captured(address), ...receipt("native") })
    ).toBeNull()
  })
  it("refuses a native view from a different H1 owner", () => {
    const h1 = receipt("native")
    expect(
      captureActorSettingsTarget({
        ...captured(native),
        ...h1,
        view: { ...h1.view, owner_key: "different" }
      })
    ).toBeNull()
  })
  it("refuses a native request account different from the original H1 owner", () => {
    expect(
      captureActorSettingsTarget({
        ...captured(native),
        ...receipt("native"),
        requestScope: { ...scope, userId: 2 }
      })
    ).toBeNull()
  })
  it.each([fresh, native])(
    "refuses unverified backend ownership for $kind scenes",
    (address) => {
      expect(
        captureActorSettingsTarget({
          ...captured(address),
          ...receipt("native"),
          accountIsCurrent: () => false
        })
      ).toBeNull()
    }
  )
  it("qualifies genuine local ownership while backend account verification is unavailable", () => {
    const target = captureActorSettingsTarget({
      ...captured(local),
      ...receipt("local"),
      requestScope: null,
      accountIsCurrent: () => false
    })
    expect(target?.key).toBe(
      JSON.stringify(["local", "local-profile", "conversation-one"])
    )
    expect(target?.isCurrent()).toBe(true)
    expect(target?.accountKey).toBeNull()
    expect(target?.profileIsCurrent()).toBe(false)
  })
  it("refuses a local profile mismatch", () => {
    expect(
      captureActorSettingsTarget({
        ...captured({ ...local, profileId: "different" }),
        ...receipt("local")
      })
    ).toBeNull()
  })
  it("retains the original local H1 lease through later IO", () => {
    let current = true
    const h1 = receipt("local")
    const target = captureActorSettingsTarget({
      ...captured(local),
      ...h1,
      owner: { ...h1.owner, validate_lease: () => current }
    })
    current = false
    expect(target?.isCurrent()).toBe(false)
  })
  it("keeps scene and optional profile leases distinct for unbound local history", () => {
    let accountCurrent = true
    const target = captureActorSettingsTarget({
      ...captured(local),
      ...receipt("local"),
      accountIsCurrent: () => accountCurrent
    })
    accountCurrent = false
    expect(target?.isCurrent()).toBe(true)
    expect(target?.profileIsCurrent()).toBe(false)
  })
  it("refuses a retired source even when the account still matches", () => {
    expect(
      captureActorSettingsTarget({ ...captured(), isCurrent: () => false })
    ).toBeNull()
  })
  it("qualifies character defaults by original verified account and exact card", () => {
    const ownerKey = chatDraftOwnerKey(scope)
    expect(
      captureActorProfileTarget(ownerKey, "card-one", () => true)?.key
    ).toBe(JSON.stringify([ownerKey, "card-one"]))
    expect(captureActorProfileTarget(null, "card-one", () => true)).toBeNull()
    expect(
      captureActorProfileTarget(ownerKey, "card-one", () => false)
    ).toBeNull()
  })
})
