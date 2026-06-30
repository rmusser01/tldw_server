import { beforeEach, describe, expect, it, vi } from "vitest"
import { resolveContextReadiness } from "../conversation-context-utils"
import type { ConversationContextComposition } from "@/types/conversation-context"

const registryLabels = vi.hoisted(() => ({
  ready: "Ready via registry",
  blocked: "Blocked via registry"
}))

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()

  return {
    ...actual,
    getDesignSystemState: vi.fn(
      (key: Parameters<typeof actual.getDesignSystemState>[0]) => {
        const state = actual.getDesignSystemState(key)

        return {
          ...state,
          label:
            key === "ready"
              ? registryLabels.ready
              : key === "blocked"
                ? registryLabels.blocked
                : state.label
        }
      }
    )
  }
})

const buildComposition = (
  readiness: ConversationContextComposition["readiness"]
): ConversationContextComposition =>
  ({
    readiness
  }) as ConversationContextComposition

describe("resolveContextReadiness", () => {
  beforeEach(() => {
    registryLabels.ready = "Ready via registry"
    registryLabels.blocked = "Blocked via registry"
  })

  it("uses design-system labels for canonical ready and blocked readiness", () => {
    expect(
      resolveContextReadiness({
        composition: buildComposition("ready"),
        status: "ready"
      })
    ).toMatchObject({
      readiness: "ready",
      label: "Ready via registry",
      tone: "ready"
    })

    expect(
      resolveContextReadiness({
        composition: buildComposition("blocked"),
        status: "ready"
      })
    ).toMatchObject({
      readiness: "blocked",
      label: "Blocked via registry",
      tone: "blocked"
    })
  })

  it("reads canonical readiness labels at resolution time", () => {
    registryLabels.ready = "Ready after registry update"
    registryLabels.blocked = "Blocked after registry update"

    expect(
      resolveContextReadiness({
        composition: buildComposition("ready"),
        status: "ready"
      })
    ).toMatchObject({
      readiness: "ready",
      label: "Ready after registry update",
      tone: "ready"
    })

    expect(
      resolveContextReadiness({
        composition: buildComposition("blocked"),
        status: "ready"
      })
    ).toMatchObject({
      readiness: "blocked",
      label: "Blocked after registry update",
      tone: "blocked"
    })
  })
})
