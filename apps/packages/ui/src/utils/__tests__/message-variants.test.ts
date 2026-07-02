import { describe, expect, it } from "vitest"
import type { Message } from "@/store/option"
import {
  applyVariantToMessage,
  buildMessageVariant,
  updateActiveVariant
} from "../message-variants"

const createMessage = (overrides: Partial<Message> = {}): Message => ({
  isBot: true,
  name: "Assistant",
  role: "assistant",
  message: "root = <Card />",
  sources: [],
  images: [],
  ...overrides
})

describe("message variant metadata", () => {
  it("copies dynamic UI metadata when building a variant", () => {
    const metadataExtra = {
      dynamic_ui: {
        renderer: "openui" as const,
        version: "v1" as const,
        source: "root = <Card />"
      },
      trace_id: "trace-1"
    }

    expect(buildMessageVariant(createMessage({ metadataExtra }))).toMatchObject({
      metadataExtra
    })
  })

  it("applies the active variant metadata instead of leaving stale metadata", () => {
    const message = createMessage({
      metadataExtra: {
        dynamic_ui: {
          renderer: "openui",
          version: "v1",
          source: "root = <OldCard />"
        },
        trace_id: "old-trace"
      }
    })

    const next = applyVariantToMessage(
      message,
      {
        id: "variant-2",
        message: "Plain text answer",
        sources: [],
        images: []
      },
      1
    )

    expect(next.metadataExtra).toBeUndefined()
  })

  it("does not inherit the prior variant's serverMessageId for an unpersisted variant", () => {
    const message = createMessage({
      serverMessageId: "server-1",
      serverMessageVersion: 3
    })

    const next = applyVariantToMessage(
      message,
      {
        // A freshly-regenerated variant that has not been persisted yet.
        id: "variant-2",
        message: "regenerated answer",
        sources: [],
        images: []
      },
      1
    )

    expect(next.serverMessageId).toBeUndefined()
    expect(next.serverMessageVersion).toBeUndefined()
  })

  it("adopts a persisted variant's own server identity when swiping", () => {
    const message = createMessage({
      serverMessageId: "server-1",
      serverMessageVersion: 3
    })

    const next = applyVariantToMessage(
      message,
      {
        id: "variant-2",
        message: "persisted answer",
        serverMessageId: "server-2",
        serverMessageVersion: 5,
        sources: [],
        images: []
      },
      1
    )

    expect(next.serverMessageId).toBe("server-2")
    expect(next.serverMessageVersion).toBe(5)
  })

  it("stores metadata updates on the active variant", () => {
    const message = createMessage({
      activeVariantIndex: 0,
      variants: [
        {
          id: "variant-1",
          message: "root = <Card />",
          sources: [],
          images: []
        }
      ]
    })
    const metadataExtra = {
      dynamic_ui: {
        renderer: "openui" as const,
        version: "v1" as const,
        source: "root = <Card />"
      }
    }

    const next = updateActiveVariant(message, { metadataExtra })

    expect(next.variants?.[0].metadataExtra).toEqual(metadataExtra)
  })
})
