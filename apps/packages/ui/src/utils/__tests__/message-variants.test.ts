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
