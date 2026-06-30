import { describe, expect, it } from "vitest"
import type {
  ChatComposerContext,
  ChatComposerDoc,
  ChatComposerSubmitPayload,
  ChatComposerSurface,
  ChatComposerVariant,
} from "../types"

describe("chat composer shared types", () => {
  it("accepts both surface values", () => {
    const surfaces: ChatComposerSurface[] = ["playground", "sidepanel"]
    expect(surfaces).toHaveLength(2)
  })

  it("accepts all three variant values", () => {
    const variants: ChatComposerVariant[] = ["v1", "v3", "v5"]
    expect(variants).toHaveLength(3)
  })

  it("allows a valid doc attachment", () => {
    const doc: ChatComposerDoc = {
      type: "tab",
      tabId: "tab-1",
      title: "Example",
      url: "https://example.test",
      favIconUrl: "https://example.test/favicon.ico",
    }
    expect(doc.type).toBe("tab")
  })

  it("accepts a shared-core submit payload without surface-specific fields", () => {
    const payload: ChatComposerSubmitPayload = {
      image: "",
      message: "hello",
      docs: [],
    }
    expect(payload.message).toBe("hello")
  })

  it("accepts sidepanel-specific fields on the submit payload", () => {
    const payload: ChatComposerSubmitPayload = {
      image: "",
      message: "hello",
      docs: [],
      uploadedFiles: [],
      requestOverrides: {
        chatMode: "normal",
        selectedModel: null,
        selectedSystemPrompt: null,
        toolChoice: null,
        webSearch: true,
      },
    }
    expect(payload.requestOverrides?.webSearch).toBe(true)
    expect(payload.requestOverrides?.selectedModel).toBeNull()
    expect(payload.requestOverrides?.selectedSystemPrompt).toBeNull()
    expect(payload.requestOverrides?.toolChoice).toBeNull()
  })

  it("accepts playground-specific fields on the submit payload", () => {
    const payload: ChatComposerSubmitPayload = {
      image: "",
      message: "hello",
      docs: [],
      userMessageType: "IMAGE_GENERATION_USER",
      imageGenerationSource: "slash-command",
    }
    expect(payload.imageGenerationSource).toBe("slash-command")
  })

  it("builds a context object with surface + variant", () => {
    const ctx: ChatComposerContext = {
      surface: "playground",
      variant: "v1",
    }
    expect(ctx.surface).toBe("playground")
    expect(ctx.variant).toBe("v1")
  })
})
