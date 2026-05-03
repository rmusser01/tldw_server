import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"

import {
  chatSubmitFailed,
  chatSubmitSkipped,
  chatSubmitSubmitted,
  isChatSubmitSuccess
} from "../chat-action-utils"

const readUiSource = (path: string) =>
  readFileSync(resolve(process.cwd(), "../packages/ui/src", path), "utf8")

describe("chat submit result contract", () => {
  it("provides explicit submit result helpers", () => {
    expect(chatSubmitSubmitted()).toEqual({ status: "submitted" })
    expect(chatSubmitFailed("boom")).toEqual({
      status: "failed",
      errorMessage: "boom"
    })
    expect(chatSubmitSkipped("empty")).toEqual({
      status: "skipped",
      reason: "empty"
    })
    expect(isChatSubmitSuccess(chatSubmitSubmitted())).toBe(true)
    expect(isChatSubmitSuccess(chatSubmitFailed("boom"))).toBe(false)
    expect(isChatSubmitSuccess(chatSubmitSkipped("empty"))).toBe(false)
  })

  it("requires the shared pipeline to return submitted or failed results", () => {
    const source = readUiSource("hooks/chat-modes/chatModePipeline.ts")

    expect(source).toMatch(/Promise<ChatSubmitResult>/)
    expect(source).toContain("return chatSubmitSubmitted()")
    expect(source).toContain("return chatSubmitFailed(interruptionReason)")
  })

  it("requires chat mode wrappers to return pipeline results", () => {
    const wrapperPaths = [
      "hooks/chat-modes/normalChatMode.ts",
      "hooks/chat-modes/ragMode.ts",
      "hooks/chat-modes/documentChatMode.ts",
      "hooks/chat-modes/tabChatMode.ts",
      "hooks/chat-modes/continueChatMode.ts"
    ]

    for (const path of wrapperPaths) {
      const source = readUiSource(path)
      expect(source, path).toMatch(/Promise<ChatSubmitResult>/)
      expect(source, path).toContain("return runChatPipeline(")
    }
  })

  it("requires useChatActions to return explicit submit results", () => {
    const source = readUiSource("hooks/chat/useChatActions.ts")

    expect(source).toMatch(/Promise<ChatSubmitResult>/)
    expect(source).toContain("resolveTurnRagMediaIds")
    expect(source).toContain("shouldUseRagForTurn")
    expect(source).toContain("return chatSubmitFailed(errorMessage)")
    expect(source).toContain("return chatSubmitSkipped(")
  })
})
