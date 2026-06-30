import { readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

import {
  aggregateChatSubmitResults,
  chatSubmitFailed,
  chatSubmitSkipped,
  chatSubmitSubmitted,
  isChatSubmitSuccess,
  throwIfChatSubmitUnsuccessful
} from "../chat-action-utils"

const srcDir = resolve(dirname(fileURLToPath(import.meta.url)), "../../..")
const readUiSource = (path: string) =>
  readFileSync(resolve(srcDir, path), "utf8")

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

  it("aggregates compare branch results without masking total failure", () => {
    expect(
      aggregateChatSubmitResults([
        chatSubmitFailed("model-a failed"),
        chatSubmitFailed("model-b failed")
      ])
    ).toEqual(chatSubmitFailed("model-a failed"))
    expect(
      aggregateChatSubmitResults([
        chatSubmitFailed("model-a failed"),
        chatSubmitSubmitted()
      ])
    ).toEqual(chatSubmitSubmitted())
    expect(aggregateChatSubmitResults([chatSubmitSkipped("offline")])).toEqual(
      chatSubmitSkipped("offline")
    )
    expect(aggregateChatSubmitResults([])).toEqual(
      chatSubmitSkipped("No chat submissions completed")
    )
  })

  it("turns unsuccessful submit results into queue-blocking errors", () => {
    expect(() => throwIfChatSubmitUnsuccessful(chatSubmitSubmitted())).not.toThrow()
    expect(() => throwIfChatSubmitUnsuccessful()).not.toThrow()
    expect(() =>
      throwIfChatSubmitUnsuccessful(chatSubmitFailed("provider failed"))
    ).toThrow("provider failed")
    expect(() =>
      throwIfChatSubmitUnsuccessful(chatSubmitSkipped("not ready"))
    ).toThrow("not ready")
  })

  it("requires the shared pipeline to return submitted or failed results", () => {
    const source = readUiSource("hooks/chat-modes/chatModePipeline.ts")

    expect(source).toMatch(/Promise<ChatSubmitResult>/)
    expect(source).toContain("return chatSubmitSubmitted()")
    expect(source).toContain("return chatSubmitFailed(interruptionReason)")
    expect(source).toContain("return chatSubmitSkipped(")
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
    expect(source).toContain("resolveTurnFileRetrievalEnabled")
    expect(source).toMatch(
      /buildChatModeParams\(\{[\s\S]*ragMediaIds:\s*turnRagMediaIds[\s\S]*fileRetrievalEnabled:\s*turnFileRetrievalEnabled[\s\S]*selectedKnowledge:\s*turnSelectedKnowledge/
    )
    expect(source).toContain("shouldUseRagForTurn")
    expect(source).toContain("const characterResult = await characterChatMode")
    expect(source).toContain("aggregateChatSubmitResults(compareResults)")
    expect(source).toContain("return chatSubmitFailed(errorMessage)")
    expect(source).toContain("return chatSubmitSkipped(")
  })

  it("only rolls continue output back into the composer after submitted results", () => {
    const source = readUiSource("hooks/chat/useChatActions.ts")

    expect(source).toMatch(
      /const shouldApplyComposerRollback[\s\S]*continueResult\.status === "submitted"[\s\S]*continueOutputTarget === "composer_input"[\s\S]*if \(shouldApplyComposerRollback\)/
    )
  })
})
