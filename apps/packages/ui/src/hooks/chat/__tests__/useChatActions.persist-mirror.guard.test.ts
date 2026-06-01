import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

describe("useChatActions persist mirror guard", () => {
  it("resets assistant persisted state after fallback failures before saving success", () => {
    const source = fs.readFileSync(
      path.resolve(__dirname, "..", "useChatActions.ts"),
      "utf8"
    )

    expect(source).toContain("assistantPersistedToServer = false")
    expect(source).toContain(
      "serverMessagesAlreadyPersisted: assistantPersistedToServer"
    )
    expect(source).not.toContain(
      "serverMessagesAlreadyPersisted: Boolean(activeChatId) && !temporaryChat"
    )
  })

  it("mirrors dynamic UI assistant metadata when saving server chat turns", () => {
    const source = fs.readFileSync(
      path.resolve(__dirname, "..", "useChatActions.ts"),
      "utf8"
    )

    expect(source).toContain("metadata_extra: payload.assistantMetadataExtra")
  })

  it("mirrors user metadata only when present while saving server chat turns", () => {
    const source = fs.readFileSync(
      path.resolve(__dirname, "..", "useChatActions.ts"),
      "utf8"
    )

    expect(source).toContain("metadata_extra: payload.userMetadataExtra")
    expect(source).toContain("...(payload.userMetadataExtra")
  })
})
