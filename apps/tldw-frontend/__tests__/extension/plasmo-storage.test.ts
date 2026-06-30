import { beforeEach, describe, expect, it } from "vitest"

import { Storage } from "@web/extension/shims/plasmo-storage"

describe("plasmo storage web shim", () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it("keeps local and sync areas from clobbering each other", async () => {
    const local = new Storage({ area: "local" })
    const sync = new Storage({ area: "sync" })
    const assistant = {
      kind: "character",
      id: "char-mira",
      name: "Mira"
    }

    await local.set("selectedAssistant", assistant)
    await sync.remove("selectedAssistant")

    expect(await local.get("selectedAssistant")).toEqual(assistant)
    expect(await sync.get("selectedAssistant")).toBeUndefined()
  })
})
