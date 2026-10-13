import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

import {
  CHAT_PERSISTENCE_KINDS,
  getChatPersistenceCopy
} from "@/utils/chat-persistence-status"

// CS-03 / XS-05 (#3104): persistence copy is plain language and tells the user
// where the chat is saved. Locale JSON overrides inline t() defaults, so the
// shipped English strings, the extension mirrors and the inline defaults must
// all agree.
const srcRoot = path.resolve(__dirname, "../../../../")
const readJson = (relative: string) =>
  JSON.parse(readFileSync(path.resolve(srcRoot, relative), "utf8"))

const assetsPersistence = readJson("assets/locale/en/playground.json").composer
  .persistence as Record<string, string>
const publicPlayground = readJson("public/_locales/en/playground.json") as Record<
  string,
  { message: string }
>
const publicMessages = readJson("public/_locales/en/messages.json") as Record<
  string,
  { message: string }
>

const expectedCopy: Record<string, string> = {
  local: "Saved on this device only. This chat is not on your tldw server.",
  localPill: "Saved on this device",
  server: "Saved on your tldw server and on this device.",
  serverPill: "Saved on server",
  serverSaving:
    "Saved on this device. Waiting for your tldw server to confirm the latest changes.",
  serverSavingPill: "Saving to server…",
  serverFailed:
    "Saved on this device. Your tldw server didn't confirm the latest changes.",
  serverFailedPill: "Couldn't save to server",
  serverInlineTitle: "Saved on your server",
  serverInlineBody:
    "This chat is saved on your tldw server and on this device, so you can reopen it from server history, keep a long-term record, and analyze it alongside other conversations."
}

describe("chat persistence English copy", () => {
  it.each(Object.entries(expectedCopy))(
    "ships plain copy for composer.persistence.%s",
    (key, value) => {
      expect(assetsPersistence[key]).toBe(value)
      expect(publicPlayground[`composer_persistence_${key}`]?.message).toBe(value)
      expect(
        publicMessages[`playground_composer_persistence_${key}`]?.message
      ).toBe(value)
    }
  )

  it("drops the 'Locally + Server' jargon from every English persistence string", () => {
    for (const value of Object.values(assetsPersistence)) {
      expect(value).not.toMatch(/locally\s*\+|locally\+server/i)
    }
  })

  it("keeps inline defaults identical to the shipped English locale", () => {
    const fallbackOnly = (_key: string, defaultValue?: unknown) =>
      String(defaultValue)
    const fromLocale = (key: string, defaultValue?: unknown) => {
      const leaf = key.replace("playground:composer.persistence.", "")
      return assetsPersistence[leaf] ?? String(defaultValue)
    }
    for (const kind of CHAT_PERSISTENCE_KINDS) {
      expect(getChatPersistenceCopy(fallbackOnly, kind)).toEqual(
        getChatPersistenceCopy(fromLocale, kind)
      )
    }
  })
})
