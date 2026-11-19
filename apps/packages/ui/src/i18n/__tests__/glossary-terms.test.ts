import { readdirSync, readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

// XP-13 (#3101, decision 7): the review's glossary (§10.2). The persona's
// artwork is an "Avatar", not a "Buddy", and the product is "tldw", not
// "tldw Assistant". This guards the English copy in both locale formats.

const SRC = path.resolve(__dirname, "../..")
const LOCALE_DIRS = ["assets/locale/en", "public/_locales/en"]

// The extension's manifest name and store description stay as published
// until the store listing is renamed (an owner decision, not copy).
const MANIFEST_KEYS = new Set(["extName", "extDescription", "extName.message", "extDescription.message"])

type Entry = { file: string; key: string; value: string }

const collect = (value: unknown, key: string, file: string, out: Entry[]) => {
  if (typeof value === "string") {
    out.push({ file, key, value })
    return
  }
  if (value && typeof value === "object") {
    for (const [child, next] of Object.entries(value as Record<string, unknown>)) {
      collect(next, key ? `${key}.${child}` : child, file, out)
    }
  }
}

const englishEntries = (): Entry[] => {
  const out: Entry[] = []
  for (const dir of LOCALE_DIRS) {
    for (const name of readdirSync(path.join(SRC, dir))) {
      if (!name.endsWith(".json")) continue
      const file = `${dir}/${name}`
      collect(JSON.parse(readFileSync(path.join(SRC, file), "utf8")), "", file, out)
    }
  }
  return out
}

const offending = (pattern: RegExp) =>
  englishEntries()
    .filter((entry) => !MANIFEST_KEYS.has(entry.key) && pattern.test(entry.value))
    .map((entry) => `${entry.file} ${entry.key}: ${entry.value}`)

describe("English copy follows the glossary (XP-13)", () => {
  it("calls the persona's artwork an avatar, never a Buddy", () => {
    expect(offending(/\bbudd(y|ies)\b/i)).toEqual([])
  })

  it("names the product tldw, not tldw Assistant", () => {
    expect(offending(/tldw Assistant/)).toEqual([])
  })
})
