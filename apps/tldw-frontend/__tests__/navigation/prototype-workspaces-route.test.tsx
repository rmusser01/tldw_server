import { existsSync, readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const testFileDirectory = dirname(fileURLToPath(import.meta.url))
const pagePath = resolve(testFileDirectory, "../../pages/prototype-workspaces.tsx")

describe("web prototype workspaces page route", () => {
  it("exposes a Next.js page shim for prototype workspaces", () => {
    expect(existsSync(pagePath)).toBe(true)

    const source = readFileSync(pagePath, "utf8")

    expect(source).toMatch(
      /dynamic\(\(\) => import\("@\/routes\/option-prototype-workspaces"\)/
    )
    expect(source).toMatch(/ssr:\s*false/)
  })
})
