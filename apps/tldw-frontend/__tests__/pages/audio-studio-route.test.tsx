import { existsSync, readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const testFileDirectory = dirname(fileURLToPath(import.meta.url))
const audioStudioPagePath = resolve(testFileDirectory, "../../pages/audio-studio.tsx")
const audiobookStudioPagePath = resolve(
  testFileDirectory,
  "../../pages/audiobook-studio.tsx"
)

describe("web audio studio page routes", () => {
  it("exposes a Next.js page shim for Audio Studio", () => {
    expect(existsSync(audioStudioPagePath)).toBe(true)

    const source = readFileSync(audioStudioPagePath, "utf8")

    expect(source).toMatch(/dynamic\(\(\) => import\("@\/routes\/option-audio-studio"\)/)
    expect(source).toMatch(/ssr:\s*false/)
  })

  it("keeps the legacy Audiobook Studio page as a compatibility route", () => {
    expect(existsSync(audiobookStudioPagePath)).toBe(true)

    const source = readFileSync(audiobookStudioPagePath, "utf8")

    expect(source).toMatch(
      /dynamic\(\(\) => import\("@\/routes\/option-audiobook-studio"\)/
    )
    expect(source).toMatch(/ssr:\s*false/)
  })
})
