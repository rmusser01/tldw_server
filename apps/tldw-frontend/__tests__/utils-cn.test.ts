import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

import { describe, expect, it } from "vitest"

import { cn } from "@web/lib/utils"

const projectRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  ".."
)

function readProjectFile(...segments: string[]) {
  return readFileSync(path.join(projectRoot, ...segments), "utf8")
}

describe("cn", () => {
  it("joins supported class values while dropping falsey values", () => {
    expect(
      cn(
        "base",
        undefined,
        null,
        false,
        true,
        0,
        ["nested", ["deep", { active: true, hidden: false }]],
        { selected: 1, disabled: 0 },
        42,
        BigInt(7)
      )
    ).toBe("base nested deep active selected 42")
  })

  it("keeps Tailwind conflict resolution", () => {
    expect(cn("px-2 text-sm", "px-4", ["text-lg"])).toBe("px-4 text-lg")
  })

  it("does not depend on clsx directly", () => {
    const packageJson = JSON.parse(readProjectFile("package.json")) as {
      dependencies?: Record<string, string>
      devDependencies?: Record<string, string>
    }
    const utilsSource = readProjectFile("lib", "utils.ts")

    expect(packageJson.dependencies).not.toHaveProperty("clsx")
    expect(packageJson.devDependencies).not.toHaveProperty("clsx")
    expect(utilsSource).not.toMatch(/from\s+['"]clsx['"]/)
  })
})
