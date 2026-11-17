/**
 * CC-12 (#3101): the improve-prompt menu used `bg-popover` and
 * `text-popover-foreground`, which neither Tailwind config defines, so the
 * menu rendered with a transparent background over the composer. Tailwind
 * silently drops unknown classes, so this guard asks Tailwind itself which
 * colour utilities in the prompt-assist surfaces produce no CSS.
 */
import { readFileSync } from "node:fs"
import { createRequire } from "node:module"
import path from "node:path"
import { fileURLToPath } from "node:url"
import postcss from "postcss"
import tailwindcss from "tailwindcss"
import { describe, expect, it } from "vitest"

const require = createRequire(import.meta.url)
const testDir = path.dirname(fileURLToPath(import.meta.url))
const srcRoot = path.resolve(testDir, "../../../..")
const appsRoot = path.resolve(srcRoot, "../../..")

const GUARDED_FILES = [
  "components/Common/PromptAssist/PromptAssistMenu.tsx",
  "components/Common/PromptAssist/PromptAssistPanel.tsx",
  "components/Common/PromptAssist/PromptReviewSurface.tsx",
  "components/Common/PromptAssist/PromptDiff.tsx",
  "components/Chat/composer/PromptAssistComposerAction.tsx"
]

// Colour-taking utilities; the captured name is checked against the theme.
const COLOR_UTILITY =
  /(?<![\w-])(?:[a-z-]+:)*((?:bg|text|border|ring|outline|fill|stroke|divide|placeholder|from|via|to)-[a-z][a-z0-9-]*(?:\/\d+)?)(?![\w-])/g

const collectCandidates = (source: string) => {
  const candidates = new Set<string>()
  for (const match of source.matchAll(COLOR_UTILITY)) candidates.add(match[1])
  return [...candidates]
}

const generatedSelectors = async (client: string, classes: string[]) => {
  const config = require(path.resolve(appsRoot, client, "tailwind.config.js"))
  const result = await postcss([
    tailwindcss({
      ...config,
      content: [{ raw: `<div class="${classes.join(" ")}"></div>` }]
    })
  ]).process("@tailwind utilities;", { from: undefined })
  const selectors = new Set<string>()
  result.root.walkRules((rule) => {
    for (const selector of rule.selectors) {
      selectors.add(selector.replace(/^\./, "").replace(/\\/g, "").split(/[:\s>]/)[0])
    }
  })
  return selectors
}

describe("prompt-assist colour classes", () => {
  it.each(["tldw-frontend", "extension"])(
    "uses only colour utilities that the %s theme defines",
    async (client) => {
      const candidates = [
        ...new Set(
          GUARDED_FILES.flatMap((file) =>
            collectCandidates(readFileSync(path.join(srcRoot, file), "utf8"))
          )
        )
      ]
      expect(candidates.length).toBeGreaterThan(10)
      const selectors = await generatedSelectors(client, candidates)
      const undefinedClasses = candidates.filter((name) => !selectors.has(name))
      expect(undefinedClasses).toEqual([])
    },
    30_000
  )
})
