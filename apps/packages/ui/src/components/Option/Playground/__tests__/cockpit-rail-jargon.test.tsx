// @vitest-environment jsdom
// CO-04 (#3113): the rails' leftover jargon. "Media scope" becomes "media
// item", the unused jargon keys go, and the rail tabs say "Context" and
// "Runtime" (decision [1] keeps the redesign's names).
import React from "react"
import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { PlaygroundContextRail } from "../PlaygroundContextRail"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string, vars?: Record<string, unknown>) =>
      (fallback ?? _key).replace(/\{\{(\w+)\}\}/g, (token, name) =>
        vars && name in vars ? String(vars[name]) : token
      )
  })
}))

const testDir = path.dirname(fileURLToPath(import.meta.url))
const srcRoot = path.resolve(testDir, "../../../../")
const read = (file: string) =>
  readFileSync(path.resolve(srcRoot, file), "utf8")

describe("cockpit rail copy", () => {
  it("counts media items, not media scopes", () => {
    const onClearMedia = vi.fn()
    render(
      <PlaygroundContextRail
        hasContext
        contextSummary={[]}
        sessionLabel="Local chat"
        historyLinked={false}
        temporaryChat={false}
        onToggleTemporaryChat={vi.fn()}
        contextCounts={{ files: 0, knowledge: 0, media: 2, research: 0 }}
        onOpenSearchContext={vi.fn()}
        onClearMedia={onClearMedia}
      />
    )

    expect(screen.getByText("2 media items")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Clear media items" }))
    expect(onClearMedia).toHaveBeenCalledTimes(1)
    expect(screen.queryByText(/scope/i)).not.toBeInTheDocument()
  })

  it("keeps 'media scope' out of the English copy and the rail fallbacks", () => {
    const locale = read("assets/locale/en/playground.json")
    expect(locale).not.toMatch(/media scope/i)
    for (const file of [
      "components/Option/Playground/PlaygroundContextRail.tsx",
      "components/Option/Playground/Playground.tsx",
      "components/Option/Playground/PlaygroundCockpitShell.tsx"
    ]) {
      const source = read(file)
      expect(source, file).not.toMatch(/"[^"]*media scope[^"]*"/i)
      expect(source, file).not.toMatch(/"(Context|Runtime) rail"/)
    }
  })

  it("drops the unused jargon keys", () => {
    const cockpit = JSON.parse(read("assets/locale/en/playground.json")).cockpit
    for (const key of ["turnIdle", "providerRoute", "scopedSettings"]) {
      expect(cockpit, key).not.toHaveProperty(key)
    }
    const mirror = JSON.parse(read("public/_locales/en/playground.json"))
    for (const key of [
      "cockpit_turnIdle",
      "cockpit_providerRoute",
      "cockpit_scopedSettings"
    ]) {
      expect(mirror, key).not.toHaveProperty(key)
    }
  })
})
