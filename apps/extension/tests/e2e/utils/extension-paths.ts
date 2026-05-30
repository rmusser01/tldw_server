import fs from "node:fs"
import path from "node:path"

const OUTPUT_SUFFIX = `${path.sep}.output${path.sep}chrome-mv3`
const BUILD_SUFFIX = `${path.sep}build${path.sep}chrome-mv3`

const classifyExtensionCandidate = (candidate: string): "custom" | "output" | "build" => {
  const normalized = String(candidate || "").trim()
  if (!normalized) return "custom"
  if (normalized.endsWith(OUTPUT_SUFFIX)) return "output"
  if (normalized.endsWith(BUILD_SUFFIX)) return "build"
  return "custom"
}

export const prioritizeExtensionBuildCandidates = (candidates: string[]): string[] => {
  const buckets: Record<"custom" | "output" | "build", string[]> = {
    custom: [],
    output: [],
    build: []
  }
  const seen = new Set<string>()

  for (const candidate of candidates) {
    const normalized = String(candidate || "").trim()
    if (!normalized || seen.has(normalized)) continue
    seen.add(normalized)
    buckets[classifyExtensionCandidate(normalized)].push(normalized)
  }

  return [...buckets.custom, ...buckets.output, ...buckets.build]
}

const isTruthyEnvValue = (value: string | undefined): boolean =>
  ["1", "true", "yes", "on", "minimal"].includes(
    String(value || "").trim().toLowerCase()
  )

const copyExtensionTreeWithoutLocales = (
  sourceDir: string,
  stagedDir: string
) => {
  for (const entry of fs.readdirSync(sourceDir, { withFileTypes: true })) {
    if (entry.name === "_locales") continue

    const sourcePath = path.join(sourceDir, entry.name)
    const stagedPath = path.join(stagedDir, entry.name)
    fs.cpSync(sourcePath, stagedPath, { recursive: true })
  }
}

export const prepareExtensionLaunchPath = (
  extensionPath: string,
  {
    minimalLocales = isTruthyEnvValue(
      process.env.TLDW_E2E_EXTENSION_MINIMAL_LOCALES ||
        process.env.TLDW_E2E_EXTENSION_LOCALE_MODE
    ),
    rootDir = path.resolve("tmp-playwright-profile", "extension-launch")
  }: {
    minimalLocales?: boolean
    rootDir?: string
  } = {}
): string => {
  if (!minimalLocales) {
    return extensionPath
  }

  if (!fs.existsSync(extensionPath)) {
    throw new Error(`Extension path does not exist: ${extensionPath}`)
  }

  fs.mkdirSync(rootDir, { recursive: true })
  const stagedPath = fs.mkdtempSync(path.join(rootDir, "chrome-mv3-"))

  copyExtensionTreeWithoutLocales(extensionPath, stagedPath)

  const defaultLocaleDir = path.join(stagedPath, "_locales", "en")
  fs.mkdirSync(defaultLocaleDir, { recursive: true })
  fs.writeFileSync(path.join(defaultLocaleDir, "messages.json"), "{}\n", "utf8")

  return stagedPath
}
