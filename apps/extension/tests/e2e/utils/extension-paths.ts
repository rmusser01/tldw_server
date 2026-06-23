import fs from "node:fs"
import path from "node:path"

const OUTPUT_SUFFIX = `${path.sep}.output${path.sep}chrome-mv3`
const BUILD_SUFFIX = `${path.sep}build${path.sep}chrome-mv3`
const DEFAULT_EXTENSION_LOCALE = "en"
const LOCALE_NAME_PATTERN = /^[A-Za-z0-9_@-]+$/
const E2E_EXTENSION_MANIFEST_KEY =
  "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAjI1q+ZCGeQEsFkXz8Jcx9BHxpWcxr4egilGW2LKpyDcxbd+2id2k0WtauiWSS+eBfvJWRnonnIjZQ/6jkNbN41z+G6Wp5HzHJaGHB609GO4LWW5kVkPo0h+KkSSEVjoXTyRQZO3ViwDbne3gqHVJmnKGWV+Tz6X2se3GwCah3I0AG2290/E4aweSV6OG/SRD15MCiDTImSCNa7WXhMQtqN61o+b8MGr3t5eN3E2UCKMFYAFH017EuRQ46vn8q29O7ATaEwHnB0U/7g9zyi3OKhCU5bI9XhZNoRH/iZqOajz5vVu4Pbq6Wq0Vu2Y1nHIjOQi4XADuUrd4ZFyQWkDFcwIDAQAB"

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

const resolveManifestDefaultLocale = (extensionPath: string): string => {
  try {
    const manifest = JSON.parse(
      fs.readFileSync(path.join(extensionPath, "manifest.json"), "utf8")
    ) as { default_locale?: unknown }
    const defaultLocale =
      typeof manifest.default_locale === "string"
        ? manifest.default_locale.trim()
        : ""
    if (defaultLocale && LOCALE_NAME_PATTERN.test(defaultLocale)) {
      return defaultLocale
    }
  } catch {
    return DEFAULT_EXTENSION_LOCALE
  }

  return DEFAULT_EXTENSION_LOCALE
}

const copyDefaultLocaleCatalog = (
  extensionPath: string,
  stagedPath: string,
  defaultLocale: string
) => {
  const sourceDefaultLocaleDir = path.join(
    extensionPath,
    "_locales",
    defaultLocale
  )
  const stagedDefaultLocaleDir = path.join(stagedPath, "_locales", defaultLocale)

  if (fs.existsSync(sourceDefaultLocaleDir)) {
    fs.cpSync(sourceDefaultLocaleDir, stagedDefaultLocaleDir, {
      recursive: true
    })
    return
  }

  fs.mkdirSync(stagedDefaultLocaleDir, { recursive: true })
  fs.writeFileSync(
    path.join(stagedDefaultLocaleDir, "messages.json"),
    "{}\n",
    "utf8"
  )
}

const ensureDeterministicManifestKey = (stagedPath: string) => {
  const manifestPath = path.join(stagedPath, "manifest.json")

  try {
    const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8")) as {
      key?: unknown
    }

    if (typeof manifest.key !== "string" || !manifest.key.trim()) {
      manifest.key = E2E_EXTENSION_MANIFEST_KEY
      fs.writeFileSync(manifestPath, JSON.stringify(manifest), "utf8")
    }
  } catch {
    // The caller already validated the build enough to launch. Leave malformed
    // manifests to Chrome/Playwright so the E2E failure reports the real cause.
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
  ensureDeterministicManifestKey(stagedPath)

  copyDefaultLocaleCatalog(
    extensionPath,
    stagedPath,
    resolveManifestDefaultLocale(extensionPath)
  )

  return stagedPath
}
