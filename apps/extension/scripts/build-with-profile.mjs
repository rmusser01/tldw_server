import { execFileSync } from "node:child_process"
import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

import {
  getCurrentGitBranch,
  normalizeBuildProfile,
  resolveBuildProfile,
} from "../../scripts/resolve-build-profile.mjs"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const projectRoot = path.resolve(__dirname, "..")
const outputRoot = path.join(projectRoot, ".output")
const buildRoot = path.join(projectRoot, "build")

const browserConfigs = {
  chrome: {
    browserArg: null,
    targetEnv: "chrome",
    targetName: "chrome-mv3",
  },
  edge: {
    browserArg: "edge",
    targetEnv: "chrome",
    targetName: "edge-mv3",
  },
  firefox: {
    browserArg: "firefox",
    targetEnv: "firefox",
    targetName: "firefox-mv2",
  },
}

function hasBrowserConfig(browser) {
  return Object.hasOwn(browserConfigs, browser)
}

function getWxtBinary() {
  return process.platform === "win32" ? "wxt.cmd" : "wxt"
}

function parseBrowser(argv = process.argv.slice(2)) {
  const browserArg = argv.find((arg) => arg.startsWith("--browser="))
  if (!browserArg) {
    throw new Error(
      'Missing required "--browser=" argument. Expected "chrome", "firefox", or "edge".'
    )
  }

  const browser = browserArg.slice("--browser=".length).trim().toLowerCase()
  if (!hasBrowserConfig(browser)) {
    throw new Error(
      `Unsupported browser "${browser}". Expected "chrome", "firefox", or "edge".`
    )
  }

  return browser
}

function resolveBrowserConfig(browser) {
  return browserConfigs[browser]
}

function runCommand(command, args, options = {}) {
  execFileSync(command, args, {
    cwd: projectRoot,
    stdio: "inherit",
    ...options,
  })
}

function resolveCanonicalArtifactDir(targetName) {
  const outputDir = path.join(outputRoot, targetName)
  if (fs.existsSync(outputDir)) {
    return outputDir
  }

  const buildDir = path.join(buildRoot, targetName)
  if (fs.existsSync(buildDir)) {
    return buildDir
  }

  return outputDir
}

export function getExportedArtifactDir(targetName, profile) {
  const normalizedProfile = normalizeBuildProfile(profile)
  const dirName =
    normalizedProfile === "development" ? `${targetName}-dev` : targetName
  return path.join("build", dirName)
}

export function getExportedZipName(fileName, profile) {
  const normalized = String(fileName || "").trim() || "extension"
  const ext = path.extname(normalized) || ".zip"
  const baseName = path.extname(normalized)
    ? path.basename(normalized, path.extname(normalized))
    : normalized

  if (normalizeBuildProfile(profile) === "development") {
    return `${baseName}-dev${ext}`
  }

  return path.extname(normalized) ? normalized : `${baseName}${ext}`
}

function syncExportedArtifactDir(sourceDir, destinationDir) {
  if (!fs.existsSync(sourceDir)) {
    throw new Error(`Extension build directory not found: ${sourceDir}`)
  }

  if (path.resolve(sourceDir) === path.resolve(destinationDir)) {
    return destinationDir
  }

  fs.rmSync(destinationDir, { recursive: true, force: true })
  fs.mkdirSync(path.dirname(destinationDir), { recursive: true })
  fs.cpSync(sourceDir, destinationDir, { recursive: true })
  return destinationDir
}

export function buildWithProfile({
  argv = process.argv.slice(2),
  cwd = projectRoot,
  env = process.env,
} = {}) {
  const browser = parseBrowser(argv)
  const browserConfig = resolveBrowserConfig(browser)
  const profile = resolveBuildProfile({
    override: env.TLDW_BUILD_PROFILE,
    branch: getCurrentGitBranch(cwd),
  })
  const wxtArgs = ["build"]

  if (browserConfig.browserArg) {
    wxtArgs.push("-b", browserConfig.browserArg)
  }

  runCommand(getWxtBinary(), wxtArgs, {
    cwd,
    env: {
      ...env,
      TARGET: browserConfig.targetEnv,
    },
  })

  const canonicalDir = resolveCanonicalArtifactDir(browserConfig.targetName)

  const exportedDir = path.join(
    projectRoot,
    getExportedArtifactDir(browserConfig.targetName, profile)
  )

  syncExportedArtifactDir(canonicalDir, exportedDir)

  return {
    browser,
    browserConfig,
    canonicalDir,
    exportedDir,
    profile,
  }
}

const isEntrypoint = process.argv[1] === fileURLToPath(import.meta.url)

if (isEntrypoint) {
  try {
    buildWithProfile()
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    console.error(message)
    process.exitCode = 1
  }
}
