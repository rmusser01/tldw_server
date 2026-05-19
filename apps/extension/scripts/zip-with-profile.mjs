import { execFileSync } from "node:child_process"
import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

import {
  getCurrentGitBranch,
  resolveBuildProfile,
} from "../../scripts/resolve-build-profile.mjs"
import { getExportedZipName } from "./build-with-profile.mjs"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const projectRoot = path.resolve(__dirname, "..")
const outputRoot = path.join(projectRoot, ".output")
const buildRoot = path.join(projectRoot, "build")

const browserConfigs = {
  chrome: {
    browserArg: null,
    targetEnv: "chrome",
  },
  firefox: {
    browserArg: "firefox",
    targetEnv: "firefox",
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
      'Missing required "--browser=" argument. Expected "chrome" or "firefox".'
    )
  }

  const browser = browserArg.slice("--browser=".length).trim().toLowerCase()
  if (!hasBrowserConfig(browser)) {
    throw new Error(
      `Unsupported browser "${browser}". Expected "chrome" or "firefox".`
    )
  }

  return browser
}

function findZipArtifacts(rootDir) {
  if (!fs.existsSync(rootDir)) {
    return []
  }

  const entries = fs.readdirSync(rootDir, { withFileTypes: true })
  const files = []

  for (const entry of entries) {
    const fullPath = path.join(rootDir, entry.name)
    if (entry.isDirectory()) {
      files.push(...findZipArtifacts(fullPath))
      continue
    }

    if (entry.isFile() && entry.name.endsWith(".zip")) {
      files.push(fullPath)
    }
  }

  return files
}

export function getZipArtifactSearchRoots(rootDir = projectRoot) {
  return [path.join(rootDir, ".output"), path.join(rootDir, "build")]
}

export function detectCreatedZipArtifact(searchRoots, beforeMtime) {
  const after = searchRoots.flatMap((rootDir) => findZipArtifacts(rootDir))
  return (
    after.find((filePath) => !beforeMtime.has(filePath)) ||
    after
      .filter(
        (filePath) =>
          fs.statSync(filePath).mtimeMs > (beforeMtime.get(filePath) ?? -Infinity)
      )
      .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs)[0]
  )
}

export function finalizeZipArtifact(createdZip, profile) {
  const desiredName = getExportedZipName(path.basename(createdZip), profile)
  const desiredPath = path.join(path.dirname(createdZip), desiredName)

  if (path.basename(createdZip) === desiredName) {
    return createdZip
  }

  fs.rmSync(desiredPath, { force: true })
  fs.renameSync(createdZip, desiredPath)
  return desiredPath
}

export function zipWithProfile({
  argv = process.argv.slice(2),
  cwd = projectRoot,
  env = process.env,
} = {}) {
  const browser = parseBrowser(argv)
  const browserConfig = browserConfigs[browser]
  const profile = resolveBuildProfile({
    override: env.TLDW_BUILD_PROFILE,
    branch: getCurrentGitBranch(cwd),
  })
  const searchRoots = getZipArtifactSearchRoots(cwd)
  const beforeMtime = new Map(
    searchRoots.flatMap((rootDir) =>
      findZipArtifacts(rootDir).map((filePath) => [
        filePath,
        fs.statSync(filePath).mtimeMs,
      ])
    )
  )
  const wxtArgs = ["zip"]

  if (browserConfig.browserArg) {
    wxtArgs.push("-b", browserConfig.browserArg)
  }

  execFileSync(getWxtBinary(), wxtArgs, {
    cwd,
    env: {
      ...env,
      TARGET: browserConfig.targetEnv,
    },
    stdio: "inherit",
  })

  const createdZip = detectCreatedZipArtifact(searchRoots, beforeMtime)

  if (!createdZip) {
    throw new Error(
      `No zip artifact found under ${searchRoots.join(" or ")}`
    )
  }

  return finalizeZipArtifact(createdZip, profile)
}

const isEntrypoint = process.argv[1] === fileURLToPath(import.meta.url)

if (isEntrypoint) {
  try {
    zipWithProfile()
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    console.error(message)
    process.exitCode = 1
  }
}
