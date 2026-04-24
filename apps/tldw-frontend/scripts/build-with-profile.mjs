import { execFileSync } from "node:child_process"
import { fileURLToPath } from "node:url"

import {
  getCurrentGitBranch,
  normalizeBuildProfile,
  resolveBuildProfile,
} from "../../scripts/resolve-build-profile.mjs"
import { validateNetworkingConfig } from "./validate-networking-config.mjs"

// Quickstart builds on main intentionally default to loopback so the local
// "user-ready" artifact still works against the standard bundled backend path.
const DEFAULT_INTERNAL_API_ORIGIN = "http://127.0.0.1:8000"
const ENTRYPOINT_PATH = fileURLToPath(import.meta.url)

function getBundler(argv = process.argv.slice(2)) {
  for (const arg of argv) {
    if (!arg.startsWith("--bundler=")) {
      continue
    }

    const bundler = arg.slice("--bundler=".length).trim().toLowerCase()
    if (bundler === "turbopack" || bundler === "webpack") {
      return bundler
    }
  }

  throw new Error(
    "Invalid build wrapper usage: pass --bundler=turbopack or --bundler=webpack."
  )
}

function getNextBinary() {
  return process.platform === "win32" ? "next.cmd" : "next"
}

function getBuildArgs(bundler) {
  return ["build", bundler === "turbopack" ? "--turbopack" : "--webpack"]
}

export function shapeWebuiBuildEnv(profile, env = process.env) {
  const nextEnv = { ...env }
  const resolvedProfile = normalizeBuildProfile(profile)
  const internalApiOrigin = String(nextEnv.TLDW_INTERNAL_API_ORIGIN || "").trim()

  if (resolvedProfile === "production") {
    nextEnv.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    delete nextEnv.NEXT_PUBLIC_API_URL
    nextEnv.TLDW_INTERNAL_API_ORIGIN =
      internalApiOrigin || DEFAULT_INTERNAL_API_ORIGIN
  } else {
    nextEnv.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "advanced"
  }

  validateNetworkingConfig(nextEnv)
  return nextEnv
}

export async function runBuildWithProfile({
  argv = process.argv.slice(2),
  env = process.env,
  cwd = process.cwd(),
} = {}) {
  const bundler = getBundler(argv)
  const profile = resolveBuildProfile({
    override: env.TLDW_BUILD_PROFILE,
    branch: getCurrentGitBranch(cwd),
  })
  const shapedEnv = shapeWebuiBuildEnv(profile, env)
  execFileSync(getNextBinary(), getBuildArgs(bundler), {
    cwd,
    env: shapedEnv,
    stdio: "inherit",
  })

  return {
    profile,
    env: shapedEnv,
  }
}

const isEntrypoint = process.argv[1] === ENTRYPOINT_PATH

if (isEntrypoint) {
  try {
    await runBuildWithProfile()
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    console.error(message)
    process.exitCode = 1
  }
}
