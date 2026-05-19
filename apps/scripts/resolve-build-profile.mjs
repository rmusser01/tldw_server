import { execFileSync } from "node:child_process"

export function normalizeBuildProfile(value) {
  const normalized = String(value || "").trim().toLowerCase()
  return normalized === "production" ? "production" : "development"
}

export function resolveBuildProfile({ override, branch } = {}) {
  const explicit = String(override || "").trim().toLowerCase()

  // Only the two supported override values short-circuit branch detection.
  if (explicit === "production" || explicit === "development") {
    return explicit
  }

  const currentBranch = String(branch || "").trim()
  if (currentBranch === "main") {
    return "production"
  }

  return "development"
}

export function getCurrentGitBranch(cwd = process.cwd()) {
  try {
    const branch = execFileSync("git", ["rev-parse", "--abbrev-ref", "HEAD"], {
      cwd,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim()

    return branch === "HEAD" ? "" : branch
  } catch {
    return ""
  }
}
