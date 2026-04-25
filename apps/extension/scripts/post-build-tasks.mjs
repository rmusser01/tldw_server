import { execFileSync } from "node:child_process"
import path from "node:path"
import { fileURLToPath } from "node:url"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const projectRoot = path.resolve(__dirname, "..")

function getNodeBinary() {
  return process.execPath
}

export function getPostBuildTasks(targetName, outDir) {
  if (!targetName) {
    throw new Error("Missing targetName for extension post-build tasks.")
  }

  if (!outDir) {
    throw new Error("Missing outDir for extension post-build tasks.")
  }

  const tasks = []

  if (String(targetName).startsWith("firefox-")) {
    tasks.push({
      args: ["--dir", outDir],
      script: "scripts/strip-dangerous-eval.mjs",
    })
  }

  tasks.push({
    args: ["--target", targetName],
    script: "scripts/verify-shared-token-sync.mjs",
  })

  return tasks
}

export function runPostBuildTasks({
  cwd = projectRoot,
  outDir,
  targetName,
} = {}) {
  for (const task of getPostBuildTasks(targetName, outDir)) {
    execFileSync(getNodeBinary(), [task.script, ...task.args], {
      cwd,
      stdio: "inherit",
    })
  }
}

export function getWxtTargetName(browser, manifestVersion) {
  if (!browser || !manifestVersion) {
    throw new Error("Missing WXT browser or manifestVersion for target name resolution.")
  }

  return `${browser}-mv${manifestVersion}`
}
