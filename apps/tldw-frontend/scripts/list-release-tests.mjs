#!/usr/bin/env node

import { spawnSync } from "node:child_process"
import path from "node:path"
import { fileURLToPath, pathToFileURL } from "node:url"

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..")

/** @param {{ workflows?: boolean, env?: Record<string, string | undefined> }} [options] */
export function buildCollectionCommand({ workflows = false, env = process.env } = {}) {
  // Fixed --list invocation: collection placeholders are never execution config.
  // Do not inherit credentials, provider targets, auth modes or reporter paths.
  const hostEnv = Object.fromEntries(
    ["PATH", "HOME", "TMPDIR", "TMP", "TEMP", "SystemRoot", "WINDIR"].flatMap(key =>
      env[key] === undefined ? [] : [[key, env[key]]])
  )
  /** @type {Record<string, string | undefined>} */
  const collectionEnv = {
    ...hostEnv,
    TLDW_WEB_AUTOSTART: "false",
    TLDW_WEB_URL: "http://127.0.0.1:1",
    TLDW_SERVER_URL: "http://127.0.0.1:1",
    TLDW_MOCK_OPENAI_URL: "http://127.0.0.1:1/v1",
    TLDW_E2E_API_KEY: "collection-only-placeholder",
    TLDW_SKILLS_CERT_SKILL_NAME: "collection-only-skill",
    TLDW_SKILLS_CERT_WEB_RESULT: path.join(frontendRoot, "test-results/collection-only-unused.json"),
  }
  return {
    command: process.execPath,
    args: [path.join(frontendRoot, "node_modules/@playwright/test/cli.js"), "test",
      ...(workflows ? ["e2e/workflows"] : []), "--list", "--reporter=json", "--retries=0", "--workers=1"],
    cwd: frontendRoot,
    env: collectionEnv,
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2)
  if (args.length > 1 || (args.length === 1 && args[0] !== "--workflows")) {
    console.error("Usage: node scripts/list-release-tests.mjs [--workflows]")
    process.exitCode = 2
  } else {
    const command = buildCollectionCommand({ workflows: args[0] === "--workflows" })
    const result = spawnSync(command.command, command.args, {
      cwd: command.cwd, env: command.env, stdio: "inherit",
    })
    if (result.error) console.error(result.error.message)
    process.exitCode = result.status ?? 1
  }
}
