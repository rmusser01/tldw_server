import fs from "node:fs";
import { pathToFileURL } from "node:url";

const resolutionName = /^((?:@[a-z0-9][a-z0-9._~-]*\/)?[a-z0-9][a-z0-9._~-]*)@.+$/;
const lifecycle = new Set(["preinstall", "install", "postinstall", "preprepare", "prepare", "postprepare"]);

export function summarizeBunInstall(log, lock) {
  if (!lock?.packages || typeof lock.packages !== "object" || Array.isArray(lock.packages)) {
    throw new Error("Invalid lockfile package map");
  }
  const allowed = new Set();
  for (const entry of Object.values(lock.packages)) {
    const match = Array.isArray(entry) && typeof entry[0] === "string" && resolutionName.exec(entry[0]);
    if (match && match[1].length <= 214) allowed.add(match[1]);
  }
  const records = new Map();
  for (const raw of log.split(/\r?\n/)) {
    if (raw.length > 1024) continue;
    const line = raw.replace(/\x1b\[[0-9;]*m/g, "");
    if (/[\x00-\x1f\x7f]/.test(line)) continue;
    const match = /^warn: deleting optional dependency '([^']+)' due to failed '([^']+)' script$/.exec(line);
    if (match && allowed.has(match[1]) && lifecycle.has(match[2])) {
      records.set(`${match[1]}:${match[2]}`, {
        classification: "optional_lifecycle_warning", package: match[1], script: match[2],
      });
    }
  }
  // This recognizes one Bun warning path; arbitrary installer output stays private.
  return records.size ? [...records.values()] : [{ classification: "unidentified_install_failure" }];
}

if (import.meta.main) {
  try {
    const lock = (await import(pathToFileURL(process.argv[2]).href, { with: { type: "jsonc" } })).default;
    const log = fs.readFileSync(process.argv[3], "utf8");
    for (const record of summarizeBunInstall(log, lock)) console.log(JSON.stringify(record));
  } catch {
    // The caller emits a fixed diagnostic-unavailable message, preserving Bun's status.
    process.exitCode = 1;
  }
}
