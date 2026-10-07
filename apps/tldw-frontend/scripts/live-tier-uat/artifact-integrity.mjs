/**
 * Exact byte/name inventory for an owned, quiescent source or build directory.
 * No paths are ignored. This is not an atomic filesystem snapshot, a signature,
 * build provenance, clean-install evidence, or proof of what a process loaded.
 * Runtime binding and before/after execution checks belong to the caller.
 */
import { createHash } from "node:crypto"
import { lstatSync, readdirSync, readFileSync, readlinkSync, realpathSync, writeFileSync } from "node:fs"
import path from "node:path"
import { isDeepStrictEqual } from "node:util"

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex")
const inside = (root, target) => {
  const relative = path.relative(root, target)
  return relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative)
}
const exactUtf8 = (bytes) => {
  const value = bytes.toString("utf8")
  if (!Buffer.from(value).equals(bytes)) throw new Error("Artifact paths must be valid UTF-8")
  return value
}
const unchanged = (before, after) =>
  ["dev", "ino", "mode", "size", "mtimeMs", "ctimeMs"].every(key => before[key] === after[key])

/** Inventory every entry and every in-root symlink referent; reject unsafe trees. */
export function createArtifactSeal(directory) {
  const root = realpathSync(directory)
  if (!lstatSync(root).isDirectory()) throw new Error("Artifact root must be a directory")
  const entries = new Map()
  const active = new Set()
  const relative = (absolute) => path.relative(root, absolute).split(path.sep).join("/") || "."
  const requireInside = (absolute) => {
    if (!inside(root, absolute)) throw new Error(`Artifact symlink escapes outside root: ${absolute}`)
  }
  const visit = (absolute) => {
    requireInside(absolute)
    if (active.has(absolute)) throw new Error(`Artifact symlink cycle: ${relative(absolute)}`)
    if (entries.has(absolute)) return entries.get(absolute)
    active.add(absolute)
    try {
      const stat = lstatSync(absolute)
      const base = { path: relative(absolute), mode: stat.mode & 0o7777 }
      let entry
      if (stat.isSymbolicLink()) {
        const targetBytes = readlinkSync(absolute, { encoding: "buffer" })
        const target = exactUtf8(targetBytes)
        requireInside(path.resolve(path.dirname(absolute), target))
        const resolved = realpathSync(absolute)
        requireInside(resolved)
        const referent = visit(resolved)
        const link = {
          target, bytes: targetBytes.length, targetSha256: sha256(targetBytes),
          resolvedPath: relative(resolved), referentSha256: referent.sha256,
        }
        entry = { ...base, type: "symlink", ...link, sha256: sha256(JSON.stringify(link)) }
      } else if (stat.isDirectory()) {
        const names = readdirSync(absolute, { encoding: "buffer" }).sort(Buffer.compare).map(exactUtf8)
        const children = names.map(name => visit(path.join(absolute, name)))
        entry = { ...base, type: "directory", sha256: sha256(JSON.stringify(children)) }
      } else if (stat.isFile()) {
        requireInside(realpathSync(absolute))
        // ponytail: one file is buffered at a time; stream hashing if artifacts exceed memory.
        const bytes = readFileSync(absolute)
        entry = { ...base, type: "file", bytes: bytes.length, sha256: sha256(bytes) }
      } else {
        throw new Error(`Unsupported artifact entry type: ${base.path}`)
      }
      if (!unchanged(stat, lstatSync(absolute))) throw new Error(`Artifact changed during inventory: ${base.path}`)
      entries.set(absolute, entry)
      return entry
    } finally {
      active.delete(absolute)
    }
  }
  visit(root)
  const inventory = [...entries.values()].sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0)
  return {
    schemaVersion: 1, kind: "artifact-integrity-only", certifiesRelease: false,
    root, sha256: sha256(JSON.stringify(inventory)), entries: inventory,
  }
}

/** Recompute the actual directory inventory, including added and removed paths. */
export function verifyArtifactSeal(directory, seal) {
  const root = realpathSync(directory)
  if (seal?.root !== root) throw new Error("Artifact root does not match seal")
  const actual = createArtifactSeal(root)
  if (!isDeepStrictEqual(seal, actual)) throw new Error("Artifact inventory changed or seal is invalid")
  return actual
}

/** Write a new external receipt exclusively; preserve every existing receipt. */
export function writeArtifactSeal(directory, receiptPath) {
  const root = realpathSync(directory)
  const output = path.resolve(receiptPath)
  const resolvedOutput = path.join(realpathSync(path.dirname(output)), path.basename(output))
  if (inside(root, resolvedOutput)) throw new Error("Artifact receipt must be outside the inventoried root")
  const seal = createArtifactSeal(root)
  writeFileSync(output, `${JSON.stringify(seal, null, 2)}\n`, { flag: "wx", mode: 0o600 })
  return seal
}
