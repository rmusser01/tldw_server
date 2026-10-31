import { spawnSync } from "node:child_process"
import { createHash } from "node:crypto"
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, unlinkSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { createArtifactSeal, verifyArtifactSeal, writeArtifactSeal } from "../live-tier-uat/artifact-integrity.mjs"

const cli = resolve(dirname(fileURLToPath(import.meta.url)), "../assert-release-uat.mjs")
const sha256 = (text: string) => createHash("sha256").update(text).digest("hex")
let temporary: string
let root: string
const runCli = (...args: string[]) => spawnSync(process.execPath, [cli, ...args], { encoding: "utf8" })

beforeEach(() => {
  temporary = mkdtempSync(join(tmpdir(), "uat392-integrity-"))
  root = join(temporary, "artifact")
  mkdirSync(root)
  mkdirSync(join(root, "static"))
  writeFileSync(join(root, "static/app.js"), "original")
})
afterEach(() => rmSync(temporary, { recursive: true, force: true }))

describe("artifact directory integrity", () => {
  it("seals the complete tree without certifying a release or interpreting BUILD_ID", () => {
    writeFileSync(join(root, "BUILD_ID"), "retained-but-not-build-proof")
    const seal = createArtifactSeal(root)
    expect(seal).toMatchObject({ kind: "artifact-integrity-only", certifiesRelease: false })
    expect(seal.entries.map((entry: { path: string }) => entry.path)).toEqual([".", "BUILD_ID", "static", "static/app.js"])
    expect(seal.entries.find((entry: { path: string }) => entry.path === "static/app.js")).toMatchObject({ type: "file", bytes: 8, sha256: sha256("original") })
    expect(verifyArtifactSeal(root, JSON.parse(JSON.stringify(seal)))).toMatchObject({ kind: "artifact-integrity-only", certifiesRelease: false, sha256: seal.sha256 })
  })
  it.each([
    ["added executable", () => writeFileSync(join(root, "injected.js"), "execute()")],
    ["hidden executable", () => { mkdirSync(join(root, ".cache")); writeFileSync(join(root, ".cache/execute.js"), "execute()") }],
    ["removed file", () => unlinkSync(join(root, "static/app.js"))],
    ["same-size replacement", () => writeFileSync(join(root, "static/app.js"), "modified")],
    ["executable mode", () => chmodSync(join(root, "static/app.js"), 0o755)],
    ["new empty directory", () => mkdirSync(join(root, "empty"))],
  ] as const)("rejects %s", (_name, mutate) => {
    const seal = createArtifactSeal(root)
    mutate()
    expect(() => verifyArtifactSeal(root, seal)).toThrow(/inventory|changed/i)
  })
  it("includes root permissions in the inventory digest", () => {
    const seal = createArtifactSeal(root)
    chmodSync(root, seal.entries.find((entry: { path: string }) => entry.path === ".")!.mode ^ 0o020)
    expect(createArtifactSeal(root).sha256).not.toBe(seal.sha256)
  })
  it("binds the canonical root instead of accepting another directory", () => {
    const seal = createArtifactSeal(root)
    const other = join(temporary, "other")
    mkdirSync(other)
    expect(() => verifyArtifactSeal(other, seal)).toThrow(/root/i)
  })
  it("records link bytes and the in-root referent bytes", () => {
    symlinkSync("static/app.js", join(root, "app.js"))
    const seal = createArtifactSeal(root)
    expect(seal.entries.find((entry: { path: string }) => entry.path === "app.js")).toMatchObject({
      type: "symlink", target: "static/app.js", bytes: 13, targetSha256: sha256("static/app.js"), resolvedPath: "static/app.js", referentSha256: sha256("original"),
    })
    writeFileSync(join(root, "static/app.js"), "modified")
    expect(() => verifyArtifactSeal(root, seal)).toThrow(/inventory|changed/i)
  })
  it("rejects link retargeting even when both referents contain identical bytes", () => {
    writeFileSync(join(root, "static/copy.js"), "original")
    symlinkSync("static/app.js", join(root, "app.js"))
    const seal = createArtifactSeal(root)
    unlinkSync(join(root, "app.js"))
    symlinkSync("static/copy.js", join(root, "app.js"))
    expect(() => verifyArtifactSeal(root, seal)).toThrow(/inventory|changed/i)
  })
  it("hashes all content reached through an in-root directory link", () => {
    symlinkSync("static", join(root, "assets"))
    const seal = createArtifactSeal(root)
    expect(() => verifyArtifactSeal(root, seal)).not.toThrow()
    writeFileSync(join(root, "static/new.js"), "added")
    expect(() => verifyArtifactSeal(root, seal)).toThrow(/inventory|changed/i)
  })
  it.each(["../outside", "../artifact-other/secret.js"])("rejects escaping targets including root-prefix siblings: %s", target => {
    mkdirSync(join(temporary, "artifact-other"))
    writeFileSync(join(temporary, "artifact-other/secret.js"), "outside")
    writeFileSync(join(temporary, "outside"), "outside")
    symlinkSync(target, join(root, "escape"))
    expect(() => createArtifactSeal(root)).toThrow(/outside|escape/i)
  })
  it.each([
    ["broken link", () => symlinkSync("missing", join(root, "broken"))],
    ["link loop", () => { symlinkSync("b", join(root, "a")); symlinkSync("a", join(root, "b")) }],
    ["ancestor directory cycle", () => symlinkSync("..", join(root, "static/parent"))],
    ["cross-directory cycle", () => { mkdirSync(join(root, "other")); symlinkSync("../other", join(root, "static/other")); symlinkSync("../static", join(root, "other/static")) }],
  ] as const)("rejects %s", (_name, prepare) => {
    prepare()
    expect(() => createArtifactSeal(root)).toThrow()
  })
  it("accepts a chain of in-root file links and records its ultimate referent", () => {
    symlinkSync("static/app.js", join(root, "second"))
    symlinkSync("second", join(root, "first"))
    const seal = createArtifactSeal(root)
    expect(seal.entries.find((entry: { path: string }) => entry.path === "first")).toMatchObject({ target: "second", resolvedPath: "static/app.js", referentSha256: sha256("original") })
    expect(() => verifyArtifactSeal(root, seal)).not.toThrow()
  })
  it("rejects non-UTF-8 link targets instead of replacing invalid bytes", () => {
    symlinkSync(Buffer.from([0xff]), join(root, "invalid-target"))
    expect(() => createArtifactSeal(root)).toThrow(/UTF-8/i)
  })
  it("rejects special files without opening or blocking on them", () => {
    const created = spawnSync("mkfifo", [join(root, "pipe")], { encoding: "utf8" })
    expect(created.status, created.stderr).toBe(0)
    expect(() => createArtifactSeal(root)).toThrow(/unsupported/i)
  })
  it("refuses receipts inside the inventoried root, including a parent-directory alias", () => {
    symlinkSync(root, join(temporary, "alias"))
    expect(() => writeArtifactSeal(root, join(temporary, "alias/receipt.json"))).toThrow(/outside/i)
  })
  it("does not overwrite an existing receipt", () => {
    const receipt = join(temporary, "seal.json")
    writeFileSync(receipt, "preserve earlier evidence")
    expect(() => writeArtifactSeal(root, receipt)).toThrow()
    expect(readFileSync(receipt, "utf8")).toBe("preserve earlier evidence")
  })
  it.each(["certifiesRelease", "sha256", "entries"])("rejects malformed receipt field: %s", field => {
    const seal = { ...createArtifactSeal(root), [field]: field === "entries" ? [] : "untrusted" }
    expect(() => verifyArtifactSeal(root, seal)).toThrow()
  })
})

describe("artifact-integrity CLI", () => {
  it("creates exclusive evidence then verifies actual files and rejects an added executable", () => {
    const receipt = join(temporary, "seal.json")
    const created = runCli("--seal-artifacts", root, receipt)
    expect(created.status, created.stderr).toBe(0)
    expect(JSON.parse(created.stdout)).toMatchObject({ kind: "artifact-integrity-only", certifiesRelease: false, action: "sealed" })
    const before = readFileSync(receipt, "utf8")
    expect(runCli("--seal-artifacts", root, receipt).status).toBe(1)
    expect(readFileSync(receipt, "utf8")).toBe(before)
    const verified = runCli("--verify-artifacts", root, receipt)
    expect(verified.status, verified.stderr).toBe(0)
    expect(JSON.parse(verified.stdout)).toMatchObject({ kind: "artifact-integrity-only", certifiesRelease: false, action: "verified" })
    writeFileSync(join(root, "added.js"), "execute()")
    const changed = runCli("--verify-artifacts", root, receipt)
    expect(changed.status).toBe(1)
    expect(changed.stdout).toBe("")
  })
})
