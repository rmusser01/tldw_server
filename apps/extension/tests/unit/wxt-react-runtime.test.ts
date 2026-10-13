import { afterEach, expect, test } from "bun:test"
import {
  cpSync,
  mkdirSync,
  mkdtempSync,
  rmSync,
  symlinkSync,
  writeFileSync
} from "node:fs"
import { createRequire } from "node:module"
import { tmpdir } from "node:os"
import path from "node:path"
import { fileURLToPath, pathToFileURL } from "node:url"
import vm from "node:vm"

import config from "../../wxt.config"

const extensionRoot = fileURLToPath(new URL("../../", import.meta.url))
const require = createRequire(import.meta.url)
// WXT owns Vite; the workspace also installs a newer direct Vite version.
const vitePackage = createRequire(require.resolve("wxt")).resolve(
  "vite/package.json"
)
const { build } = await import(
  new URL("./dist/node/index.js", pathToFileURL(vitePackage)).href
)
const temporaryDirectories: string[] = []

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true })
  }
})

test("renders a linked dependency's hook through the extension React dispatcher", async () => {
  const directory = mkdtempSync(path.join(tmpdir(), "wxt-react-runtime-"))
  temporaryDirectories.push(directory)
  const modules = path.join(directory, "node_modules")
  const peer = path.join(modules, "linked-react-peer")
  mkdirSync(path.join(peer, "node_modules"), { recursive: true })
  const reactDirectory = path.dirname(require.resolve("react/package.json"))
  symlinkSync(reactDirectory, path.join(modules, "react"), "dir")
  symlinkSync(
    path.dirname(require.resolve("react-dom/package.json")),
    path.join(modules, "react-dom"),
    "dir"
  )
  // A second physical copy of the same version reproduces linked workspace peers.
  cpSync(reactDirectory, path.join(peer, "node_modules", "react"), {
    recursive: true
  })
  writeFileSync(
    path.join(peer, "package.json"),
    JSON.stringify({ name: "linked-react-peer", main: "index.js" })
  )
  writeFileSync(
    path.join(peer, "index.js"),
    `
    import { useReducer } from "react"
    export function useLinkedState() {
      return useReducer((state) => state, 7)[0]
    }
  `
  )
  const entry = path.join(directory, "entry.js")
  writeFileSync(
    entry,
    `
    import React from "react"
    import { renderToString } from "react-dom/server"
    import { useLinkedState } from "linked-react-peer"
    function LinkedComponent() {
      return React.createElement("span", null, useLinkedState())
    }
    export function render() {
      return renderToString(React.createElement(LinkedComponent))
    }
  `
  )

  const production = await config.vite?.({
    browser: "chrome",
    command: "build",
    manifestVersion: 3,
    mode: "production"
  })
  const result = await build({
    ...production,
    root: extensionRoot,
    configFile: false,
    define: {
      ...production?.define,
      "process.env.NODE_ENV": JSON.stringify("production")
    },
    cacheDir: path.join(directory, "vite-cache"),
    logLevel: "silent",
    build: {
      ...production?.build,
      write: false,
      minify: false,
      lib: { entry, formats: ["cjs"] }
    }
  })
  const outputs = Array.isArray(result)
    ? result.flatMap((output) => output.output)
    : result.output
  const chunk = outputs.find((output) => output.type === "chunk")
  if (!chunk || chunk.type !== "chunk") {
    throw new Error("Missing runtime identity regression bundle")
  }
  const loaded = { exports: {} as { render: () => string } }
  vm.runInNewContext(chunk.code, {
    module: loaded,
    exports: loaded.exports,
    TextEncoder,
    TextDecoder,
    ReadableStream,
    setTimeout,
    clearTimeout
  })
  expect(loaded.exports.render()).toBe("<span>7</span>")
}, 30_000)
