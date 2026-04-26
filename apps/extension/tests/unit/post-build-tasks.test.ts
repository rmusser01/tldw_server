import path from "node:path"
import { pathToFileURL } from "node:url"

import { afterEach, describe, expect, it, vi } from "vitest"

afterEach(() => {
  vi.resetModules()
  vi.restoreAllMocks()
})

describe("extension post-build tasks", () => {
  it("includes Firefox dangerous-eval stripping before token verification", async () => {
    const moduleUrl = pathToFileURL(
      path.resolve(__dirname, "../../scripts/post-build-tasks.mjs")
    ).href
    const { getPostBuildTasks } = await import(moduleUrl)

    expect(getPostBuildTasks("firefox-mv2", "/tmp/firefox-mv2")).toEqual([
      {
        args: ["--dir", "/tmp/firefox-mv2"],
        script: "scripts/strip-dangerous-eval.mjs",
      },
      {
        args: ["--target", "firefox-mv2"],
        script: "scripts/verify-shared-token-sync.mjs",
      },
    ])
  })

  it("only verifies shared token sync for Chromium targets", async () => {
    const moduleUrl = pathToFileURL(
      path.resolve(__dirname, "../../scripts/post-build-tasks.mjs")
    ).href
    const { getPostBuildTasks } = await import(moduleUrl)

    expect(getPostBuildTasks("chrome-mv3", "/tmp/chrome-mv3")).toEqual([
      {
        args: ["--target", "chrome-mv3"],
        script: "scripts/verify-shared-token-sync.mjs",
      },
    ])
  })

  it("runs post-build tasks from the WXT build hook for production archive builds", async () => {
    const execFileSync = vi.fn()

    vi.doMock("node:child_process", () => ({
      execFileSync,
    }))

    const config = (await import("../../wxt.config.ts")).default
    const buildDoneHook = config.hooks?.["build:done"]

    expect(buildDoneHook).toBeTypeOf("function")

    await buildDoneHook?.(
      {
        config: {
          browser: "firefox",
          command: "build",
          manifestVersion: 2,
          outDir: path.join("/tmp", "firefox-mv2"),
          root: "/tmp",
        },
      } as any,
      {} as any
    )

    expect(execFileSync).toHaveBeenNthCalledWith(
      1,
      process.execPath,
      ["scripts/strip-dangerous-eval.mjs", "--dir", path.join("/tmp", "firefox-mv2")],
      expect.objectContaining({
        cwd: "/tmp",
        stdio: "inherit",
      })
    )
    expect(execFileSync).toHaveBeenNthCalledWith(
      2,
      process.execPath,
      ["scripts/verify-shared-token-sync.mjs", "--target", "firefox-mv2"],
      expect.objectContaining({
        cwd: "/tmp",
        stdio: "inherit",
      })
    )
  })

  it("skips post-build tasks while the WXT dev server is running", async () => {
    const execFileSync = vi.fn()

    vi.doMock("node:child_process", () => ({
      execFileSync,
    }))

    const config = (await import("../../wxt.config.ts")).default
    const buildDoneHook = config.hooks?.["build:done"]

    await buildDoneHook?.(
      {
        config: {
          browser: "chrome",
          command: "serve",
          manifestVersion: 3,
          outDir: path.join("/tmp", "chrome-mv3"),
          root: "/tmp",
        },
      } as any,
      {} as any
    )

    expect(execFileSync).not.toHaveBeenCalled()
  })
})
