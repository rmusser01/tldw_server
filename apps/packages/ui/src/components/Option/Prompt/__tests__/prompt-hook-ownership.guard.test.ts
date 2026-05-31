import { promises as fs } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const promptRoot = path.resolve(__dirname, "..")
const managedHooks = [
  {
    name: "usePromptSync",
    modulePath: "hooks/usePromptSync",
    implementationPath: "hooks/usePromptSync.tsx"
  },
  {
    name: "usePromptEditor",
    modulePath: "hooks/usePromptEditor",
    implementationPath: "hooks/usePromptEditor.tsx"
  },
  {
    name: "usePromptBulkActions",
    modulePath: "hooks/usePromptBulkActions",
    implementationPath: "hooks/usePromptBulkActions.tsx"
  }
] as const

const sourceExtensions = new Set([".ts", ".tsx"])

type PromptSourceFile = {
  absolutePath: string
  relativePath: string
  source: string
}

const escapeRegExp = (value: string) =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")

const managedHookCallPattern = (hookName: string) =>
  new RegExp(`\\b${escapeRegExp(hookName)}\\s*\\(`)

const managedHookModuleImportPattern = (modulePath: string) =>
  new RegExp(
    [
      `\\bfrom\\s+["'][^"']*${escapeRegExp(modulePath)}["']`,
      `\\bimport\\s*\\(\\s*["'][^"']*${escapeRegExp(modulePath)}["']\\s*\\)`
    ].join("|")
  )

const managedHookReferencePattern = (hookName: string) =>
  new RegExp(`\\b${escapeRegExp(hookName)}\\b`)

const walkSourceFiles = async (directory: string): Promise<string[]> => {
  const entries = await fs.readdir(directory, { withFileTypes: true })
  const nestedPaths = await Promise.all(entries.map(async (entry) => {
    const absolutePath = path.join(directory, entry.name)

    if (entry.isDirectory()) {
      if (entry.name === "__tests__") return []
      return walkSourceFiles(absolutePath)
    }

    if (!entry.isFile() || !sourceExtensions.has(path.extname(entry.name))) {
      return []
    }

    return [absolutePath]
  }))

  return nestedPaths.flat()
}

const toPromptRelativePath = (absolutePath: string) =>
  path.relative(promptRoot, absolutePath).split(path.sep).join("/")

const readPromptSources = async (): Promise<PromptSourceFile[]> => {
  const files = await walkSourceFiles(promptRoot)
  return Promise.all(files.map(async (absolutePath) => ({
    absolutePath,
    relativePath: toPromptRelativePath(absolutePath),
    source: await fs.readFile(absolutePath, "utf8")
  })))
}

describe("prompt hook ownership guards", () => {
  it("keeps prompt mutation hook instances owned by the PromptBody orchestrator", async () => {
    const componentSources = (await readPromptSources()).filter(
      ({ relativePath }) => !relativePath.startsWith("hooks/")
    )

    for (const hook of managedHooks) {
      const filesCallingHook = componentSources
        .filter(({ source }) => managedHookCallPattern(hook.name).test(source))
        .map(({ relativePath }) => relativePath)

      const filesImportingHookModule = componentSources
        .filter(({ source }) => managedHookModuleImportPattern(hook.modulePath).test(source))
        .map(({ relativePath }) => relativePath)

      expect(filesCallingHook).toEqual(["index.tsx"])
      expect(filesImportingHookModule).toEqual(["index.tsx"])
    }
  })

  it("does not expose managed prompt hooks through aliasable hook files", async () => {
    const hookSources = (await readPromptSources()).filter(({ relativePath }) =>
      relativePath.startsWith("hooks/")
    )

    for (const hook of managedHooks) {
      const filesReferencingHookOutsideImplementation = hookSources
        .filter(({ relativePath }) => relativePath !== hook.implementationPath)
        .filter(({ source }) => managedHookReferencePattern(hook.name).test(source))
        .map(({ relativePath }) => relativePath)

      expect(filesReferencingHookOutsideImplementation).toEqual([])
    }
  })
})
