import fs from "node:fs"
import { promises as fsPromises } from "node:fs"
import path from "node:path"

export type DocumentationSource = "extension" | "server"

export type DocumentationManifestEntry = {
  id: string
  title: string
  source: DocumentationSource
  relativePath: string
  fullPath: string
}

export type DocumentationManifest = Record<
  DocumentationSource,
  DocumentationManifestEntry[]
>

const DOC_EXTENSIONS = new Set([".md", ".mdx"])
const SOURCE_ROOTS: Record<DocumentationSource, string> = {
  extension: "Docs/User_Documentation",
  server: "Docs/Published",
}

const normalizeTitle = (value: string) =>
  value.replace(/\s+/g, " ").trim()

const toTitleCase = (value: string) =>
  value.replace(/\b\w/g, (char) => char.toUpperCase())

const fileTitleFromPath = (relativePath: string) => {
  const fileName = relativePath.split("/").pop() || relativePath
  const baseName = fileName.replace(/\.[^.]+$/, "")
  return toTitleCase(normalizeTitle(baseName.replace(/[_-]+/g, " ")))
}

const toPosixPath = (value: string) => value.split(path.sep).join("/")

const resolveRepoRoot = () => {
  let candidate = process.cwd()
  // Both the checkout and standalone runtime live at <root>/apps/tldw-frontend.
  // Do not fall back to documentation outside that root when packaging is incomplete.
  for (let depth = 0; depth <= 2; depth += 1) {
    if (fs.existsSync(path.join(candidate, SOURCE_ROOTS.server))) {
      return candidate
    }
    candidate = path.dirname(candidate)
  }

  throw new Error("Unable to resolve repository root for documentation sources.")
}

const resolveSourceRoot = (source: DocumentationSource) =>
  path.join(resolveRepoRoot(), SOURCE_ROOTS[source])

const walkMarkdownFiles = async (directoryPath: string): Promise<string[]> => {
  let entries: fs.Dirent[]
  try {
    entries = await fsPromises.readdir(directoryPath, { withFileTypes: true })
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return []
    }
    throw error
  }

  const markdownFiles = await Promise.all(
    entries.map(async (entry) => {
      const absolutePath = path.join(directoryPath, entry.name)
      if (entry.isDirectory()) {
        return walkMarkdownFiles(absolutePath)
      }
      if (!entry.isFile()) return []
      if (!DOC_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) return []
      return [absolutePath]
    })
  )

  return markdownFiles.flat()
}

export const listDocumentationManifest = async (): Promise<DocumentationManifest> => {
  const manifestEntries = await Promise.all(
    (Object.keys(SOURCE_ROOTS) as DocumentationSource[]).map(async (source) => {
      const sourceRoot = resolveSourceRoot(source)
      const files = await walkMarkdownFiles(sourceRoot)
      const docs = files
        .map((absolutePath) => {
          const relativePath = toPosixPath(path.relative(sourceRoot, absolutePath))
          const fullPath = `${SOURCE_ROOTS[source]}/${relativePath}`
          return {
            id: `${source}:${fullPath}`,
            title: fileTitleFromPath(relativePath),
            source,
            relativePath,
            fullPath,
          } satisfies DocumentationManifestEntry
        })
        .sort((a, b) =>
          a.title.localeCompare(b.title) || a.relativePath.localeCompare(b.relativePath)
        )

      return [source, docs] as const
    })
  )

  return Object.fromEntries(manifestEntries) as DocumentationManifest
}

const sanitizeRelativePath = (relativePath: string) =>
  relativePath.replace(/^\/+/, "")

export const readDocumentationContent = async (
  source: DocumentationSource,
  relativePath: string
): Promise<string> => {
  const sanitizedPath = sanitizeRelativePath(relativePath)
  const sourceRoot = resolveSourceRoot(source)
  const absolutePath = path.resolve(sourceRoot, sanitizedPath)
  const relativeToRoot = path.relative(sourceRoot, absolutePath)

  if (
    !sanitizedPath ||
    sanitizedPath.includes("\0") ||
    relativeToRoot.startsWith("..") ||
    path.isAbsolute(relativeToRoot)
  ) {
    throw new Error("Invalid documentation path.")
  }

  if (!DOC_EXTENSIONS.has(path.extname(absolutePath).toLowerCase())) {
    throw new Error("Unsupported documentation file type.")
  }

  return fsPromises.readFile(absolutePath, "utf8")
}
