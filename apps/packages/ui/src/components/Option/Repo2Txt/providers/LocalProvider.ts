import JSZip from "jszip"
import { BaseProvider } from "./BaseProvider"
import {
  decodeText,
  MAX_FILE_BYTES,
  MAX_EXPORT_BYTES,
  MAX_REPO_ENTRIES,
  validateRepoPath
} from "../exportPolicy"
import type { ParsedRepoInfo, RepoFileContent, RepoTreeNode } from "./types"

type LocalSource = "directory" | "zip"

export type LocalProviderInitializeOptions = {
  source: LocalSource
  files?: FileList
  signal?: AbortSignal
  zipFile?: File
  onProgress?: (progress: number, message: string) => void
}

export class LocalProvider extends BaseProvider {
  private source: LocalSource | null = null
  private fileMap = new Map<string, File>()
  private zipInstance: JSZip | null = null

  getType() {
    return "local" as const
  }

  getName() {
    return "Local"
  }

  requiresAuth() {
    return false
  }

  validateUrl(url: string): boolean {
    return url.startsWith("local://")
  }

  parseUrl(url: string): ParsedRepoInfo {
    if (!this.validateUrl(url)) {
      return {
        url,
        isValid: false,
        error: "Invalid local URL. Expected local://directory or local://zip"
      }
    }
    return {
      url,
      isValid: true
    }
  }

  async initialize(options: LocalProviderInitializeOptions): Promise<void> {
    options.signal?.throwIfAborted()
    this.source = options.source
    this.fileMap.clear()
    this.zipInstance = null

    if (options.source === "directory") {
      if (!options.files || options.files.length === 0) {
        throw new Error("Directory initialization requires files")
      }
      if (options.files.length > MAX_REPO_ENTRIES)
        throw new Error("Repository exceeds 5000 files")
      for (let i = 0; i < options.files.length; i++) {
        const file = options.files[i]
        const relativePath = (file as File & { webkitRelativePath?: string })
          .webkitRelativePath
        const path = relativePath || file.name
        validateRepoPath(path)
        if (this.fileMap.has(path)) {
          throw new Error(
            `Duplicate filename without directory context: ${path}. Use directory picker or zip source to preserve paths.`
          )
        }
        this.fileMap.set(path, file)
      }
      return
    }

    if (!options.zipFile) {
      throw new Error("Zip initialization requires a zipFile")
    }

    if (options.zipFile.size > MAX_EXPORT_BYTES)
      throw new Error("ZIP exceeds 10 MiB")
    options.onProgress?.(0, "Loading zip file")
    this.zipInstance = await JSZip.loadAsync(options.zipFile)
    options.signal?.throwIfAborted()
    const entries = Object.values(this.zipInstance.files)
    if (entries.length > MAX_REPO_ENTRIES)
      throw new Error("Repository exceeds 5000 entries")
    for (const entry of entries) {
      const original =
        (entry as typeof entry & { unsafeOriginalName?: string })
          .unsafeOriginalName ?? entry.name
      validateRepoPath(original.replace(/\/$/, ""))
      if ((Number(entry.unixPermissions) & 0o170000) === 0o120000) {
        throw new Error(`Symbolic link cannot be exported: ${entry.name}`)
      }
    }
    options.onProgress?.(100, "Zip file loaded")
  }

  async fetchTree(_url: string): Promise<RepoTreeNode[]> {
    if (!this.source) {
      throw new Error("LocalProvider must be initialized before fetchTree")
    }

    if (this.source === "directory") {
      return Array.from(this.fileMap.entries()).map(([path, file]) => ({
        path,
        type: "blob",
        url: path,
        urlType: "directory",
        size: file.size
      }))
    }

    if (!this.zipInstance) {
      return []
    }

    const tree: RepoTreeNode[] = []
    this.zipInstance.forEach((path, file) => {
      if (file.dir) return
      tree.push({
        path,
        type: "blob",
        url: path,
        urlType: "zip"
      })
    })
    return tree
  }

  async fetchFile(
    node: RepoTreeNode,
    signal?: AbortSignal
  ): Promise<RepoFileContent> {
    signal?.throwIfAborted()
    validateRepoPath(node.path)
    if (node.urlType === "directory") {
      const file = this.fileMap.get(node.path)
      if (!file) {
        throw new Error(`Local file not found: ${node.path}`)
      }
      if (file.size > MAX_FILE_BYTES)
        throw new Error(`File exceeds 1 MiB: ${node.path}`)
      const bytes = await new Promise<Uint8Array>((resolve, reject) => {
        const reader = new FileReader()
        const abort = () => reader.abort()
        const clear = () => signal?.removeEventListener("abort", abort)
        reader.onload = () => {
          clear()
          resolve(new Uint8Array(reader.result as ArrayBuffer))
        }
        reader.onerror = () => {
          clear()
          reject(reader.error ?? new Error("Local file could not be read"))
        }
        reader.onabort = () => {
          clear()
          reject(new DOMException("Cancelled", "AbortError"))
        }
        signal?.addEventListener("abort", abort, { once: true })
        reader.readAsArrayBuffer(file)
      })
      signal?.throwIfAborted()
      const text = decodeText(bytes, node.path)
      return {
        path: node.path,
        text,
        lineCount: text.split("\n").length
      }
    }

    if (node.urlType === "zip") {
      const zipFile = this.zipInstance?.file(node.path)
      if (!zipFile) {
        throw new Error(`Zip file entry not found: ${node.path}`)
      }
      const bytes = await new Promise<Uint8Array>((resolve, reject) => {
        const stream = zipFile.internalStream("uint8array")
        const chunks: Uint8Array[] = []
        let size = 0
        const abort = () => {
          stream.pause()
          reject(new DOMException("Cancelled", "AbortError"))
        }
        const clear = () => signal?.removeEventListener("abort", abort)
        signal?.addEventListener("abort", abort, { once: true })
        stream
          .on("data", (chunk: Uint8Array) => {
            size += chunk.byteLength
            if (size > MAX_FILE_BYTES) {
              stream.pause()
              clear()
              reject(new Error(`File exceeds 1 MiB: ${node.path}`))
              return
            }
            chunks.push(chunk)
          })
          .on("error", (error) => {
            clear()
            reject(error)
          })
          .on("end", () => {
            clear()
            const result = new Uint8Array(size)
            let offset = 0
            for (const chunk of chunks) {
              result.set(chunk, offset)
              offset += chunk.byteLength
            }
            resolve(result)
          })
          .resume()
      })
      signal?.throwIfAborted()
      const text = decodeText(bytes, node.path)
      return {
        path: node.path,
        text,
        lineCount: text.split("\n").length
      }
    }

    return super.fetchFile(node, signal)
  }
}
