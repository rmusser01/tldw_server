import { getTokenizerWorker, terminateTokenizerWorker } from "./TokenizerWorker"

type TreeItem = {
  name?: string
  path: string
  type?: string
}

type FileContentItem = {
  path: string
  text: string
}

type FormattedOutput = {
  directoryTree: string
  fileContents: string
  tokenCount: number
  lineCount: number
}

export class Formatter {
  static async formatAsync(
    tree: TreeItem[],
    fileContents: FileContentItem[],
    onProgress?: (progress: number, current: number, total: number) => void,
    signal?: AbortSignal
  ): Promise<FormattedOutput> {
    signal?.throwIfAborted()
    const abort = () => terminateTokenizerWorker()
    signal?.addEventListener("abort", abort, { once: true })
    try {
      const directoryTree = this.generateDirectoryTree(tree)
      const worker = getTokenizerWorker()
      const fileContentsText = this.generateFileContents(fileContents)
      const fullOutput = `${directoryTree}\n\n${fileContentsText}`
      const tokenCount = await worker.tokenize(fullOutput)
      signal?.throwIfAborted()
      onProgress?.(100, fileContents.length, fileContents.length)

      return {
        directoryTree,
        fileContents: fileContentsText,
        tokenCount,
        lineCount: fullOutput.split("\n").length
      }
    } finally {
      signal?.removeEventListener("abort", abort)
    }
  }

  private static generateDirectoryTree(tree: TreeItem[]): string {
    const lines = ["Directory Structure:", "---"]
    for (const node of tree) {
      const name = node.path || node.name || "unknown"
      lines.push(`- ${name}`)
    }
    return lines.join("\n")
  }

  private static generateFileContents(fileContents: FileContentItem[]): string {
    if (fileContents.length === 0) {
      return "File Contents:\n---\n\nNo files selected."
    }

    const sections: string[] = ["File Contents:", "---"]
    for (const file of fileContents) {
      sections.push("")
      sections.push(`File: ${file.path}`)
      sections.push("---")
      sections.push(file.text)
    }
    return sections.join("\n")
  }
}
