import {
  decodeText,
  MAX_FILE_BYTES,
  MAX_EXPORT_BYTES,
  MAX_REPO_ENTRIES,
  readBoundedResponse,
  validateRepoPath
} from "../exportPolicy"
import { BaseProvider } from "./BaseProvider"
import type {
  FetchTreeOptions,
  ParsedRepoInfo,
  RepoFileContent,
  RepoTreeNode
} from "./types"

type GitHubContentResponse = {
  content?: string
  encoding?: string
}

type GitHubTreeEntry = {
  path: string
  type: "blob" | "tree" | string
  url?: string
  size?: number
  mode?: string
  sha?: string
}

type GitHubTreeResponse = {
  truncated?: boolean
  tree?: GitHubTreeEntry[]
}

const decodeBase64 = (value: string, path: string): string =>
  decodeText(
    Uint8Array.from(atob(value), (char) => char.charCodeAt(0)),
    path
  )

export class GitHubProvider extends BaseProvider {
  private static readonly API_BASE = "https://api.github.com"
  private static readonly URL_PATTERN =
    /^https:\/\/github\.com\/([a-zA-Z0-9_.-]+)\/([a-zA-Z0-9_.-]+)(?:\/tree\/([^?#]+))?$/

  getType() {
    return "github" as const
  }

  getName() {
    return "GitHub"
  }

  requiresAuth() {
    return false
  }

  validateUrl(url: string): boolean {
    const normalized = url.trim().replace(/\/$/, "")
    return GitHubProvider.URL_PATTERN.test(normalized)
  }

  parseUrl(url: string): ParsedRepoInfo {
    const normalized = url.trim().replace(/\/$/, "")
    const match = normalized.match(GitHubProvider.URL_PATTERN)

    if (!match) {
      return {
        url,
        isValid: false,
        error:
          "Invalid GitHub URL format. Expected https://github.com/owner/repo"
      }
    }

    const [, owner, repo, branch] = match
    return {
      owner,
      repo,
      branch,
      url,
      isValid: true
    }
  }

  async fetchTree(
    url: string,
    options: FetchTreeOptions = {}
  ): Promise<RepoTreeNode[]> {
    const parsed = this.parseUrl(url)
    if (!parsed.isValid || !parsed.owner || !parsed.repo) {
      throw new Error(parsed.error ?? "Invalid GitHub URL")
    }

    const ref = options.branch ?? parsed.branch ?? "HEAD"
    const treeUrl =
      `${GitHubProvider.API_BASE}/repos/${parsed.owner}/${parsed.repo}/git/trees/` +
      `${encodeURIComponent(ref)}?recursive=1`
    const response = await fetch(treeUrl, {
      headers: this.buildHeaders(),
      signal: options.signal
    })

    if (!response.ok) {
      throw new Error(`Failed to fetch GitHub tree: HTTP ${response.status}`)
    }

    const payload = JSON.parse(
      new TextDecoder().decode(
        await readBoundedResponse(response, MAX_EXPORT_BYTES, options.signal)
      )
    ) as GitHubTreeResponse
    if (payload.truncated)
      throw new Error(
        "GitHub returned an incomplete truncated tree; select a smaller repository"
      )
    const tree = payload.tree ?? []
    if (tree.length > MAX_REPO_ENTRIES)
      throw new Error("Repository exceeds 5000 entries")
    for (const entry of tree) {
      validateRepoPath(entry.path)
      if (entry.mode === "120000")
        throw new Error(`Symbolic link cannot be exported: ${entry.path}`)
    }
    return tree.map((entry) => ({
      path: entry.path,
      type: entry.type === "blob" ? "blob" : "tree",
      url: entry.url,
      size: entry.size,
      sha: entry.sha
    }))
  }

  async fetchFile(
    node: RepoTreeNode,
    signal?: AbortSignal
  ): Promise<RepoFileContent> {
    if (!node.url) {
      throw new Error(`Cannot fetch file content for ${node.path}: missing URL`)
    }

    validateRepoPath(node.path)
    const url = new URL(node.url)
    if (
      url.origin !== GitHubProvider.API_BASE ||
      !url.pathname.startsWith("/repos/")
    )
      throw new Error("Invalid GitHub file URL")
    if ((node.size ?? 0) > MAX_FILE_BYTES)
      throw new Error(`File exceeds 1 MiB: ${node.path}`)
    const response = await fetch(node.url, {
      headers: this.buildHeaders(),
      signal
    })
    if (!response.ok) {
      throw new Error(
        `Failed to fetch file ${node.path}: HTTP ${response.status}`
      )
    }

    const data = JSON.parse(
      new TextDecoder().decode(
        await readBoundedResponse(response, 2 * MAX_FILE_BYTES, signal)
      )
    ) as GitHubContentResponse
    let text = data.content ?? ""
    if (data.encoding === "base64") {
      text = decodeBase64(text.replace(/\s/g, ""), node.path)
    }

    if (data.encoding !== "base64")
      throw new Error("GitHub file content is not base64 encoded")
    return {
      path: node.path,
      text,
      url: node.url,
      lineCount: text.split("\n").length
    }
  }

  private buildHeaders(): HeadersInit {
    const headers: Record<string, string> = {
      Accept: "application/vnd.github+json"
    }

    if (this.credentials?.token) {
      headers.Authorization = `token ${this.credentials.token}`
    }

    return headers
  }
}
