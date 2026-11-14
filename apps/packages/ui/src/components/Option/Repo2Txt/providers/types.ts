export type ProviderType = "github" | "local"

export type ProviderCredentials = {
  token?: string
}

export type ParsedRepoInfo = {
  owner?: string
  repo?: string
  branch?: string
  path?: string
  url: string
  isValid: boolean
  error?: string
}

export type FetchTreeOptions = {
  branch?: string
  path?: string
  signal?: AbortSignal
}

export type RepoTreeNode = {
  path: string
  type: "blob" | "tree"
  url?: string
  urlType?: "api" | "directory" | "zip"
  size?: number
  sha?: string
}

export type RepoFileContent = {
  path: string
  text: string
  url?: string
  lineCount: number
}

export interface IProvider {
  getType(): ProviderType
  getName(): string
  requiresAuth(): boolean
  setCredentials(credentials: ProviderCredentials): void
  validateUrl(url: string): boolean
  parseUrl(url: string): ParsedRepoInfo
  fetchTree(url: string, options?: FetchTreeOptions): Promise<RepoTreeNode[]>
  fetchFile(node: RepoTreeNode, signal?: AbortSignal): Promise<RepoFileContent>
}
