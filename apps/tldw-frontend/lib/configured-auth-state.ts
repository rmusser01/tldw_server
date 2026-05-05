type TldwClientLike = {
  getConfig: () => Promise<Record<string, unknown> | null>
}

type TldwAuthLike = {
  getCurrentUser: () => Promise<unknown>
}

export const loadTldwClient = async (): Promise<TldwClientLike> => {
  const clientModule = await import("@/services/tldw/TldwApiClient")
  return clientModule.tldwClient as unknown as TldwClientLike
}

export const loadTldwAuth = async (): Promise<TldwAuthLike> => {
  const authModule = await import("@/services/tldw/TldwAuth")
  return authModule.tldwAuth as TldwAuthLike
}
