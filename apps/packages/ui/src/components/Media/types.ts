export type MediaResultItem = {
  kind: "media" | "note"
  id: string | number
  title?: string
  snippet?: string
  keywords?: string[]
  meta?: Record<string, any>
  raw: any
}

/** Result identities keep Notes and Media with equal backend IDs distinct. */
export const mediaResultKey = (item: Pick<MediaResultItem, 'kind' | 'id'>): string => `${item.kind}:${item.id}`
