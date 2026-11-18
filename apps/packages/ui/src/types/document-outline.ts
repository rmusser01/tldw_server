export interface TocItem {
  title: string
  page: number
  level: number
  children?: TocItem[]
  /** For EPUB: the href to navigate to */
  href?: string
  /** Generated PDF entries resolve their physical page only when selected. */
  navigationNodeId?: string
  unavailable?: boolean
}

export interface DocumentOutline {
  documentId: number
  items: TocItem[]
}
