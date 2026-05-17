export type ReadAlongSegmentKind =
  | 'transcript-line'
  | 'sentence'
  | 'paragraph'
  | 'transient-selection'

export type ReadAlongScope =
  | 'selection'
  | 'from-here'
  | 'current-section'
  | 'full-item'

export interface ReadAlongSegment {
  id: string
  index: number
  kind: ReadAlongSegmentKind
  text: string
  sourceStart: number
  sourceEnd: number
  displayStart?: number
  displayEnd?: number
  sectionId?: string
  timestampSeconds?: number
}

export interface ReadAlongSelection {
  selectedText: string
  anchorRect: DOMRect
  startSegmentId?: string
  endSegmentId?: string
  sourceStart?: number
  sourceEnd?: number
  mappingConfidence: 'exact' | 'nearest' | 'text-only'
}

export interface BuildReadAlongSegmentsInput {
  mediaId: string
  content: string
  displayContent?: string
  renderMode?: 'plain' | 'markdown' | 'html' | string
  hideTranscriptTimings?: boolean
}

export interface ResolveReadAlongScopeInput {
  scope: ReadAlongScope
  segments: ReadAlongSegment[]
  selection: ReadAlongSelection
}

export interface ReadAlongTtsRequestSegment {
  id: string
  parentSegmentId: string
  index: number
  text: string
  sourceStart: number
  sourceEnd: number
}
