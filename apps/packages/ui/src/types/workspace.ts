/**
 * Workspace Types
 * Types for the NotebookLM-style three-pane research interface
 */

import type { WorkProductTemplateId } from "@/workspace-templates/types"

// ─────────────────────────────────────────────────────────────────────────────
// Source Types
// ─────────────────────────────────────────────────────────────────────────────

export type WorkspaceSourceType =
  | "pdf"
  | "video"
  | "audio"
  | "website"
  | "document"
  | "text"

export type WorkspaceSourceStatus = "processing" | "ready" | "error"

export interface WorkspaceSource {
  id: string
  mediaId: number // Server-side media ID
  title: string
  type: WorkspaceSourceType
  status?: WorkspaceSourceStatus
  statusMessage?: string
  thumbnailUrl?: string
  addedAt: Date
  sourceCreatedAt?: Date
  // Optional metadata
  url?: string
  fileSize?: number
  duration?: number // For audio/video in seconds
  pageCount?: number // For PDFs
}

export interface WorkspaceSourceFolder {
  id: string
  workspaceId: string
  name: string
  parentFolderId: string | null
  createdAt: Date
  updatedAt: Date
}

export interface WorkspaceSourceFolderMembership {
  folderId: string
  sourceId: string
}

export type WorkspaceSourceTransferMode = "copy" | "move"

export type WorkspaceSourceTransferConflictResolution =
  | "skip"
  | "merge-folders"
  | "replace-transferred-folders"

export type WorkspaceSourceTransferEmptyFolderPolicy =
  | "keep"
  | "delete-empty-folders"

export type WorkspaceSourceTransferIdKind = "source" | "folder"

export interface WorkspaceSourceTransferSnapshot {
  workspaceId: string
  sources: WorkspaceSource[]
  sourceFolders: WorkspaceSourceFolder[]
  sourceFolderMemberships: WorkspaceSourceFolderMembership[]
}

export interface WorkspaceSourceTransferInput {
  mode: WorkspaceSourceTransferMode
  originSnapshot: WorkspaceSourceTransferSnapshot
  destinationSnapshot: WorkspaceSourceTransferSnapshot
  selectedSourceIds: string[]
  conflictResolutions: Record<number, WorkspaceSourceTransferConflictResolution>
  emptyFolderPolicy: WorkspaceSourceTransferEmptyFolderPolicy
  sourceFolderFallbackName: string
  generateId: (kind: WorkspaceSourceTransferIdKind) => string
}

export interface WorkspaceSourceTransferResult {
  originSnapshot: WorkspaceSourceTransferSnapshot
  destinationSnapshot: WorkspaceSourceTransferSnapshot
  transferredMediaIds: number[]
  transferredDestinationSourceIds: string[]
  removedOriginSourceIds: string[]
  newlyEmptiedOriginFolderIds: string[]
  conflictsResolved: number[]
  conflictsSkipped: number[]
}

export interface WorkspaceCollection {
  id: string
  name: string
  description: string | null
  createdAt: Date
  updatedAt: Date
}

// ─────────────────────────────────────────────────────────────────────────────
// Artifact Types
// ─────────────────────────────────────────────────────────────────────────────

export type ArtifactType =
  | "summary"
  | "audio_overview"
  | "mindmap"
  | "report"
  | "compare_sources"
  | "flashcards"
  | "quiz"
  | "timeline"
  | "slides"
  | "data_table"

export type ArtifactStatus = "pending" | "generating" | "completed" | "failed"
export type ArtifactReviewStatus =
  | "draft"
  | "reviewing"
  | "accepted"
  | "needs_revision"
  | "rejected"
  | "exported"
  | "assigned"
  | "archived"

export type ArtifactExportTarget =
  | "markdown"
  | "docx"
  | "pdf"
  | "slides"
  | "chatbook"

export interface ArtifactSourceLineage {
  sourceId: string
  sourceType?: string
  mediaId?: number
  title?: string
  label?: string
  citationCount?: number
  citationSpans?: unknown[]
  evidenceIds?: string[]
  coverageNotes?: string
  [key: string]: unknown
}

export interface ArtifactReviewChecklistItem {
  id: string
  label: string
  checked: boolean
}

export interface TraceableArtifactProducerLinks {
  session?: string
  run?: string
  review?: string
  diagnostics?: string
  artifacts?: string
  audit?: string
  [key: string]: string | undefined
}

export interface TraceableArtifactProducerMetadata {
  producerType?: string
  producerId?: string
  runId?: string
  sessionId?: string
  reviewId?: string
  taskId?: string
  promptId?: string
  templateId?: string
  model?: string
  provider?: string
  completionReason?: string
  links?: TraceableArtifactProducerLinks
  [key: string]: unknown
}

export interface TraceableArtifactReviewMetadata {
  reviewerId?: string
  decision?: ArtifactReviewStatus | string
  decidedAt?: string
  reason?: string
  checklist?: ArtifactReviewChecklistItem[]
  [key: string]: unknown
}

export interface TraceableArtifactVersionMetadata {
  revisionReason?: string
  versionLabel?: string
  comparedToVersionId?: string
  [key: string]: unknown
}

export interface TraceableArtifactExportRef {
  id?: number | string
  format: string
  fileId?: number | string
  jobId?: number | string
  artifactVersionId?: string
  generatedAt?: string
  expiresAt?: string
  status?: string
  url?: string
  error?: string
  [key: string]: unknown
}

export interface TraceableArtifactRedaction {
  supportSafe?: boolean
  redacted?: boolean
  retentionClass?: string
  redactedFields?: string[]
  visibility?: string
  [key: string]: unknown
}

export type StudyMaterialsPolicy = "general" | "workspace"

export type WorkspaceStudyArtifactSource = {
  source_type: string
  source_id: string
}

export interface WorkspaceStudyArtifactData {
  quizId?: number
  deckId?: number
  workspaceId?: string | null
  sourceMediaIds?: number[]
  sourceBundle?: WorkspaceStudyArtifactSource[]
  questions?: Array<{
    question?: string
    question_text?: string
    options: string[]
    answer?: string
    correct_answer?: string
    explanation?: string
    sourceMediaId?: number
  }>
  flashcards?: Array<{
    front: string
    back: string
  }>
}

export interface GeneratedArtifact {
  id: string
  type: ArtifactType
  title: string
  status: ArtifactStatus
  templateId?: WorkProductTemplateId
  reviewStatus?: ArtifactReviewStatus
  sourceLineage?: ArtifactSourceLineage[]
  reviewChecklist?: ArtifactReviewChecklistItem[]
  exportTargets?: ArtifactExportTarget[]
  version?: number
  previousVersionId?: string
  rootArtifactId?: string
  artifactVersionId?: string
  ownerScope?: string
  ownerId?: string
  projectId?: string
  taskId?: string
  sourceCollectionId?: string
  contentType?: string
  previewText?: string
  summary?: string
  schemaVersion?: number
  producerMetadata?: TraceableArtifactProducerMetadata
  reviewMetadata?: TraceableArtifactReviewMetadata
  versionMetadata?: TraceableArtifactVersionMetadata
  exportRefs?: TraceableArtifactExportRef[]
  redaction?: TraceableArtifactRedaction
  estimatedTokens?: number
  estimatedCostUsd?: number
  totalTokens?: number
  totalCostUsd?: number
  serverId?: number | string // ID from outputs/quizzes/data-tables/slides endpoint
  content?: string // For text-based artifacts like summary, mindmap
  audioUrl?: string // For audio_overview - object URL to audio blob
  audioFormat?: string // Audio format (mp3, wav, etc.)
  presentationId?: string // For slides - ID of the generated presentation
  presentationVersion?: number // For slides - version for export
  errorMessage?: string // If status is failed
  data?: WorkspaceStudyArtifactData & Record<string, unknown> // Optional structured artifact payload (quiz, flashcards, tables, etc.)
  createdAt: Date
  completedAt?: Date
}

// ─────────────────────────────────────────────────────────────────────────────
// Output Configuration Types
// ─────────────────────────────────────────────────────────────────────────────

export interface OutputTypeConfig {
  type: ArtifactType
  label: string
  icon: string // Lucide icon name
  description: string
  // API configuration
  endpoint?: string
  requiresSelectedSources?: boolean
}

export const OUTPUT_TYPES: OutputTypeConfig[] = [
  {
    type: "audio_overview",
    label: "Audio Summary",
    icon: "Headphones",
    description: "Generate a spoken summary of your sources",
    requiresSelectedSources: true
  },
  {
    type: "summary",
    label: "Summary",
    icon: "FileText",
    description: "Create a concise summary of key points",
    requiresSelectedSources: true
  },
  {
    type: "mindmap",
    label: "Mind Map",
    icon: "GitBranch",
    description: "Visualize concepts and relationships",
    requiresSelectedSources: true
  },
  {
    type: "report",
    label: "Report",
    icon: "FileSpreadsheet",
    description: "Generate a detailed report document",
    requiresSelectedSources: true
  },
  {
    type: "compare_sources",
    label: "Compare Sources",
    icon: "Scale",
    description: "Compare claims, agreements, and conflicts across sources",
    requiresSelectedSources: true
  },
  {
    type: "flashcards",
    label: "Flashcards",
    icon: "Layers",
    description: "Create study flashcards for review",
    requiresSelectedSources: true
  },
  {
    type: "quiz",
    label: "Quiz",
    icon: "HelpCircle",
    description: "Generate a quiz to test understanding",
    requiresSelectedSources: true
  },
  {
    type: "timeline",
    label: "Timeline",
    icon: "Calendar",
    description: "Create a chronological timeline",
    requiresSelectedSources: true
  },
  {
    type: "slides",
    label: "Slides",
    icon: "Presentation",
    description: "Generate presentation slides",
    requiresSelectedSources: true
  },
  {
    type: "data_table",
    label: "Data Table",
    icon: "Table",
    description: "Extract structured data into a table",
    requiresSelectedSources: true
  }
]

// ─────────────────────────────────────────────────────────────────────────────
// Workspace Configuration
// ─────────────────────────────────────────────────────────────────────────────

export interface WorkspaceConfig {
  id: string
  name: string
  tag: string // Format: "workspace:<slug>"
  createdAt: Date
  updatedAt: Date
}

// ─────────────────────────────────────────────────────────────────────────────
// Add Source Modal Types
// ─────────────────────────────────────────────────────────────────────────────

export type AddSourceTab = "upload" | "url" | "paste" | "search" | "existing"

export interface AddSourceModalState {
  open: boolean
  activeTab: AddSourceTab
  isProcessing: boolean
  error: string | null
}

// ─────────────────────────────────────────────────────────────────────────────
// UI State Types
// ─────────────────────────────────────────────────────────────────────────────

export interface WorkspaceUIState {
  leftPaneCollapsed: boolean
  rightPaneCollapsed: boolean
  sourceSearchQuery: string
  addSourceModalState: AddSourceModalState
}

// ─────────────────────────────────────────────────────────────────────────────
// Workspace Banner Types
// ─────────────────────────────────────────────────────────────────────────────

export type WorkspaceBannerImageMimeType =
  | "image/jpeg"
  | "image/png"
  | "image/webp"

export interface WorkspaceBannerImage {
  dataUrl: string
  mimeType: WorkspaceBannerImageMimeType
  width: number
  height: number
  bytes: number
  updatedAt: Date
}

export interface WorkspaceBanner {
  title: string
  subtitle: string
  image: WorkspaceBannerImage | null
}

export const DEFAULT_WORKSPACE_BANNER: WorkspaceBanner = {
  title: "",
  subtitle: "",
  image: null
}

// ─────────────────────────────────────────────────────────────────────────────
// Audio Generation Settings
// ─────────────────────────────────────────────────────────────────────────────

export type AudioTtsProvider = "browser" | "elevenlabs" | "openai" | "tldw"

export interface AudioGenerationSettings {
  provider: AudioTtsProvider
  model: string // e.g., "kokoro", "tts-1", "tts-1-hd"
  voice: string // e.g., "af_heart", "alloy"
  speed: number // 0.5 - 2.0
  format: "mp3" | "wav" | "opus" | "aac" | "flac"
}

export const DEFAULT_AUDIO_SETTINGS: AudioGenerationSettings = {
  provider: "browser",
  model: "kokoro",
  voice: "af_heart",
  speed: 1.0,
  format: "mp3"
}

// ─────────────────────────────────────────────────────────────────────────────
// Workspace Note Types (for Quick Notes feature)
// ─────────────────────────────────────────────────────────────────────────────

export interface WorkspaceNote {
  id?: number // undefined = new note, number = existing note
  title: string
  content: string
  keywords: string[]
  version?: number // For optimistic locking on updates
  isDirty: boolean // Has unsaved changes
}

export const DEFAULT_WORKSPACE_NOTE: WorkspaceNote = {
  id: undefined,
  title: "",
  content: "",
  keywords: [],
  version: undefined,
  isDirty: false
}

// ─────────────────────────────────────────────────────────────────────────────
// Saved Workspaces (for workspace switcher)
// ─────────────────────────────────────────────────────────────────────────────

export interface SavedWorkspace {
  id: string
  name: string
  tag: string
  collectionId: string | null
  createdAt: Date
  lastAccessedAt: Date
  /** Number of sources in this workspace */
  sourceCount: number
}

// ─────────────────────────────────────────────────────────────────────────────
// Slides/Presentation Types
// ─────────────────────────────────────────────────────────────────────────────

export type SlideLayout =
  | "title"
  | "content"
  | "two_column"
  | "quote"
  | "section"
  | "blank"

export interface Slide {
  order: number
  layout: SlideLayout
  title?: string
  content: string
  speaker_notes?: string
  metadata?: Record<string, unknown>
}

export interface PresentationResponse {
  id: string
  title: string
  description?: string
  theme: string
  slides: Slide[]
  version: number
  created_at: string
  last_modified: string
  deleted?: boolean
  source_type?: string
  source_ref?: string | number | string[] | null
}
