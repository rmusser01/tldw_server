import { ChatDocuments } from '@/models/ChatTypes';
import type { DiscoSkillComment } from '@/types/disco-skills';

export type LastUsedModelType = { prompt_id?: string; prompt_content?: string }

export type HistoryInfo = {
  id: string;
  title: string;
  is_rag: boolean;
  message_source?: 'copilot' | 'web-ui' | 'branch' | 'server';
  is_pinned?: boolean;
  createdAt: number;
  doc_id?: string;
  last_used_prompt?: LastUsedModelType;
  model_id?: string;
  server_chat_id?: string;
  // Timeline/branching fields (server-compatible with ChaChaDB)
  root_id?: string;                    // All forks share same root_id
  parent_conversation_id?: string;     // Parent in fork tree
  forked_from_message_id?: string;     // Message that spawned this fork
  character_id?: number;               // Associated character/assistant
};

export type WebSearch = {
  search_engine: string;
  search_url: string;
  search_query: string;
  search_results: {
    title: string;
    link: string;
  }[];
};

export type UploadedFile = {
  id: string;
  filename: string;
  type: string;
  content: string;
  size: number;
  uploadedAt: number;
  embedding?: number[];
  processed: boolean;
};

export type SessionFiles = {
  id?: any 
  sessionId: string;
  files: UploadedFile[];
  retrievalEnabled: boolean;
  createdAt: number;
};

export type Message = {
  id: string;
  history_id: string;
  name: string;
  role: string;
  content: string;
  images?: string[];
  sources?: string[];
  search?: WebSearch;
  createdAt: number;
  reasoning_time_taken?: number;
  messageType?: string;
  clusterId?: string;
  modelId?: string;
  generationInfo?: any;
  metadataExtra?: Record<string, unknown>;
  modelName?: string;
  modelImage?: string;
  documents?: ChatDocuments;
  discoSkillComment?: DiscoSkillComment;
  // Server-sync fields
  serverMessageId?: string | null;     // Canonical server-side message ID
  serverMessageVersion?: number;       // Server-side message version
  // Timeline/branching fields (server-compatible with ChaChaDB)
  parent_message_id?: string | null;   // Parent message for threading/swipes
  depth?: number;                       // Computed depth in conversation tree
};

export type CompareParentMeta = {
  parentHistoryId: string;
  clusterId?: string;
};
export type CompareContinuationMode = "winner" | "compare";

export type CompareState = {
  history_id: string;
  compareMode: boolean;
  compareSelectedModels: string[];
  compareSelectionByCluster: Record<string, string[]>;
  compareCanonicalByCluster: Record<string, string | null>;
  compareContinuationModeByCluster: Record<string, CompareContinuationMode>;
  compareSplitChats: Record<string, Record<string, string>>;
  compareActiveModelsByCluster: Record<string, string[]>;
  compareParent?: CompareParentMeta | null;
  updatedAt: number;
};

export type Webshare = {
  id: string;
  title: string;
  url: string;
  api_url: string;
  share_id: string;
  createdAt: number;
};

// ─────────────────────────────────────────────────────────────────────────────
// Prompt Studio Types (for sync)
// ─────────────────────────────────────────────────────────────────────────────

export type PromptModule = {
  type: string;
  enabled?: boolean;
  config?: Record<string, any> | null;
};

export type FewShotExample = {
  inputs: Record<string, any>;
  outputs: Record<string, any>;
  explanation?: string | null;
};

export type PromptSyncStatus = 'local' | 'synced' | 'pending' | 'conflict';
export type PromptSourceSystem = 'workspace' | 'studio' | 'copilot';
export type PromptFormat = 'legacy' | 'structured';
export type StructuredPromptDefinition = Record<string, any>;

// ─────────────────────────────────────────────────────────────────────────────
// Unified Prompt Type
// ─────────────────────────────────────────────────────────────────────────────

export type Prompt = {
  // Identity - Local
  id: string;
  title: string;
  content: string;
  is_system: boolean;

  // API-aligned fields
  name?: string;
  author?: string;
  details?: string;
  system_prompt?: string | null;
  user_prompt?: string | null;
  keywords?: string[];
  createdBy?: string;
  createdAt: number;
  updatedAt?: number;

  // Optional metadata
  tags?: string[];
  favorite?: boolean;
  usageCount?: number;
  lastUsedAt?: number | null;

  // Soft delete support
  deletedAt?: number | null;

  // ─── Server Sync Fields (Prompt Studio integration) ───
  serverId?: number | null;           // Server ID if synced to Prompt Studio
  studioProjectId?: number | null;    // Prompt Studio project link
  studioPromptId?: number | null;     // Prompt Studio prompt link (same as serverId usually)

  // Sync state
  syncStatus?: PromptSyncStatus;      // 'local' | 'synced' | 'pending' | 'conflict'
  sourceSystem?: PromptSourceSystem;  // 'workspace' | 'studio' | 'copilot'
  lastSyncedAt?: number | null;       // Timestamp of last successful sync
  serverUpdatedAt?: string | null;    // Server's updated_at for conflict detection

  // ─── Studio-specific Fields (progressive disclosure) ───
  promptFormat?: PromptFormat;
  promptSchemaVersion?: number | null;
  structuredPromptDefinition?: StructuredPromptDefinition | null;
  syncPayloadVersion?: number | null;
  fewShotExamples?: FewShotExample[] | null;
  modulesConfig?: PromptModule[] | null;
  versionNumber?: number | null;
  changeDescription?: string | null;
  parentVersionId?: string | null;    // Local ID of parent version (for local branching)
  serverParentVersionId?: number | null; // Server's parent_version_id
};

export type UserSettings = {
  id: string;
  user_id: string;
};

export type Model = {
  id: string
  model_id: string
  name: string
  model_name?: string,
  model_image?: string,
  provider_id: string
  lookup: string
  model_type: string
  db_type: string
}

export type ModelNickname = {
  id: string,
  model_id: string,
  model_name: string,
  model_avatar?: string
}


export type MessageHistory = Message[];
export type ChatHistory = HistoryInfo[];
export type Prompts = Prompt[];
export type Models = Model[]
export type ModelNicknames = ModelNickname[]

// Processed media (local-only store for 'process' results)
export type ProcessedMedia = {
  id: string
  url: string
  title?: string
  content?: string
  metadata?: Record<string, any>
  createdAt: number
}

// ─────────────────────────────────────────────────────────────────────────────
// Folder System Types (mirrors tldw_server keyword_collections)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Folder entity - mirrors server's `keyword_collections` table.
 * Folders organize keywords hierarchically via parent_id.
 */
export type Folder = {
  id: number
  name: string
  parent_id: number | null
  created_at: string
  last_modified: string
  version: number
  client_id?: string
  deleted: boolean
}

/**
 * Keyword entity - mirrors server's `keywords` table.
 * Keywords are tags that can be attached to conversations/prompts.
 */
export type Keyword = {
  id: number
  keyword: string
  created_at: string
  last_modified: string
  version: number
  client_id?: string
  deleted: boolean
}

/**
 * Link between a folder and its keywords.
 * Mirrors server's `collection_keywords` junction table.
 */
export type FolderKeywordLink = {
  folder_id: number
  keyword_id: number
}

/**
 * Link between a conversation and its keywords.
 * Mirrors server's `conversation_keywords` junction table.
 */
export type ConversationKeywordLink = {
  conversation_id: string
  keyword_id: number
}

export type Folders = Folder[]
export type Keywords = Keyword[]

export type DraftStatus =
  | "pending"
  | "in_progress"
  | "reviewed"
  | "committed"
  | "discarded"

export type DraftRevision = {
  id: string
  content: string
  metadata?: Record<string, any>
  timestamp: number
  changeDescription?: string
}

export type DraftSection = {
  id: string
  label: string
  kind: "heading" | "paragraph" | "speaker_turn" | "page" | "chapter"
  startOffset: number
  endOffset: number
  content: string
  level?: number
  source: "server" | "heuristic"
  meta?: Record<string, any>
}

export type DraftSource = {
  kind: "url" | "file"
  url?: string
  fileName?: string
  mimeType?: string
  sizeBytes?: number
  lastModified?: number
}

export type DraftAsset = {
  id: string
  draftId: string
  kind: "file"
  fileName: string
  mimeType: string
  sizeBytes: number
  blob: Blob
  createdAt: number
}

export type ContentDraft = {
  id: string
  batchId: string
  source: DraftSource
  sourceAssetId?: string
  mediaType: "html" | "pdf" | "document" | "audio" | "video"
  title: string
  originalTitle?: string
  content: string
  originalContent: string
  contentFormat: "plain" | "markdown"
  originalContentFormat?: "plain" | "markdown"
  metadata: Record<string, any>
  originalMetadata?: Record<string, any>
  keywords: string[]
  sections?: DraftSection[]
  excludedSectionIds?: string[]
  sectionStrategy?: "server" | "headings" | "paragraphs" | "timestamps"
  revisions: DraftRevision[]
  processingOptions: {
    perform_analysis: boolean
    perform_chunking: boolean
    overwrite_existing: boolean
    advancedValues: Record<string, any>
  }
  status: DraftStatus
  reviewNotes?: string
  createdAt: number
  updatedAt: number
  reviewedAt?: number
  committedAt?: number
  expiresAt?: number
  analysis?: string
  prompt?: string
  originalAnalysis?: string
  originalPrompt?: string
}

export type DraftBatch = {
  id: string
  name?: string
  source: "url_list" | "file_upload" | "quick_ingest" | "manual"
  sourceDetails?: Record<string, any>
  totalItems?: number
  readyToSubmitCount?: number
  committedCount?: number
  discardedCount?: number
  createdAt: number
  updatedAt: number
  completedAt?: number
}

// ─────────────────────────────────────────────────────────────────────────────
// Audiobook Project Types
// ─────────────────────────────────────────────────────────────────────────────

export type AudiobookProjectStatus = "draft" | "in_progress" | "completed"

export type SerializedAudioChapter = {
  id: string
  title: string
  content: string
  order: number
  voiceConfig: Record<string, any>
  status: "pending" | "generating" | "completed" | "error"
  audioDuration?: number
  errorMessage?: string
}

export type AudiobookProject = {
  id: string
  title: string
  author: string
  description?: string
  coverImageAssetId?: string
  coverImageUrl?: string
  rawContent: string
  chapters: SerializedAudioChapter[]
  chapterAudioAssetIds: Record<string, string> // chapterId -> AudiobookChapterAsset.id
  defaultVoiceConfig: Record<string, any>
  status: AudiobookProjectStatus
  totalDuration?: number
  createdAt: number
  updatedAt: number
  lastOpenedAt?: number
}

export type AudiobookChapterAsset = {
  id: string
  projectId: string
  chapterId: string
  mimeType: string
  sizeBytes: number
  blob: Blob
  createdAt: number
}

// ─────────────────────────────────────────────────────────────────────────────
// STT Recording Types
// ─────────────────────────────────────────────────────────────────────────────

export type SttRecordingRow = {
  id: string
  blob: Blob
  mimeType: string
  durationMs: number
  createdAt: number
}

// ─────────────────────────────────────────────────────────────────────────────
// Media Read-Along Audio Cache Types
// ─────────────────────────────────────────────────────────────────────────────

export type MediaReadAlongAudioCacheEntry = {
  id: string
  createdAt: number
  lastUsedAt: number
  mediaId: string
  mediaKind: string
  segmentId: string
  settingsSignature: string
  textHash: string
  mimeType: string
  format: string
  blob: Blob
  sizeBytes: number
}

// ─────────────────────────────────────────────────────────────────────────────
// TTS Clip Types
// ─────────────────────────────────────────────────────────────────────────────

export type TtsClipSegment = {
  id: string
  index: number
  text: string
  format: string
  mimeType: string
  blob: Blob
  sizeBytes: number
}

export type TtsClip = {
  id: string
  createdAt: number
  provider: string
  model?: string | null
  voice?: string | null
  format?: string
  mimeType?: string
  playbackSpeed?: number
  utterance: string
  textPreview: string
  totalBytes: number
  segments: TtsClipSegment[]
  source?: "chat" | "selection"
  role?: "user" | "assistant" | "system"
  historyId?: string | null
  serverChatId?: string | null
  messageId?: string | null
  serverMessageId?: string | null
}
