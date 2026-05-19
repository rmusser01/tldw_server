export const QUICK_INGEST_ACCEPT_STRING =
  ".pdf,.txt,.rtf,.doc,.docx,.md,.markdown,.html,.htm,.xhtml,.xml,.json,.epub,.mp3,.wav,.m4a,.flac,.aac,.ogg,.mp4,.webm,.mkv,.mov,.avi,application/pdf,text/plain,text/markdown,text/html,text/xml,application/xml,application/xhtml+xml,application/json,application/rtf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/epub+zip,audio/*,video/*"

export const QUICK_INGEST_BUFFERED_UPLOAD_MAX_FILE_SIZE = 50 * 1024 * 1024 // 50MB buffered client upload guard
export const QUICK_INGEST_TRANSPORT_REDESIGN_FILE_SIZE = 500 * 1024 * 1024 // future direct-upload target
export const QUICK_INGEST_MAX_FILE_SIZE = QUICK_INGEST_BUFFERED_UPLOAD_MAX_FILE_SIZE
export const QUICK_INGEST_MAX_FILE_SIZE_LABEL = "50 MB"

// ---------------------------------------------------------------------------
// Duplicate / skip detection
// ---------------------------------------------------------------------------

/** Default English message shown when an item is skipped as a duplicate. */
export const DUPLICATE_SKIP_MESSAGE =
  "This item already exists in your library. Use the \u2018Deep\u2019 preset to overwrite."

/** Check if a backend result indicates a duplicate/already-exists skip. */
export const isDbMessageDuplicate = (data: Record<string, unknown> | null | undefined): boolean =>
  typeof data?.db_message === "string" &&
  (data.db_message as string).toLowerCase().includes("already exists")
