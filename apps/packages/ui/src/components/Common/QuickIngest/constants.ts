export const QUICK_INGEST_ACCEPT_STRING =
  ".pdf,.txt,.rtf,.doc,.docx,.md,.markdown,.epub,.mp3,.wav,.m4a,.flac,.aac,.ogg,.mp4,.webm,.mkv,.mov,.avi,application/pdf,text/plain,text/markdown,application/rtf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/epub+zip,audio/*,video/*"

export const QUICK_INGEST_MAX_FILE_SIZE = 500 * 1024 * 1024 // 500MB

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
