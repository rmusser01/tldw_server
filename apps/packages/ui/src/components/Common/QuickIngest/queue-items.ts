import type {
  DetectedMediaType,
  WizardQueueItem,
  QueueItemValidation
} from "./types"
import { normalizeUrlForDedupe } from "@/entries/shared/ingest-payloads"
import {
  QUICK_INGEST_MAX_FILE_SIZE,
  QUICK_INGEST_MAX_FILE_SIZE_LABEL
} from "./constants"

const hostnameMatches = (hostname: string, host: string) =>
  hostname === host || hostname.endsWith(`.${host}`)

export const parseQueueUrls = (input: string): string[] =>
  input
    .split(/\r?\n|,\s+(?=https?:\/\/)|[ \t]+(?=https?:\/\/)/i)
    .map((url) => url.trim())
    .filter(Boolean)

export const isValidQueueUrl = (raw: string): boolean => {
  if (!raw.trim() || /\s|,https?:\/\//i.test(raw.trim())) return false
  try {
    const url = new URL(raw.trim())
    return url.protocol === "https:" || url.protocol === "http:"
  } catch {
    return false
  }
}

export const detectTypeFromUrl = (url: string): DetectedMediaType => {
  try {
    const parsed = new URL(url)
    const pathname = parsed.pathname.toLowerCase()
    const hostname = parsed.hostname.toLowerCase()
    // Check common file extensions in URL path
    const ext = pathname.split(".").pop() || ""
    if (["mp3", "wav", "ogg", "flac", "m4a"].includes(ext)) return "audio"
    if (["mp4", "mkv", "avi", "mov", "webm"].includes(ext)) return "video"
    if (ext === "pdf") return "pdf"
    if (["epub", "mobi"].includes(ext)) return "ebook"
    if (["docx", "txt", "rtf", "md", "markdown", "xml", "json"].includes(ext))
      return "document"
    // YouTube and common video platforms
    if (
      hostnameMatches(hostname, "youtube.com") ||
      hostnameMatches(hostname, "youtu.be")
    )
      return "video"
    if (hostnameMatches(hostname, "vimeo.com")) return "video"
    if (hostnameMatches(hostname, "soundcloud.com")) return "audio"
    if (hostnameMatches(hostname, "spotify.com")) return "audio"
    // Default for URLs is web
    return "web"
  } catch {
    return "web"
  }
}

export const validateQueueItem = (
  item: WizardQueueItem
): QueueItemValidation => {
  const errors: string[] = []

  if (item.url) {
    if (!isValidQueueUrl(item.url)) {
      errors.push("Invalid URL format")
    }
  }

  if (item.file) {
    if (item.fileSize > QUICK_INGEST_MAX_FILE_SIZE) {
      errors.push(
        `File exceeds ${QUICK_INGEST_MAX_FILE_SIZE_LABEL} quick-ingest limit`
      )
    }
  }

  if (item.detectedType === "unknown") {
    errors.push(
      "Unsupported file type. Quick Ingest supports PDF, EPUB, DOCX, TXT/RTF, Markdown, HTML, XML, JSON, audio, and video."
    )
  }

  return {
    valid: errors.length === 0,
    errors: errors.length > 0 ? errors : undefined
  }
}

export const createUrlQueueItems = (input: string): WizardQueueItem[] =>
  parseQueueUrls(input).map((url) => {
    const detectedType = detectTypeFromUrl(url)
    const item: WizardQueueItem = {
      id: crypto.randomUUID(),
      kind: "url",
      url,
      detectedType,
      icon: "Globe",
      fileSize: 0,
      validation: { valid: true }
    }
    item.validation = validateQueueItem(item)
    return item
  })

export type QueueExclusionReason = "invalid" | "unselected" | "duplicate"
export const getQueueItemExclusionReason = (
  item: WizardQueueItem,
  items: WizardQueueItem[]
): QueueExclusionReason | null => {
  if (!item.validation.valid) return "invalid"
  if (item.conferenceOverride?.selected === false) return "unselected"
  if (item.processAgain) return null
  if (
    (item.playlist?.duplicateStatus === "duplicate_existing" ||
      item.playlist?.duplicateStatus === "duplicate_in_batch") &&
    (!item.conferenceOverride?.duplicatePolicy ||
      item.conferenceOverride.duplicatePolicy === "skip")
  )
    return "duplicate"
  // ponytail: O(n²) scan for small import queues; use a keyed projection if queue sizes grow.
  const previous = items.slice(0, items.indexOf(item))
  if (
    previous.some(
      (other) =>
        other.validation.valid &&
        other.conferenceOverride?.selected !== false &&
        !(
          (other.playlist?.duplicateStatus === "duplicate_existing" ||
            other.playlist?.duplicateStatus === "duplicate_in_batch") &&
          !other.processAgain &&
          (!other.conferenceOverride?.duplicatePolicy ||
            other.conferenceOverride.duplicatePolicy === "skip")
        ) &&
        (item.url && other.url
          ? normalizeUrlForDedupe(item.url) === normalizeUrlForDedupe(other.url)
          : !item.url &&
            !other.url &&
            item.fileName &&
            item.fileName === other.fileName &&
            item.fileSize === other.fileSize)
    )
  )
    return "duplicate"
  return null
}

export const getEligibleQueueItems = (
  items: WizardQueueItem[]
): WizardQueueItem[] =>
  items.filter((item) => !getQueueItemExclusionReason(item, items))
