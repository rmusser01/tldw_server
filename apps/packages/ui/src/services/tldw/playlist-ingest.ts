export type PlaylistIngestContractVersion = 2;

export type PlaylistPreflightStatus =
  | "pending"
  | "running"
  | "ready"
  | "blocked"
  | "cancelled"
  | "expired";

export type PlaylistRunItemState =
  | "staged"
  | "preparing"
  | "awaiting_upload"
  | "submit_pending"
  | "queued"
  | "running"
  | "cancellation_requested"
  | "status_unavailable"
  | "terminal";

export type PlaylistRunItemOutcome =
  | "completed"
  | "included_existing"
  | "metadata_updated"
  | "skipped_existing"
  | "submit_failed"
  | "processing_failed"
  | "metadata_update_failed"
  | "cancelled";

export type PlaylistDuplicatePolicy =
  | "skip"
  | "include_existing"
  | "update_metadata_only"
  | "overwrite";

export type ApiPlaylistDisplayMetadata = {
  title?: string | null;
  channel_or_uploader?: string | null;
  duration_seconds?: number | null;
  published_at?: string | null;
  thumbnail_url?: string | null;
  playlist_id?: string | null;
  playlist_title?: string | null;
};

export type ApiPlaylistPreflightLimits = {
  max_items: number;
  global_capacity: number;
  owner_capacity: number;
};

export type ApiPlaylistPreflightCreateRequest = {
  url: string;
  max_items?: number;
  timeout_seconds?: number;
};

export type ApiPlaylistPreflightAcceptedResponse = {
  contract_version: PlaylistIngestContractVersion;
  preflight_id: string;
  status: "pending";
  status_url: string;
  items_url: string;
  expires_at: string;
  limits: ApiPlaylistPreflightLimits;
};

export type ApiPlaylistPreflightSummaryCounts = {
  playlist_title?: string | null;
  total_count?: number;
  loaded_count?: number;
  ingestible_count?: number;
  unavailable_count?: number;
  duplicate_count?: number;
  selected_count?: number;
  warnings?: unknown[];
};

export type ApiPlaylistPreflightSummaryResponse = {
  contract_version: PlaylistIngestContractVersion;
  preflight_id: string;
  status: PlaylistPreflightStatus;
  source_url: string;
  source_kind: string;
  playlist_id: string | null;
  summary: ApiPlaylistPreflightSummaryCounts | null;
  error: Record<string, string> | null;
  created_at: string;
  updated_at: string;
  expires_at: string;
};

export type ApiPlaylistPreflightItem = {
  occurrence_id: string;
  ordinal: number;
  occurrence_index_for_source: number | null;
  source_url: string | null;
  normalized_source_id: string | null;
  source_kind: string;
  availability: string | null;
  duplicate_status: string | null;
  duplicate_of_occurrence_id: string | null;
  selected_by_default: boolean | null;
  display_metadata: ApiPlaylistDisplayMetadata;
};

export type ApiPlaylistPreflightItemsPageResponse = {
  contract_version: PlaylistIngestContractVersion;
  preflight_id: string;
  items: ApiPlaylistPreflightItem[];
  next_cursor: string | null;
};

export type ApiPlaylistMaterializationItem = {
  occurrence_id: string;
  ordinal: number;
  source_url: string;
  normalized_source_id: string | null;
  source_kind: string;
  display_metadata: ApiPlaylistDisplayMetadata;
};

export type ApiPlaylistMaterializationResponse = {
  contract_version: PlaylistIngestContractVersion;
  materialization_id: string;
  preflight_id: string;
  status: "ready";
  items: ApiPlaylistMaterializationItem[];
  expires_at: string;
};

export type ApiPlaylistMaterializationCreateRequest = {
  occurrence_ids: string[];
};

export type ApiPlaylistMetadataPatch = {
  title?: string;
  author?: string;
  keywords_add?: string[];
};

export type ApiPlaylistReviewOverride = {
  duplicate_policy: string;
  metadata_patch?: ApiPlaylistMetadataPatch;
  existing_media_id?: number;
  duplicate_of_occurrence_id?: string;
};

export type ApiPlaylistMaterializedItemInput = {
  input_kind: "materialized_playlist_item";
  occurrence_id: string;
  materialization_id: string;
};

export type ApiPlaylistDirectUrlInput = {
  input_kind: "direct_url";
  occurrence_id: string;
  url: string;
  source_kind?: string;
  display_metadata: ApiPlaylistDisplayMetadata;
};

export type ApiPlaylistFileStubInput = {
  input_kind: "file_stub";
  occurrence_id: string;
  name: string;
  content_type?: string;
  size_bytes: number;
  display_metadata: ApiPlaylistDisplayMetadata;
};

export type ApiPlaylistRunInput =
  | ApiPlaylistMaterializedItemInput
  | ApiPlaylistDirectUrlInput
  | ApiPlaylistFileStubInput;

export type ApiPlaylistNewCollection = {
  name: string;
  description?: string;
  source_url?: string;
  default_tags: string[];
};

export type ApiPlaylistIngestRunCreateRequest = {
  client_request_id: string;
  inputs: ApiPlaylistRunInput[];
  review_overrides?: Record<string, ApiPlaylistReviewOverride>;
  processing_options?: Record<string, unknown>;
  playlist_summaries?: Array<Record<string, unknown>>;
  new_collection?: ApiPlaylistNewCollection;
};

export type ApiPlaylistProcessingOccurrence = {
  occurrence_id: string;
  ordinal: number;
  input_kind: string;
  source_url: string | null;
  source_kind: string | null;
  display_metadata: ApiPlaylistDisplayMetadata;
  state: "staged" | "awaiting_upload";
  outcome: null;
  job_id: null;
  batch_id: null;
  attempt: number;
  planned_collection_item_id: number | null;
};

export type ApiPlaylistIngestRunCreateResponse = {
  contract_version: PlaylistIngestContractVersion;
  run_id: string;
  status: string;
  version: number;
  status_url: string;
  items_url: string;
  events_url: string;
  processing_occurrences: ApiPlaylistProcessingOccurrence[];
};

export type ApiPlaylistIngestRunSummaryResponse = {
  contract_version: PlaylistIngestContractVersion;
  run_id: string;
  status: string;
  counts: Record<string, number>;
  version: number;
  collection_id: number | null;
  batch_ids: string[];
  created_at: string;
  updated_at: string;
  expires_at: string;
};

export type ApiPlaylistIngestRunItem = {
  occurrence_id: string;
  ordinal: number;
  input_kind: string;
  source_url: string | null;
  normalized_source_id: string | null;
  source_kind: string | null;
  display_metadata: ApiPlaylistDisplayMetadata;
  action: string;
  state: PlaylistRunItemState;
  outcome: PlaylistRunItemOutcome | null;
  progress_percent: number | null;
  progress_message: string | null;
  job_id: number | null;
  batch_id: string | null;
  media_id: number | null;
  planned_collection_item_id: number | null;
  attempt: number;
  retryable: boolean;
};

export type ApiPlaylistIngestRunItemsPageResponse = {
  contract_version: PlaylistIngestContractVersion;
  run_id: string;
  version: number;
  items: ApiPlaylistIngestRunItem[];
  next_cursor: string | null;
};

export type ApiPlaylistIngestRunRetryResponse = {
  contract_version: PlaylistIngestContractVersion;
  run_id: string;
  version: number;
  processing_occurrences: ApiPlaylistProcessingOccurrence[];
};

export type ApiPlaylistIngestRunCancelRequest = {
  occurrence_ids?: string[];
  reason?: string;
};

export type ApiPlaylistIngestRunRetryRequest = {
  occurrence_ids: string[];
};

export type ApiPlaylistIngestRunEvent = {
  event_id: number;
  run_id: string;
  occurrence_id: string | null;
  job_id: number | null;
  batch_id: string | null;
  event_type: string;
  state: PlaylistRunItemState | null;
  outcome: PlaylistRunItemOutcome | null;
  progress_percent: number | null;
  progress_message: string | null;
  occurred_at: string;
};

export type ApiPlaylistIngestResyncRequired = {
  run_id: string;
  min_event_id?: number | null;
  latest_event_id?: number | null;
};

export type ApiPlaylistIngestStatusUnavailable = {
  run_id: string;
  code: "run_status_unavailable";
};

export type PlaylistDisplayMetadata = {
  title?: string | null;
  channelOrUploader?: string | null;
  durationSeconds?: number | null;
  publishedAt?: string | null;
  thumbnailUrl?: string | null;
  playlistId?: string | null;
  playlistTitle?: string | null;
};

export type PlaylistPreflightAccepted = {
  contractVersion: PlaylistIngestContractVersion;
  preflightId: string;
  status: "pending";
  statusUrl: string;
  itemsUrl: string;
  expiresAt: string;
  limits: {
    maxItems: number;
    globalCapacity: number;
    ownerCapacity: number;
  };
};

export type PlaylistPreflightSummary = {
  contractVersion: PlaylistIngestContractVersion;
  preflightId: string;
  status: PlaylistPreflightStatus;
  sourceUrl: string;
  sourceKind: string;
  playlistId: string | null;
  summary: {
    playlistTitle: string | null;
    totalCount: number | null;
    loadedCount: number;
    ingestibleCount: number;
    unavailableCount: number;
    duplicateCount: number;
    selectedCount: number;
    warnings: PlaylistIngestWarning[];
  } | null;
  error: PlaylistIngestErrorInfo | null;
  createdAt: string;
  updatedAt: string;
  expiresAt: string;
};

export type PlaylistPreflightItem = {
  occurrenceId: string;
  ordinal: number;
  occurrenceIndexForSource: number | null;
  sourceUrl: string | null;
  normalizedSourceId: string | null;
  sourceKind: string;
  availability: string | null;
  duplicateStatus: string | null;
  duplicateOfOccurrenceId: string | null;
  selectedByDefault: boolean | null;
  displayMetadata: PlaylistDisplayMetadata;
};

export type PlaylistPreflightItemsPage = {
  contractVersion: PlaylistIngestContractVersion;
  preflightId: string;
  items: PlaylistPreflightItem[];
  nextCursor: string | null;
};

export type PlaylistMaterialization = {
  contractVersion: PlaylistIngestContractVersion;
  materializationId: string;
  preflightId: string;
  status: "ready";
  items: Array<{
    occurrenceId: string;
    ordinal: number;
    sourceUrl: string;
    normalizedSourceId: string | null;
    sourceKind: string;
    displayMetadata: PlaylistDisplayMetadata;
  }>;
  expiresAt: string;
};

export type PlaylistMetadataPatch = {
  title?: string;
  author?: string;
  keywordsAdd?: string[];
};

export type PlaylistReviewOverride = {
  duplicatePolicy: PlaylistDuplicatePolicy | string;
  metadataPatch?: PlaylistMetadataPatch;
  existingMediaId?: number;
  duplicateOfOccurrenceId?: string;
};

export type PlaylistMaterializedItemInput = {
  inputKind: "materialized_playlist_item";
  occurrenceId: string;
  materializationId: string;
};

export type PlaylistDirectUrlInput = {
  inputKind: "direct_url";
  occurrenceId: string;
  url: string;
  sourceKind?: string;
  displayMetadata?: PlaylistDisplayMetadata;
};

export type PlaylistFileStubInput = {
  inputKind: "file_stub";
  occurrenceId: string;
  /** Existing-run resume attempt; omitted when creating a new run. */
  attempt?: number;
  name: string;
  contentType?: string;
  sizeBytes: number;
  displayMetadata?: PlaylistDisplayMetadata;
};

export type PlaylistRunInput =
  | PlaylistMaterializedItemInput
  | PlaylistDirectUrlInput
  | PlaylistFileStubInput;

export type PlaylistIngestRunCreateRequest = {
  inputs: PlaylistRunInput[];
  reviewOverrides?: Record<string, PlaylistReviewOverride>;
  processingOptions?: Record<string, unknown>;
  playlistSummaries?: Array<Record<string, unknown>>;
  newCollection?: {
    name: string;
    description?: string;
    sourceUrl?: string;
    defaultTags?: string[];
  };
};

export type PlaylistIngestRunSubmissionRequest =
  PlaylistIngestRunCreateRequest & {
    clientRequestId: string;
  };

export type PlaylistPreflightCreateRequest = {
  url: string;
  maxItems?: number;
  timeoutSeconds?: number;
};

export type PlaylistIngestRunCancelRequest = {
  occurrenceIds?: string[];
  reason?: string;
};

export type PlaylistProcessingOccurrence = {
  occurrenceId: string;
  ordinal: number;
  inputKind: string;
  sourceUrl: string | null;
  sourceKind: string | null;
  displayMetadata: PlaylistDisplayMetadata;
  state: "staged" | "awaiting_upload";
  outcome: null;
  jobId: null;
  batchId: null;
  attempt: number;
  plannedCollectionItemId: number | null;
};

export type PlaylistIngestRunCreateResult = {
  contractVersion: PlaylistIngestContractVersion;
  runId: string;
  status: string;
  version: number;
  statusUrl: string;
  itemsUrl: string;
  eventsUrl: string;
  processingOccurrences: PlaylistProcessingOccurrence[];
};

export type PlaylistIngestRunSummary = {
  contractVersion: PlaylistIngestContractVersion;
  runId: string;
  status: string;
  counts: Record<string, number>;
  version: number;
  collectionId: number | null;
  batchIds: string[];
  createdAt: string;
  updatedAt: string;
  expiresAt: string;
};

export type PlaylistIngestRunItem = {
  occurrenceId: string;
  ordinal: number;
  inputKind: string;
  sourceUrl: string | null;
  normalizedSourceId: string | null;
  sourceKind: string | null;
  displayMetadata: PlaylistDisplayMetadata;
  action: string;
  state: PlaylistRunItemState;
  outcome: PlaylistRunItemOutcome | null;
  progressPercent: number | null;
  progressMessage: string | null;
  jobId: number | null;
  batchId: string | null;
  mediaId: number | null;
  plannedCollectionItemId: number | null;
  attempt: number;
  retryable: boolean;
};

export type PlaylistIngestRunItemsPage = {
  contractVersion: PlaylistIngestContractVersion;
  runId: string;
  version: number;
  items: PlaylistIngestRunItem[];
  nextCursor: string | null;
};

export type PlaylistIngestRunRetryResult = {
  contractVersion: PlaylistIngestContractVersion;
  runId: string;
  version: number;
  processingOccurrences: PlaylistProcessingOccurrence[];
};

export type PlaylistIngestRunEvent = {
  eventId: number;
  runId: string;
  occurrenceId: string | null;
  jobId: number | null;
  batchId: string | null;
  eventType: string;
  state: PlaylistRunItemState | null;
  outcome: PlaylistRunItemOutcome | null;
  progressPercent: number | null;
  progressMessage: string | null;
  occurredAt: string;
};

export type PlaylistIngestRunStreamEvent =
  | { kind: "snapshot"; summary: PlaylistIngestRunSummary }
  | { kind: "occurrence" | "run"; event: PlaylistIngestRunEvent }
  | {
      kind: "resyncRequired";
      runId: string;
      minEventId: number | null;
      latestEventId: number | null;
    }
  | {
      kind: "statusUnavailable";
      runId: string;
      code: "run_status_unavailable";
    };

export type PlaylistIngestPageParams = {
  cursor?: string | null;
  limit?: number;
};

export type PlaylistIngestRequestOptions = {
  timeoutMs?: number;
  signal?: AbortSignal;
  preferDirect?: boolean;
};

export type PlaylistIngestStreamOptions = {
  afterId?: number;
  signal?: AbortSignal;
  streamIdleTimeoutMs?: number;
};

export type PlaylistIngestWarning = {
  code: string;
};

export type PlaylistIngestErrorInfo = {
  code: PlaylistIngestPublicErrorCode;
  message: string;
  retryable: boolean;
};

export type PlaylistReviewRequiredReason =
  | "duplicate_action_required"
  | "duplicate_no_longer_present"
  | "duplicate_target_changed"
  | "invalid_duplicate_override"
  | "unknown_review_override"
  | "in_run_duplicate_requires_processing_or_skip";

export type PlaylistDuplicateEvidenceKind = "library" | "in_run" | "none";

export type PlaylistReviewRequiredRecoveryItem = {
  occurrenceId: string;
  reason: PlaylistReviewRequiredReason;
  evidence: {
    kind: PlaylistDuplicateEvidenceKind;
    existingMediaId: number | null;
    duplicateOfOccurrenceId: string | null;
  };
  allowedActions: PlaylistDuplicatePolicy[];
};

export type PlaylistIngestRecovery =
  | {
      kind: "reviewRequired";
      items: PlaylistReviewRequiredRecoveryItem[];
    }
  | {
      kind: "duplicateActionPending";
      runId: string;
    };

export type PlaylistIngestPublicErrorCode =
  | "invalid_playlist_url"
  | "playlist_not_found"
  | "playlist_private_or_auth_required"
  | "playlist_metadata_unavailable"
  | "playlist_too_large"
  | "preflight_busy"
  | "preflight_timeout"
  | "preflight_expired"
  | "preflight_cancelled"
  | "preflight_incomplete"
  | "materialization_expired"
  | "authentication_required"
  | "authorization_denied"
  | "quota_exceeded"
  | "rate_limited"
  | "worker_unavailable"
  | "server_draining"
  | "invalid_ownership"
  | "review_required"
  | "playlist_preflight_required"
  | "invalid_occurrence_selection"
  | "run_status_unavailable"
  | "duplicate_action_pending"
  | "invalid_run_request"
  | "invalid_direct_url"
  | "library_lookup_failed"
  | "ingest_run_conflict"
  | "server_unreachable"
  | "playlist_ingest_failed";

const PUBLIC_ERROR_MESSAGES: Record<PlaylistIngestPublicErrorCode, string> = {
  invalid_playlist_url: "Enter a valid YouTube playlist URL.",
  playlist_not_found: "The playlist could not be found.",
  playlist_private_or_auth_required:
    "This playlist is private or requires authentication.",
  playlist_metadata_unavailable: "Playlist details are currently unavailable.",
  playlist_too_large: "This playlist exceeds the server's configured limit.",
  preflight_busy: "The server is busy inspecting playlists. Try again shortly.",
  preflight_timeout: "Playlist inspection timed out. Try again.",
  preflight_expired: "Playlist inspection expired. Inspect the playlist again.",
  preflight_cancelled: "Playlist inspection was cancelled.",
  preflight_incomplete: "Playlist inspection is incomplete. Try again.",
  materialization_expired: "The staged playlist expired. Inspect it again.",
  authentication_required: "Sign in or update your API key, then try again.",
  authorization_denied: "You do not have permission to ingest this playlist.",
  quota_exceeded: "The server quota for this operation has been reached.",
  rate_limited: "Too many requests were sent. Try again shortly.",
  worker_unavailable: "The media worker is unavailable. Try again later.",
  server_draining: "The server is pausing new work. Try again later.",
  invalid_ownership: "This playlist ingest resource is no longer available.",
  review_required: "Review the updated duplicate choices before continuing.",
  playlist_preflight_required: "Inspect this playlist before processing it.",
  invalid_occurrence_selection:
    "The selected playlist items are no longer valid.",
  run_status_unavailable:
    "Run status is temporarily unavailable. Reconnect to try again.",
  duplicate_action_pending: "A duplicate action is still being prepared.",
  invalid_run_request: "The playlist ingest request is no longer valid.",
  invalid_direct_url:
    "One of the selected URLs is not valid for direct ingestion.",
  library_lookup_failed: "The media library could not be checked. Try again.",
  ingest_run_conflict:
    "The ingest run changed. Refresh its status and try again.",
  server_unreachable: "The server could not be reached. Try again.",
  playlist_ingest_failed: "Playlist ingestion is unavailable. Try again.",
};

const RETRYABLE_PUBLIC_ERRORS = new Set<PlaylistIngestPublicErrorCode>([
  "playlist_metadata_unavailable",
  "preflight_busy",
  "preflight_timeout",
  "preflight_incomplete",
  "rate_limited",
  "worker_unavailable",
  "server_draining",
  "run_status_unavailable",
  "duplicate_action_pending",
  "library_lookup_failed",
  "ingest_run_conflict",
  "server_unreachable",
  "playlist_ingest_failed",
]);

const PUBLIC_ERROR_CODES = new Set<PlaylistIngestPublicErrorCode>(
  Object.keys(PUBLIC_ERROR_MESSAGES) as PlaylistIngestPublicErrorCode[],
);

const REVIEW_REQUIRED_REASONS = new Set<PlaylistReviewRequiredReason>([
  "duplicate_action_required",
  "duplicate_no_longer_present",
  "duplicate_target_changed",
  "invalid_duplicate_override",
  "unknown_review_override",
  "in_run_duplicate_requires_processing_or_skip",
]);

const DUPLICATE_EVIDENCE_KINDS = new Set<PlaylistDuplicateEvidenceKind>([
  "library",
  "in_run",
  "none",
]);

const DUPLICATE_POLICIES = new Set<PlaylistDuplicatePolicy>([
  "skip",
  "include_existing",
  "update_metadata_only",
  "overwrite",
]);

const SERVER_ERROR_CODE_MAP: Record<string, PlaylistIngestPublicErrorCode> = {
  preflight_not_found: "preflight_expired",
  preflight_unavailable: "server_unreachable",
  playlist_preflight_failed: "playlist_ingest_failed",
  invalid_materialization_request: "invalid_occurrence_selection",
  ingest_run_not_found: "invalid_ownership",
  playlist_ingest_run_failed: "playlist_ingest_failed",
  invalid_run_cancel_request: "invalid_run_request",
  invalid_run_retry_request: "invalid_run_request",
  collection_planning_failed: "invalid_run_request",
  collection_planning_reconciliation_failed: "invalid_run_request",
  collection_planning_cleanup_failed: "invalid_run_request",
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

const finiteNumber = (value: unknown, fallback = 0): number =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;

const nullableNumber = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

const nullableString = (value: unknown): string | null =>
  typeof value === "string" ? value : null;

const boundedIdentifier = (value: unknown): string | null => {
  if (typeof value !== "string") return null;
  const identifier = value.trim();
  return identifier.length > 0 && identifier.length <= 255 ? identifier : null;
};

const normalizeDuplicateEvidence = (
  value: unknown,
): PlaylistReviewRequiredRecoveryItem["evidence"] | null => {
  if (!isRecord(value)) return null;
  const kind = value.kind;
  if (
    typeof kind !== "string" ||
    !DUPLICATE_EVIDENCE_KINDS.has(kind as PlaylistDuplicateEvidenceKind)
  ) {
    return null;
  }

  const existingMediaId =
    value.existing_media_id === null
      ? null
      : typeof value.existing_media_id === "number" &&
          Number.isSafeInteger(value.existing_media_id) &&
          value.existing_media_id > 0
        ? value.existing_media_id
        : null;
  if (value.existing_media_id !== null && existingMediaId === null) return null;

  const duplicateOfOccurrenceId =
    value.duplicate_of_occurrence_id === null
      ? null
      : boundedIdentifier(value.duplicate_of_occurrence_id);
  if (
    value.duplicate_of_occurrence_id !== null &&
    duplicateOfOccurrenceId === null
  ) {
    return null;
  }

  if (
    (kind === "library" &&
      (existingMediaId === null || duplicateOfOccurrenceId !== null)) ||
    (kind === "in_run" &&
      (existingMediaId !== null || duplicateOfOccurrenceId === null)) ||
    (kind === "none" &&
      (existingMediaId !== null || duplicateOfOccurrenceId !== null))
  ) {
    return null;
  }

  return {
    kind: kind as PlaylistDuplicateEvidenceKind,
    existingMediaId,
    duplicateOfOccurrenceId,
  };
};

const normalizeReviewRequiredItem = (
  value: unknown,
): PlaylistReviewRequiredRecoveryItem | null => {
  if (!isRecord(value)) return null;
  const occurrenceId = boundedIdentifier(value.occurrence_id);
  const reason = value.reason;
  const evidence = normalizeDuplicateEvidence(value.evidence);
  const allowedActions = value.allowed_actions;
  if (
    occurrenceId === null ||
    typeof reason !== "string" ||
    !REVIEW_REQUIRED_REASONS.has(reason as PlaylistReviewRequiredReason) ||
    evidence === null ||
    !Array.isArray(allowedActions) ||
    allowedActions.length > 4 ||
    allowedActions.some(
      (action) =>
        typeof action !== "string" ||
        !DUPLICATE_POLICIES.has(action as PlaylistDuplicatePolicy),
    )
  ) {
    return null;
  }
  return {
    occurrenceId,
    reason: reason as PlaylistReviewRequiredReason,
    evidence,
    allowedActions: allowedActions as PlaylistDuplicatePolicy[],
  };
};

const normalizePlaylistIngestRecovery = (
  error: unknown,
): PlaylistIngestRecovery | null => {
  if (!isRecord(error)) return null;
  const details = isRecord(error.details) ? error.details : null;
  const detail = isRecord(details?.detail) ? details.detail : null;
  if (!detail) return null;

  if (detail.code === "review_required") {
    if (
      !Array.isArray(detail.items) ||
      detail.items.length === 0 ||
      detail.items.length > 500
    ) {
      return null;
    }
    const items = detail.items.map(normalizeReviewRequiredItem);
    if (items.some((item) => item === null)) return null;
    return {
      kind: "reviewRequired",
      items: items as PlaylistReviewRequiredRecoveryItem[],
    };
  }

  if (detail.code === "duplicate_action_pending") {
    const runId = boundedIdentifier(detail.run_id);
    return runId === null ? null : { kind: "duplicateActionPending", runId };
  }

  return null;
};

const extractServerErrorCode = (value: unknown): string | null => {
  if (!isRecord(value)) return null;
  const details = isRecord(value.details) ? value.details : null;
  const detail = details?.detail;
  if (typeof detail === "string") return detail;
  if (isRecord(detail) && typeof detail.code === "string") return detail.code;
  if (typeof details?.code === "string") return details.code;
  if (typeof value.code === "string") return value.code;
  return null;
};

const publicCodeFromServerError = (
  rawCode: string | null,
  status: number | null,
  fallbackCode: PlaylistIngestPublicErrorCode,
): PlaylistIngestPublicErrorCode => {
  if (
    rawCode &&
    PUBLIC_ERROR_CODES.has(rawCode as PlaylistIngestPublicErrorCode)
  ) {
    return rawCode as PlaylistIngestPublicErrorCode;
  }
  if (rawCode && SERVER_ERROR_CODE_MAP[rawCode]) {
    return SERVER_ERROR_CODE_MAP[rawCode];
  }
  if (status === 401) return "authentication_required";
  if (status === 403) return "authorization_denied";
  if (status === 429) return "rate_limited";
  if (status === 503) return "server_unreachable";
  if (status === 0) return "server_unreachable";
  if (status === null) return fallbackCode;
  return fallbackCode;
};

export class PlaylistIngestPublicError extends Error {
  readonly code: PlaylistIngestPublicErrorCode;
  readonly status: number | null;
  readonly retryable: boolean;
  readonly recovery: PlaylistIngestRecovery | null;

  constructor(
    code: PlaylistIngestPublicErrorCode,
    status: number | null = null,
    recovery: PlaylistIngestRecovery | null = null,
  ) {
    super(PUBLIC_ERROR_MESSAGES[code]);
    this.name = "PlaylistIngestPublicError";
    this.code = code;
    this.status = status;
    this.retryable = RETRYABLE_PUBLIC_ERRORS.has(code);
    this.recovery = recovery;
  }
}

type LoadCompletePlaylistPreflightItemsOptions = {
  preflightId: string;
  summary: PlaylistPreflightSummary;
  signal: AbortSignal;
  loadPage: (
    preflightId: string,
    params: PlaylistIngestPageParams,
    options: PlaylistIngestRequestOptions,
  ) => Promise<PlaylistPreflightItemsPage>;
  pageSize?: number;
};

const throwIfPlaylistPagingAborted = (signal: AbortSignal): void => {
  if (signal.aborted) throw new DOMException("Aborted", "AbortError");
};

export const loadCompletePlaylistPreflightItems = async ({
  preflightId,
  summary,
  signal,
  loadPage,
  pageSize = 100,
}: LoadCompletePlaylistPreflightItemsOptions): Promise<
  PlaylistPreflightItem[]
> => {
  const counts = summary.summary;
  if (
    summary.contractVersion !== 2 ||
    summary.status !== "ready" ||
    summary.preflightId !== preflightId ||
    !counts ||
    !Number.isSafeInteger(counts.loadedCount) ||
    counts.loadedCount < 0
  ) {
    throw new PlaylistIngestPublicError("preflight_incomplete");
  }

  const items: PlaylistPreflightItem[] = [];
  const occurrenceIds = new Set<string>();
  const completedCursors = new Set<string>();
  let cursor: string | null = null;

  while (true) {
    throwIfPlaylistPagingAborted(signal);
    if (cursor !== null) {
      if (completedCursors.has(cursor)) {
        throw new PlaylistIngestPublicError("preflight_incomplete");
      }
      completedCursors.add(cursor);
    }
    const page = await loadPage(
      preflightId,
      {
        ...(cursor !== null ? { cursor } : {}),
        limit: pageSize,
      },
      { signal },
    );
    throwIfPlaylistPagingAborted(signal);
    if (page.contractVersion !== 2 || page.preflightId !== preflightId) {
      throw new PlaylistIngestPublicError("preflight_incomplete");
    }
    for (const item of page.items) {
      if (!item.occurrenceId || occurrenceIds.has(item.occurrenceId)) {
        throw new PlaylistIngestPublicError("preflight_incomplete");
      }
      occurrenceIds.add(item.occurrenceId);
      items.push(item);
      if (items.length > counts.loadedCount) {
        throw new PlaylistIngestPublicError("preflight_incomplete");
      }
    }
    if (page.nextCursor === null) break;
    if (page.items.length === 0 || items.length >= counts.loadedCount) {
      throw new PlaylistIngestPublicError("preflight_incomplete");
    }
    cursor = page.nextCursor;
  }

  const trustworthyTotal =
    counts.totalCount !== null &&
    Number.isSafeInteger(counts.totalCount) &&
    counts.totalCount >= 0
      ? counts.totalCount
      : null;
  if (
    items.length !== counts.loadedCount ||
    (trustworthyTotal !== null && items.length !== trustworthyTotal)
  ) {
    throw new PlaylistIngestPublicError("preflight_incomplete");
  }
  return items;
};

export const toPlaylistIngestPublicError = (
  error: unknown,
  fallbackCode: PlaylistIngestPublicErrorCode = "playlist_ingest_failed",
): PlaylistIngestPublicError => {
  if (error instanceof PlaylistIngestPublicError) return error;
  const statusValue = isRecord(error) ? error.status : null;
  const status = nullableNumber(statusValue);
  return new PlaylistIngestPublicError(
    publicCodeFromServerError(
      extractServerErrorCode(error),
      status,
      fallbackCode,
    ),
    status,
    normalizePlaylistIngestRecovery(error),
  );
};

const normalizePublicErrorInfo = (
  value: Record<string, string> | null,
): PlaylistIngestErrorInfo | null => {
  if (!value) return null;
  const rawCode = typeof value.code === "string" ? value.code : null;
  const code = publicCodeFromServerError(
    rawCode,
    null,
    "playlist_ingest_failed",
  );
  return {
    code,
    message: PUBLIC_ERROR_MESSAGES[code],
    retryable: RETRYABLE_PUBLIC_ERRORS.has(code),
  };
};

const normalizeWarnings = (warnings: unknown): PlaylistIngestWarning[] => {
  if (!Array.isArray(warnings)) return [];
  return warnings.map((warning) => {
    const rawCode =
      typeof warning === "string"
        ? warning
        : isRecord(warning) && typeof warning.code === "string"
          ? warning.code
          : "playlist_warning";
    const code = /^[a-z][a-z0-9_]{0,63}$/.test(rawCode)
      ? rawCode
      : "playlist_warning";
    return { code };
  });
};

export const normalizePlaylistDisplayMetadata = (
  value: ApiPlaylistDisplayMetadata | null | undefined,
): PlaylistDisplayMetadata => {
  if (!value) return {};
  const result: PlaylistDisplayMetadata = {};
  if ("title" in value) result.title = nullableString(value.title);
  if ("channel_or_uploader" in value) {
    result.channelOrUploader = nullableString(value.channel_or_uploader);
  }
  if ("duration_seconds" in value) {
    result.durationSeconds = nullableNumber(value.duration_seconds);
  }
  if ("published_at" in value)
    result.publishedAt = nullableString(value.published_at);
  if ("thumbnail_url" in value)
    result.thumbnailUrl = nullableString(value.thumbnail_url);
  if ("playlist_id" in value)
    result.playlistId = nullableString(value.playlist_id);
  if ("playlist_title" in value)
    result.playlistTitle = nullableString(value.playlist_title);
  return result;
};

export const toApiPlaylistDisplayMetadata = (
  value: PlaylistDisplayMetadata | null | undefined,
): ApiPlaylistDisplayMetadata => {
  if (!value) return {};
  const result: ApiPlaylistDisplayMetadata = {};
  if (value.title !== undefined) result.title = value.title;
  if (value.channelOrUploader !== undefined) {
    result.channel_or_uploader = value.channelOrUploader;
  }
  if (value.durationSeconds !== undefined)
    result.duration_seconds = value.durationSeconds;
  if (value.publishedAt !== undefined) result.published_at = value.publishedAt;
  if (value.thumbnailUrl !== undefined)
    result.thumbnail_url = value.thumbnailUrl;
  if (value.playlistId !== undefined) result.playlist_id = value.playlistId;
  if (value.playlistTitle !== undefined)
    result.playlist_title = value.playlistTitle;
  return result;
};

export const normalizePlaylistPreflightAccepted = (
  value: ApiPlaylistPreflightAcceptedResponse,
): PlaylistPreflightAccepted => ({
  contractVersion: value.contract_version,
  preflightId: value.preflight_id,
  status: value.status,
  statusUrl: value.status_url,
  itemsUrl: value.items_url,
  expiresAt: value.expires_at,
  limits: {
    maxItems: value.limits.max_items,
    globalCapacity: value.limits.global_capacity,
    ownerCapacity: value.limits.owner_capacity,
  },
});

export const normalizePlaylistPreflightSummary = (
  value: ApiPlaylistPreflightSummaryResponse,
): PlaylistPreflightSummary => ({
  contractVersion: value.contract_version,
  preflightId: value.preflight_id,
  status: value.status,
  sourceUrl: value.source_url,
  sourceKind: value.source_kind,
  playlistId: value.playlist_id,
  summary: value.summary
    ? {
        playlistTitle: nullableString(value.summary.playlist_title),
        totalCount: nullableNumber(value.summary.total_count),
        loadedCount: finiteNumber(value.summary.loaded_count, Number.NaN),
        ingestibleCount: finiteNumber(value.summary.ingestible_count),
        unavailableCount: finiteNumber(value.summary.unavailable_count),
        duplicateCount: finiteNumber(value.summary.duplicate_count),
        selectedCount: finiteNumber(value.summary.selected_count),
        warnings: normalizeWarnings(value.summary.warnings),
      }
    : null,
  error: normalizePublicErrorInfo(value.error),
  createdAt: value.created_at,
  updatedAt: value.updated_at,
  expiresAt: value.expires_at,
});

const normalizePlaylistPreflightItem = (
  value: ApiPlaylistPreflightItem,
): PlaylistPreflightItem => ({
  occurrenceId: value.occurrence_id,
  ordinal: value.ordinal,
  occurrenceIndexForSource: value.occurrence_index_for_source,
  sourceUrl: value.source_url,
  normalizedSourceId: value.normalized_source_id,
  sourceKind: value.source_kind,
  availability: value.availability,
  duplicateStatus: value.duplicate_status,
  duplicateOfOccurrenceId: value.duplicate_of_occurrence_id,
  selectedByDefault: value.selected_by_default,
  displayMetadata: normalizePlaylistDisplayMetadata(value.display_metadata),
});

export const normalizePlaylistPreflightItemsPage = (
  value: ApiPlaylistPreflightItemsPageResponse,
): PlaylistPreflightItemsPage => ({
  contractVersion: value.contract_version,
  preflightId: value.preflight_id,
  items: value.items.map(normalizePlaylistPreflightItem),
  nextCursor: value.next_cursor,
});

export const normalizePlaylistMaterialization = (
  value: ApiPlaylistMaterializationResponse,
): PlaylistMaterialization => ({
  contractVersion: value.contract_version,
  materializationId: value.materialization_id,
  preflightId: value.preflight_id,
  status: value.status,
  items: value.items.map((item) => ({
    occurrenceId: item.occurrence_id,
    ordinal: item.ordinal,
    sourceUrl: item.source_url,
    normalizedSourceId: item.normalized_source_id,
    sourceKind: item.source_kind,
    displayMetadata: normalizePlaylistDisplayMetadata(item.display_metadata),
  })),
  expiresAt: value.expires_at,
});

const toApiPlaylistRunInput = (
  value: PlaylistRunInput,
): ApiPlaylistRunInput => {
  if (value.inputKind === "materialized_playlist_item") {
    return {
      input_kind: value.inputKind,
      occurrence_id: value.occurrenceId,
      materialization_id: value.materializationId,
    };
  }
  if (value.inputKind === "direct_url") {
    const result: ApiPlaylistDirectUrlInput = {
      input_kind: value.inputKind,
      occurrence_id: value.occurrenceId,
      url: value.url,
      display_metadata: toApiPlaylistDisplayMetadata(value.displayMetadata),
    };
    if (value.sourceKind !== undefined) result.source_kind = value.sourceKind;
    return result;
  }
  const result: ApiPlaylistFileStubInput = {
    input_kind: value.inputKind,
    occurrence_id: value.occurrenceId,
    name: value.name,
    size_bytes: value.sizeBytes,
    display_metadata: toApiPlaylistDisplayMetadata(value.displayMetadata),
  };
  if (value.contentType !== undefined) result.content_type = value.contentType;
  return result;
};

const toApiPlaylistReviewOverride = (
  value: PlaylistReviewOverride,
): ApiPlaylistReviewOverride => {
  const result: ApiPlaylistReviewOverride = {
    duplicate_policy: value.duplicatePolicy,
  };
  if (value.metadataPatch) {
    const patch: ApiPlaylistMetadataPatch = {};
    if (value.metadataPatch.title !== undefined)
      patch.title = value.metadataPatch.title;
    if (value.metadataPatch.author !== undefined)
      patch.author = value.metadataPatch.author;
    if (value.metadataPatch.keywordsAdd !== undefined) {
      patch.keywords_add = value.metadataPatch.keywordsAdd;
    }
    result.metadata_patch = patch;
  }
  if (value.existingMediaId !== undefined)
    result.existing_media_id = value.existingMediaId;
  if (value.duplicateOfOccurrenceId !== undefined) {
    result.duplicate_of_occurrence_id = value.duplicateOfOccurrenceId;
  }
  return result;
};

export const toApiPlaylistIngestRunCreateRequest = (
  value: PlaylistIngestRunSubmissionRequest,
): ApiPlaylistIngestRunCreateRequest => {
  const result: ApiPlaylistIngestRunCreateRequest = {
    client_request_id: value.clientRequestId,
    inputs: value.inputs.map(toApiPlaylistRunInput),
  };
  if (value.reviewOverrides !== undefined) {
    result.review_overrides = Object.fromEntries(
      Object.entries(value.reviewOverrides).map(([occurrenceId, override]) => [
        occurrenceId,
        toApiPlaylistReviewOverride(override),
      ]),
    );
  }
  if (value.processingOptions !== undefined) {
    result.processing_options = value.processingOptions;
  }
  if (value.playlistSummaries !== undefined) {
    result.playlist_summaries = value.playlistSummaries;
  }
  if (value.newCollection !== undefined) {
    result.new_collection = {
      name: value.newCollection.name,
      default_tags: value.newCollection.defaultTags ?? [],
    };
    if (value.newCollection.description !== undefined) {
      result.new_collection.description = value.newCollection.description;
    }
    if (value.newCollection.sourceUrl !== undefined) {
      result.new_collection.source_url = value.newCollection.sourceUrl;
    }
  }
  return result;
};

const normalizePlaylistProcessingOccurrence = (
  value: ApiPlaylistProcessingOccurrence,
): PlaylistProcessingOccurrence => ({
  occurrenceId: value.occurrence_id,
  ordinal: value.ordinal,
  inputKind: value.input_kind,
  sourceUrl: value.source_url,
  sourceKind: value.source_kind,
  displayMetadata: normalizePlaylistDisplayMetadata(value.display_metadata),
  state: value.state,
  outcome: value.outcome,
  jobId: value.job_id,
  batchId: value.batch_id,
  attempt: value.attempt,
  plannedCollectionItemId: value.planned_collection_item_id,
});

export const normalizePlaylistIngestRunCreateResult = (
  value: ApiPlaylistIngestRunCreateResponse,
): PlaylistIngestRunCreateResult => ({
  contractVersion: value.contract_version,
  runId: value.run_id,
  status: value.status,
  version: value.version,
  statusUrl: value.status_url,
  itemsUrl: value.items_url,
  eventsUrl: value.events_url,
  processingOccurrences: value.processing_occurrences.map(
    normalizePlaylistProcessingOccurrence,
  ),
});

export const normalizePlaylistIngestRunSummary = (
  value: ApiPlaylistIngestRunSummaryResponse,
): PlaylistIngestRunSummary => ({
  contractVersion: value.contract_version,
  runId: value.run_id,
  status: value.status,
  counts: value.counts,
  version: value.version,
  collectionId: value.collection_id,
  batchIds: value.batch_ids,
  createdAt: value.created_at,
  updatedAt: value.updated_at,
  expiresAt: value.expires_at,
});

const normalizePlaylistIngestRunItem = (
  value: ApiPlaylistIngestRunItem,
): PlaylistIngestRunItem => ({
  occurrenceId: value.occurrence_id,
  ordinal: value.ordinal,
  inputKind: value.input_kind,
  sourceUrl: value.source_url,
  normalizedSourceId: value.normalized_source_id,
  sourceKind: value.source_kind,
  displayMetadata: normalizePlaylistDisplayMetadata(value.display_metadata),
  action: value.action,
  state: value.state,
  outcome: value.outcome,
  progressPercent: value.progress_percent,
  progressMessage: value.progress_message,
  jobId: value.job_id,
  batchId: value.batch_id,
  mediaId: value.media_id,
  plannedCollectionItemId: value.planned_collection_item_id,
  attempt: value.attempt,
  retryable: value.retryable,
});

export const normalizePlaylistIngestRunItemsPage = (
  value: ApiPlaylistIngestRunItemsPageResponse,
): PlaylistIngestRunItemsPage => ({
  contractVersion: value.contract_version,
  runId: value.run_id,
  version: value.version,
  items: value.items.map(normalizePlaylistIngestRunItem),
  nextCursor: value.next_cursor,
});

export const normalizePlaylistIngestRunRetryResult = (
  value: ApiPlaylistIngestRunRetryResponse,
): PlaylistIngestRunRetryResult => ({
  contractVersion: value.contract_version,
  runId: value.run_id,
  version: value.version,
  processingOccurrences: value.processing_occurrences.map(
    normalizePlaylistProcessingOccurrence,
  ),
});

const normalizePlaylistIngestRunEvent = (
  value: ApiPlaylistIngestRunEvent,
): PlaylistIngestRunEvent => ({
  eventId: value.event_id,
  runId: value.run_id,
  occurrenceId: value.occurrence_id,
  jobId: value.job_id,
  batchId: value.batch_id,
  eventType: value.event_type,
  state: value.state,
  outcome: value.outcome,
  progressPercent: value.progress_percent,
  progressMessage: value.progress_message,
  occurredAt: value.occurred_at,
});

const PLAYLIST_RUN_ITEM_STATES = new Set<PlaylistRunItemState>([
  "staged",
  "preparing",
  "awaiting_upload",
  "submit_pending",
  "queued",
  "running",
  "cancellation_requested",
  "status_unavailable",
  "terminal",
]);

const PLAYLIST_RUN_ITEM_OUTCOMES = new Set<PlaylistRunItemOutcome>([
  "completed",
  "included_existing",
  "metadata_updated",
  "skipped_existing",
  "submit_failed",
  "processing_failed",
  "metadata_update_failed",
  "cancelled",
]);

const isNonEmptyString = (value: unknown): value is string =>
  typeof value === "string" && value.trim().length > 0;

const isBoundedIdentifier = (value: unknown): value is string =>
  typeof value === "string" && boundedIdentifier(value) === value;

const isValidPlaylistIngestRunEvent = (
  value: Record<string, unknown>,
): value is Record<string, unknown> & ApiPlaylistIngestRunEvent =>
  Number.isSafeInteger(value.event_id) &&
  (value.event_id as number) >= 0 &&
  isBoundedIdentifier(value.run_id) &&
  (value.occurrence_id === null || isBoundedIdentifier(value.occurrence_id)) &&
  (value.job_id === null ||
    (typeof value.job_id === "number" &&
      Number.isSafeInteger(value.job_id) &&
      value.job_id > 0)) &&
  (value.batch_id === null || isBoundedIdentifier(value.batch_id)) &&
  isNonEmptyString(value.event_type) &&
  (value.state === null ||
    (typeof value.state === "string" &&
      PLAYLIST_RUN_ITEM_STATES.has(value.state as PlaylistRunItemState))) &&
  (value.outcome === null ||
    (typeof value.outcome === "string" &&
      PLAYLIST_RUN_ITEM_OUTCOMES.has(
        value.outcome as PlaylistRunItemOutcome,
      ))) &&
  (value.progress_percent === null ||
    (typeof value.progress_percent === "number" &&
      Number.isFinite(value.progress_percent) &&
      value.progress_percent >= 0 &&
      value.progress_percent <= 100)) &&
  (value.progress_message === null ||
    (typeof value.progress_message === "string" &&
      value.progress_message.length <= 1000)) &&
  isNonEmptyString(value.occurred_at);

export const parsePlaylistIngestRunStreamLine = (
  line: string,
): PlaylistIngestRunStreamEvent | null => {
  let payload: unknown;
  try {
    payload = JSON.parse(line);
  } catch {
    return null;
  }
  if (!isRecord(payload) || !isNonEmptyString(payload.run_id)) return null;

  if (
    payload.contract_version === 2 &&
    typeof payload.version === "number" &&
    isRecord(payload.counts)
  ) {
    return {
      kind: "snapshot",
      summary: normalizePlaylistIngestRunSummary(
        payload as ApiPlaylistIngestRunSummaryResponse,
      ),
    };
  }

  if (isValidPlaylistIngestRunEvent(payload)) {
    const event = normalizePlaylistIngestRunEvent(payload);
    return {
      kind: event.occurrenceId === null ? "run" : "occurrence",
      event,
    };
  }

  if (payload.code === "run_status_unavailable") {
    return {
      kind: "statusUnavailable",
      runId: payload.run_id,
      code: "run_status_unavailable",
    };
  }

  if (
    "min_event_id" in payload ||
    "latest_event_id" in payload ||
    Object.keys(payload).length === 1
  ) {
    return {
      kind: "resyncRequired",
      runId: payload.run_id,
      minEventId: nullableNumber(payload.min_event_id),
      latestEventId: nullableNumber(payload.latest_event_id),
    };
  }

  return null;
};

export const buildPlaylistIngestPageQuery = (
  params?: PlaylistIngestPageParams,
): string => {
  const query = new URLSearchParams();
  if (params?.cursor !== undefined && params.cursor !== null) {
    query.set("cursor", params.cursor);
  }
  if (params?.limit !== undefined) query.set("limit", String(params.limit));
  const encoded = query.toString();
  return encoded ? `?${encoded}` : "";
};

// Keep run-bound submissions comfortably below the backend's 500-item ceiling.
export const PLAYLIST_INGEST_SUBMIT_CHUNK_SIZE = 50;

export type PlaylistIngestRunApi = {
  createPlaylistIngestRun: (
    payload: PlaylistIngestRunSubmissionRequest,
    options?: PlaylistIngestRequestOptions,
  ) => Promise<PlaylistIngestRunCreateResult>;
  getPlaylistIngestRun: (
    runId: string,
    options?: PlaylistIngestRequestOptions,
  ) => Promise<PlaylistIngestRunSummary>;
  listPlaylistIngestRunItems: (
    runId: string,
    params?: PlaylistIngestPageParams,
    options?: PlaylistIngestRequestOptions,
  ) => Promise<PlaylistIngestRunItemsPage>;
  streamPlaylistIngestRunEvents: (
    runId: string,
    options?: PlaylistIngestStreamOptions,
  ) => AsyncGenerator<PlaylistIngestRunStreamEvent>;
  cancelPlaylistIngestRun: (
    runId: string,
    payload?: PlaylistIngestRunCancelRequest,
    options?: PlaylistIngestRequestOptions,
  ) => Promise<PlaylistIngestRunSummary>;
  retryPlaylistIngestRunItems: (
    runId: string,
    occurrenceIds: string[],
    options?: PlaylistIngestRequestOptions,
  ) => Promise<PlaylistIngestRunRetryResult>;
};

export type PlaylistIngestRunItemsSnapshot = {
  contractVersion: PlaylistIngestContractVersion;
  runId: string;
  version: number;
  items: PlaylistIngestRunItem[];
};

export const createRun = (
  api: PlaylistIngestRunApi,
  payload: PlaylistIngestRunSubmissionRequest,
  options?: PlaylistIngestRequestOptions,
): Promise<PlaylistIngestRunCreateResult> =>
  options === undefined
    ? api.createPlaylistIngestRun(payload)
    : api.createPlaylistIngestRun(payload, options);

export const getRun = (
  api: PlaylistIngestRunApi,
  runId: string,
  options?: PlaylistIngestRequestOptions,
): Promise<PlaylistIngestRunSummary> =>
  options === undefined
    ? api.getPlaylistIngestRun(runId)
    : api.getPlaylistIngestRun(runId, options);

type ListRunItemsOptions = PlaylistIngestRequestOptions & {
  pageSize?: number;
  maxVersionRestarts?: number;
};

const MAX_PLAYLIST_RUN_ITEMS = 500;
const MAX_PLAYLIST_RUN_CURSOR_LENGTH = 4096;

const runRequestOptions = (
  options: PlaylistIngestRequestOptions,
): PlaylistIngestRequestOptions | undefined =>
  options.signal === undefined &&
  options.timeoutMs === undefined &&
  options.preferDirect === undefined
    ? undefined
    : {
        ...(options.signal === undefined ? {} : { signal: options.signal }),
        ...(options.timeoutMs === undefined
          ? {}
          : { timeoutMs: options.timeoutMs }),
        ...(options.preferDirect === undefined
          ? {}
          : { preferDirect: options.preferDirect }),
      };

export const listRunItems = async (
  api: PlaylistIngestRunApi,
  runId: string,
  options: ListRunItemsOptions = {},
): Promise<PlaylistIngestRunItemsSnapshot> => {
  const pageSize =
    typeof options.pageSize === "number" &&
    Number.isSafeInteger(options.pageSize) &&
    options.pageSize > 0
      ? Math.min(options.pageSize, 500)
      : 100;
  const maxRestarts =
    typeof options.maxVersionRestarts === "number" &&
    Number.isSafeInteger(options.maxVersionRestarts) &&
    options.maxVersionRestarts >= 0
      ? options.maxVersionRestarts
      : 2;

  for (let restart = 0; restart <= maxRestarts; restart += 1) {
    const items: PlaylistIngestRunItem[] = [];
    const occurrenceIds = new Set<string>();
    const completedCursors = new Set<string>();
    let cursor: string | null = null;
    let version: number | null = null;
    let coherent = true;

    while (true) {
      const params: PlaylistIngestPageParams = {
        ...(cursor === null ? {} : { cursor }),
        limit: pageSize,
      };
      const requestOptions = runRequestOptions(options);
      const page =
        requestOptions === undefined
          ? await api.listPlaylistIngestRunItems(runId, params)
          : await api.listPlaylistIngestRunItems(runId, params, requestOptions);
      if (
        page.contractVersion !== 2 ||
        page.runId !== runId ||
        (version !== null && page.version !== version)
      ) {
        coherent = false;
        break;
      }
      version = page.version;
      for (const item of page.items) {
        if (
          items.length >= MAX_PLAYLIST_RUN_ITEMS ||
          !item.occurrenceId ||
          occurrenceIds.has(item.occurrenceId)
        ) {
          throw new PlaylistIngestPublicError("run_status_unavailable");
        }
        occurrenceIds.add(item.occurrenceId);
        items.push(item);
      }
      if (page.nextCursor === null) break;
      if (
        page.items.length === 0 ||
        typeof page.nextCursor !== "string" ||
        page.nextCursor.length === 0 ||
        page.nextCursor.length > MAX_PLAYLIST_RUN_CURSOR_LENGTH ||
        items.length >= MAX_PLAYLIST_RUN_ITEMS ||
        completedCursors.has(page.nextCursor)
      ) {
        throw new PlaylistIngestPublicError("run_status_unavailable");
      }
      completedCursors.add(page.nextCursor);
      cursor = page.nextCursor;
    }

    if (coherent && version !== null) {
      return { contractVersion: 2, runId, version, items };
    }
  }
  throw new PlaylistIngestPublicError("run_status_unavailable");
};

export const cancelRun = (
  api: PlaylistIngestRunApi,
  runId: string,
  payload?: PlaylistIngestRunCancelRequest,
  options?: PlaylistIngestRequestOptions,
): Promise<PlaylistIngestRunSummary> =>
  options === undefined
    ? api.cancelPlaylistIngestRun(runId, payload)
    : api.cancelPlaylistIngestRun(runId, payload, options);

export const retryRunItems = (
  api: PlaylistIngestRunApi,
  runId: string,
  occurrenceIds: string[],
  options?: PlaylistIngestRequestOptions,
): Promise<PlaylistIngestRunRetryResult> =>
  options === undefined
    ? api.retryPlaylistIngestRunItems(runId, occurrenceIds)
    : api.retryPlaylistIngestRunItems(runId, occurrenceIds, options);

export type PlaylistIngestRunSnapshot = {
  summary: PlaylistIngestRunSummary;
  items: PlaylistIngestRunItem[];
  lastEventId: number | null;
};

type RunSnapshotOptions = PlaylistIngestRequestOptions & {
  pageSize?: number;
  maxVersionRestarts?: number;
};

const loadRunSnapshot = async (
  api: PlaylistIngestRunApi,
  runId: string,
  options: RunSnapshotOptions = {},
): Promise<PlaylistIngestRunSnapshot> => {
  const maxRestarts = options.maxVersionRestarts ?? 2;
  for (let restart = 0; restart <= maxRestarts; restart += 1) {
    const requestOptions = runRequestOptions(options);
    const summary = await getRun(api, runId, requestOptions);
    const items = await listRunItems(api, runId, options);
    if (summary.version === items.version) {
      return { summary, items: items.items, lastEventId: null };
    }
  }
  throw new PlaylistIngestPublicError("run_status_unavailable");
};

const mergeRunSnapshotsByOccurrenceId = (
  current: PlaylistIngestRunSnapshot,
  authoritative: PlaylistIngestRunSnapshot,
): PlaylistIngestRunSnapshot => {
  const byOccurrenceId = new Map(
    authoritative.items.map((item) => [item.occurrenceId, item] as const),
  );
  const items = current.items.flatMap((item) => {
    const replacement = byOccurrenceId.get(item.occurrenceId);
    if (!replacement) return [item];
    byOccurrenceId.delete(item.occurrenceId);
    return [{ ...item, ...replacement }];
  });
  items.push(...byOccurrenceId.values());
  return {
    summary: authoritative.summary,
    items,
    lastEventId: authoritative.lastEventId,
  };
};

export const pollRunSnapshot = async (
  api: PlaylistIngestRunApi,
  runId: string,
  current?: PlaylistIngestRunSnapshot,
  options: RunSnapshotOptions = {},
): Promise<PlaylistIngestRunSnapshot> => {
  const authoritative = await loadRunSnapshot(api, runId, options);
  return current
    ? mergeRunSnapshotsByOccurrenceId(current, authoritative)
    : authoritative;
};

const applyOccurrenceEvent = (
  snapshot: PlaylistIngestRunSnapshot,
  event: PlaylistIngestRunEvent,
): PlaylistIngestRunSnapshot | null => {
  if (event.runId !== snapshot.summary.runId || event.occurrenceId === null) {
    return null;
  }
  let matched = false;
  const items = snapshot.items.map((item) => {
    if (item.occurrenceId !== event.occurrenceId) return item;
    matched = true;
    return {
      ...item,
      ...(event.state === null ? {} : { state: event.state }),
      outcome: event.outcome,
      progressPercent: event.progressPercent,
      progressMessage: event.progressMessage,
      ...(event.jobId === null ? {} : { jobId: event.jobId }),
      ...(event.batchId === null ? {} : { batchId: event.batchId }),
    };
  });
  return matched
    ? { ...snapshot, items, lastEventId: event.eventId }
    : null;
};

export const streamRunEvents = async function* (
  api: PlaylistIngestRunApi,
  initial: PlaylistIngestRunSnapshot,
  options: PlaylistIngestStreamOptions & RunSnapshotOptions = {},
): AsyncGenerator<PlaylistIngestRunSnapshot> {
  const runId = initial.summary.runId;
  let snapshot = initial;
  const lacksSafeEventBoundary =
    options.afterId === undefined && initial.lastEventId === null;
  const streamOptions: PlaylistIngestStreamOptions = {
    ...(options.afterId === undefined && initial.lastEventId !== null
      ? { afterId: initial.lastEventId }
      : options.afterId === undefined
        ? {}
        : { afterId: options.afterId }),
    ...(options.signal === undefined ? {} : { signal: options.signal }),
    ...(options.streamIdleTimeoutMs === undefined
      ? {}
      : { streamIdleTimeoutMs: options.streamIdleTimeoutMs }),
  };

  for await (const streamed of api.streamPlaylistIngestRunEvents(
    runId,
    streamOptions,
  )) {
    if (streamed.kind === "snapshot") {
      if (streamed.summary.runId !== runId) continue;
      if (streamed.summary.version !== snapshot.summary.version) {
        const lastEventId = snapshot.lastEventId;
        snapshot = await loadRunSnapshot(api, runId, options);
        snapshot.lastEventId = lastEventId;
      } else {
        snapshot = { ...snapshot, summary: streamed.summary };
      }
      yield snapshot;
      continue;
    }
    if (streamed.kind === "occurrence") {
      if (lacksSafeEventBoundary) {
        snapshot = await loadRunSnapshot(api, runId, options);
        snapshot.lastEventId = streamed.event.eventId;
        yield snapshot;
        continue;
      }
      if (
        snapshot.lastEventId !== null &&
        streamed.event.eventId <= snapshot.lastEventId
      ) {
        continue;
      }
      const updated = applyOccurrenceEvent(snapshot, streamed.event);
      if (updated) {
        snapshot = updated;
      } else {
        snapshot = await loadRunSnapshot(api, runId, options);
        snapshot.lastEventId = streamed.event.eventId;
      }
      yield snapshot;
      continue;
    }
    if (streamed.kind === "run") {
      if (streamed.event.runId !== runId) continue;
      snapshot = { ...snapshot, lastEventId: streamed.event.eventId };
      yield snapshot;
      continue;
    }
    if (streamed.kind === "resyncRequired") {
      if (streamed.runId !== runId) continue;
      snapshot = await loadRunSnapshot(api, runId, options);
      snapshot.lastEventId = streamed.latestEventId;
      yield snapshot;
      continue;
    }
    if (streamed.runId !== runId) continue;
    snapshot = {
      ...snapshot,
      items: snapshot.items.map((item) =>
        item.state === "terminal"
          ? item
          : { ...item, state: "status_unavailable" },
      ),
    };
    yield snapshot;
  }
};

export type PlaylistIngestSubmissionFile = {
  fieldName?: "files";
  name?: string;
  type?: string;
  data: ArrayBuffer | Uint8Array | number[];
};

export type PlaylistIngestSubmissionFields = Record<string, unknown> & {
  run_id: string;
  urls?: string[];
  occurrence_ids?: string[];
  attempts?: number[];
  planned_item_ids?: number[];
  file_occurrence_ids?: string[];
  file_attempts?: number[];
  file_planned_item_ids?: number[];
};

export type PlaylistIngestSubmissionRequest = {
  path: "/api/v1/media/ingest/jobs";
  method: "POST";
  fields: PlaylistIngestSubmissionFields;
  files?: PlaylistIngestSubmissionFile[];
};

export type ApiPlaylistIngestOccurrenceSubmission = {
  occurrence_id: string;
  status: string;
  accepted: boolean;
  job_id: number | null;
  batch_id: string;
  error_code: string | null;
  message: string | null;
  retryable: boolean;
  attempt: number;
};

export type ApiPlaylistIngestSubmissionResponse = {
  batch_id: string;
  jobs: unknown[];
  errors: string[];
  submissions: ApiPlaylistIngestOccurrenceSubmission[];
};

export type PlaylistIngestOccurrenceSubmission = {
  occurrenceId: string;
  status: string;
  accepted: boolean;
  jobId: number | null;
  batchId: string;
  errorCode: string | null;
  message: string | null;
  retryable: boolean;
  attempt: number;
};

export type PlaylistIngestSubmitPendingResult = {
  submissions: PlaylistIngestOccurrenceSubmission[];
  batchIds: string[];
  stopped: boolean;
  retryAfterMs: number | null;
  unsentOccurrenceIds: string[];
  error: unknown | null;
};

type SubmitPendingChunksOptions = {
  run: PlaylistIngestRunCreateResult;
  baseFields: Record<string, unknown>;
  baseFieldsByOccurrenceId?: Record<string, Record<string, unknown>>;
  filesByOccurrenceId?: Record<string, PlaylistIngestSubmissionFile>;
  /** Display-only cache accepted for caller convenience and intentionally ignored. */
  cachedSourceUrls?: Record<string, string>;
  chunkSize?: number;
  submitChunk: (
    request: PlaylistIngestSubmissionRequest,
  ) => Promise<ApiPlaylistIngestSubmissionResponse>;
  shouldStop?: () => boolean;
  isOccurrenceCancelled?: (occurrenceId: string) => boolean;
  onProgress?: (
    result: PlaylistIngestSubmitPendingResult,
  ) => void | Promise<void>;
};

const normalizeOccurrenceSubmission = (
  value: ApiPlaylistIngestOccurrenceSubmission,
): PlaylistIngestOccurrenceSubmission => ({
  occurrenceId: value.occurrence_id,
  status: value.status,
  accepted: value.accepted,
  jobId: value.job_id,
  batchId: value.batch_id,
  errorCode: value.error_code,
  message: value.message,
  retryable: value.retryable,
  attempt: value.attempt,
});

const playlistRetryAfterMs = (error: unknown): number | null => {
  if (!isRecord(error)) return null;
  if (
    typeof error.retryAfterMs === "number" &&
    Number.isFinite(error.retryAfterMs) &&
    error.retryAfterMs >= 0
  ) {
    return error.retryAfterMs;
  }
  const headers = isRecord(error.headers) ? error.headers : null;
  const raw = headers?.["retry-after"] ?? headers?.["Retry-After"];
  if (typeof raw !== "string" || !raw.trim()) return null;
  const seconds = Number(raw);
  if (Number.isFinite(seconds) && seconds >= 0) return seconds * 1000;
  const retryAt = Date.parse(raw);
  return Number.isFinite(retryAt) ? Math.max(0, retryAt - Date.now()) : null;
};

const isAmbiguousSubmissionFailure = (error: unknown): boolean => {
  if ((error as { name?: unknown } | null)?.name === "AbortError") return false;
  if (!isRecord(error)) return true;
  return (
    error.status === undefined ||
    error.status === null ||
    error.status === 0
  );
};

const boundedPlaylistChunkSize = (value: number | undefined): number =>
  typeof value === "number" && Number.isSafeInteger(value) && value > 0
    ? Math.min(value, 499)
    : PLAYLIST_INGEST_SUBMIT_CHUNK_SIZE;

/**
 * Submit only the server-returned processing occurrences. Cached queue URLs are
 * never consulted because the run response is the source authority.
 */
export const submitPendingChunks = async ({
  run,
  baseFields,
  baseFieldsByOccurrenceId = {},
  filesByOccurrenceId = {},
  chunkSize,
  submitChunk,
  shouldStop,
  isOccurrenceCancelled,
  onProgress,
}: SubmitPendingChunksOptions): Promise<PlaylistIngestSubmitPendingResult> => {
  const pending = run.processingOccurrences.filter((occurrence) =>
    occurrence.inputKind === "file_stub"
      ? occurrence.state === "awaiting_upload" &&
        filesByOccurrenceId[occurrence.occurrenceId] !== undefined
      : occurrence.state === "staged" && isNonEmptyString(occurrence.sourceUrl),
  );
  const omittedOccurrenceIds = run.processingOccurrences
    .filter((occurrence) => !pending.includes(occurrence))
    .map((occurrence) => occurrence.occurrenceId);
  const submissions: PlaylistIngestOccurrenceSubmission[] = [];
  const batchIds = new Set<string>();
  const size = boundedPlaylistChunkSize(chunkSize);
  const groupedPending = new Map<
    string,
    { fields: Record<string, unknown>; occurrences: PlaylistProcessingOccurrence[] }
  >();
  for (const occurrence of pending) {
    const fields = baseFieldsByOccurrenceId[occurrence.occurrenceId] ?? baseFields;
    const key = JSON.stringify(fields);
    const group = groupedPending.get(key);
    if (group) {
      group.occurrences.push(occurrence);
    } else {
      groupedPending.set(key, { fields, occurrences: [occurrence] });
    }
  }
  const chunks = [...groupedPending.values()].flatMap((group) => {
    const groupedChunks: Array<{
      fields: Record<string, unknown>;
      occurrences: PlaylistProcessingOccurrence[];
    }> = [];
    for (let offset = 0; offset < group.occurrences.length; offset += size) {
      groupedChunks.push({
        fields: group.fields,
        occurrences: group.occurrences.slice(offset, offset + size),
      });
    }
    return groupedChunks;
  });

  const remainingOccurrenceIds = (chunkIndex: number): string[] => [
    ...chunks
      .slice(chunkIndex)
      .flatMap((remaining) =>
        remaining.occurrences.map((occurrence) => occurrence.occurrenceId),
      )
      .filter((occurrenceId) => !isOccurrenceCancelled?.(occurrenceId)),
    ...omittedOccurrenceIds,
  ];

  for (const [chunkIndex, groupedChunk] of chunks.entries()) {
    if (shouldStop?.()) {
      return {
        submissions,
        batchIds: [...batchIds],
        stopped: true,
        retryAfterMs: null,
        unsentOccurrenceIds: remainingOccurrenceIds(chunkIndex),
        error: new DOMException("Aborted", "AbortError"),
      };
    }
    const chunk = groupedChunk.occurrences.filter(
      (occurrence) => !isOccurrenceCancelled?.(occurrence.occurrenceId),
    );
    if (chunk.length === 0) continue;
    const urls = chunk.filter(
      (occurrence) => occurrence.inputKind !== "file_stub",
    );
    const fileOccurrences = chunk.filter(
      (occurrence) => occurrence.inputKind === "file_stub",
    );
    const fields: PlaylistIngestSubmissionFields = {
      ...groupedChunk.fields,
      run_id: run.runId,
    };
    if (urls.length > 0) {
      fields.urls = urls.map((occurrence) => occurrence.sourceUrl as string);
      fields.occurrence_ids = urls.map((occurrence) => occurrence.occurrenceId);
      fields.attempts = urls.map((occurrence) => occurrence.attempt);
      if (
        urls.every((occurrence) => occurrence.plannedCollectionItemId !== null)
      ) {
        fields.planned_item_ids = urls.map(
          (occurrence) => occurrence.plannedCollectionItemId as number,
        );
      }
    }

    let files: PlaylistIngestSubmissionFile[] | undefined;
    if (fileOccurrences.length > 0) {
      fields.file_occurrence_ids = fileOccurrences.map(
        (occurrence) => occurrence.occurrenceId,
      );
      fields.file_attempts = fileOccurrences.map(
        (occurrence) => occurrence.attempt,
      );
      if (
        fileOccurrences.every(
          (occurrence) => occurrence.plannedCollectionItemId !== null,
        )
      ) {
        fields.file_planned_item_ids = fileOccurrences.map(
          (occurrence) => occurrence.plannedCollectionItemId as number,
        );
      }
      files = fileOccurrences.map((occurrence) => ({
        ...filesByOccurrenceId[occurrence.occurrenceId],
        fieldName: "files",
      }));
    }

    const request: PlaylistIngestSubmissionRequest = {
      path: "/api/v1/media/ingest/jobs",
      method: "POST",
      fields,
      ...(files ? { files } : {}),
    };
    try {
      let response: ApiPlaylistIngestSubmissionResponse;
      try {
        response = await submitChunk(request);
      } catch (error) {
        if (!isAmbiguousSubmissionFailure(error)) throw error;
        response = await submitChunk(request);
      }
      if (isNonEmptyString(response.batch_id)) batchIds.add(response.batch_id);
      for (const value of response.submissions) {
        const submission = normalizeOccurrenceSubmission(value);
        submissions.push(submission);
        if (isNonEmptyString(submission.batchId)) {
          batchIds.add(submission.batchId);
        }
      }
      await onProgress?.({
        submissions: [...submissions],
        batchIds: [...batchIds],
        stopped: false,
        retryAfterMs: null,
        unsentOccurrenceIds: remainingOccurrenceIds(chunkIndex + 1),
        error: null,
      });
      if (shouldStop?.()) {
        return {
          submissions,
          batchIds: [...batchIds],
          stopped: true,
          retryAfterMs: null,
          unsentOccurrenceIds: remainingOccurrenceIds(chunkIndex + 1),
          error: new DOMException("Aborted", "AbortError"),
        };
      }
    } catch (error) {
      return {
        submissions,
        batchIds: [...batchIds],
        stopped: true,
        retryAfterMs: playlistRetryAfterMs(error),
        unsentOccurrenceIds: [
          ...chunk.map((occurrence) => occurrence.occurrenceId),
          ...remainingOccurrenceIds(chunkIndex + 1),
        ],
        error,
      };
    }
  }

  return {
    submissions,
    batchIds: [...batchIds],
    stopped: omittedOccurrenceIds.length > 0,
    retryAfterMs: null,
    unsentOccurrenceIds: omittedOccurrenceIds,
    error:
      omittedOccurrenceIds.length > 0
        ? new Error("One or more processing occurrences could not be submitted.")
        : null,
  };
};
