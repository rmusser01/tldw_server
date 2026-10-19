import { bgRequest, type BgRequestInit } from '@/services/background-proxy';
import { appendPathQuery, toAllowedPath } from '@/services/tldw/path-utils';
import type { TldwConfig } from '@/services/tldw/TldwApiClient';

export type ClaimReviewStatus = 'pending' | 'approved' | 'rejected' | 'flagged' | 'reassigned';
export type ClaimReviewItem = {
  id: number;
  media_id: number;
  media_title?: string | null;
  chunk_index: number;
  span_start?: number | null;
  span_end?: number | null;
  claim_text: string;
  review_status: ClaimReviewStatus;
  review_version: number;
  review_notes?: string | null;
  review_reason_code?: string | null;
};
export type ClaimHistoryItem = {
  id: number;
  old_status?: string | null;
  new_status: string;
  old_text?: string | null;
  new_text?: string | null;
  reviewer_id?: number | null;
  notes?: string | null;
  reason_code?: string | null;
  created_at: string;
};
export type ClaimReviewQueue = {
  items: ClaimReviewItem[];
  has_more: boolean;
  next_offset: number | null;
};
export type ClaimReviewDecision = {
  status: ClaimReviewStatus;
  review_version: number;
  notes?: string;
  reason_code?: string;
  corrected_text?: string;
};
export type ClaimBulkResult = {
  updated: number[];
  conflicts: number[];
  missing: number[];
  invalid: number[];
};
export type ClaimRequestScope = { configSnapshot: TldwConfig; abortSignal: AbortSignal } &
  Pick<BgRequestInit, 'headers'>;

export const listClaimsForReview = (
  status: ClaimReviewStatus,
  offset: number,
  scope: ClaimRequestScope
) =>
  bgRequest<ClaimReviewQueue>({
    path: appendPathQuery(
      '/api/v1/claims/review-queue',
      '?' +
        new URLSearchParams({
          status_filter: status,
          envelope: 'true',
          limit: '25',
          offset: String(offset),
        }).toString()
    ),
    method: 'GET',
    ...scope,
  });

export const getClaimForReview = (id: number, scope: ClaimRequestScope) =>
  bgRequest<ClaimReviewItem>({
    path: toAllowedPath(`/api/v1/claims/items/${id}`),
    method: 'GET',
    ...scope,
  });

export const getClaimReviewHistory = (id: number, scope: ClaimRequestScope) =>
  bgRequest<ClaimHistoryItem[]>({
    path: toAllowedPath(`/api/v1/claims/${id}/history`),
    method: 'GET',
    ...scope,
  });

export const reviewClaim = (id: number, body: ClaimReviewDecision, scope: ClaimRequestScope) =>
  bgRequest<ClaimReviewItem>({
    path: toAllowedPath(`/api/v1/claims/${id}/review`),
    method: 'PATCH',
    body,
    ...scope,
  });

export const bulkReviewClaims = (
  body: { claim_ids: number[]; status: ClaimReviewStatus; notes?: string; reason_code?: string },
  scope: ClaimRequestScope
) =>
  bgRequest<ClaimBulkResult>({
    path: '/api/v1/claims/review/bulk',
    method: 'POST',
    body,
    ...scope,
  });

/** Current source text is separate from the frozen extraction chunk and its offsets. */
export const getClaimSource = (mediaId: number, scope: ClaimRequestScope) =>
  bgRequest<{ content: { text?: string | null } }>({
    path: toAllowedPath(`/api/v1/media/${mediaId}?include_content=true&include_versions=false`),
    method: 'GET',
    maxResponseBytes: 1024 * 1024,
    ...scope,
  });
