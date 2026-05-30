import React, { useEffect, useReducer, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';

import { buildChatThreadPath } from '@/routes/route-paths';
import { useToast } from '@web/components/ui/ToastProvider';
import {
  approveResearchCheckpoint,
  cancelResearchRun,
  createResearchRun,
  getResearchArtifact,
  getResearchBundle,
  getResearchRun,
  listResearchRuns,
  pauseResearchRun,
  resumeResearchRun,
  subscribeResearchRunEvents,
  type ResearchArtifactManifestEntry,
  type ResearchBundle,
  type ResearchCheckpointSummary,
  type ResearchContradiction,
  type ResearchRunCreateRequest,
  type ResearchOutlineCheckpointPayload,
  type ResearchOutlineSeedSection,
  type ResearchPlanCheckpointPayload,
  type ResearchRun,
  type ResearchRunListItem,
  type ResearchRunSnapshot,
  type ResearchSourceTrust,
  type ResearchRunStreamEvent,
  type ResearchSourceInventoryItem,
  type ResearchSourcesCheckpointPayload,
  type ResearchUnsupportedClaim,
  type ResearchVerificationSummary,
} from '@web/lib/api/researchRuns';

const TRUST_ARTIFACT_NAMES = [
  'verification_summary.json',
  'unsupported_claims.json',
  'contradictions.json',
  'source_trust.json',
] as const;

const TRUST_READY_PHASES = new Set(['awaiting_outline_review', 'packaging', 'completed']);
const TRUST_INVALIDATION_PHASES = new Set(['collecting', 'synthesizing']);

type ResearchLaunchParams = {
  query: string | null;
  sourcePolicy: string | null;
  autonomyMode: string | null;
  autorun: boolean;
  runId: string | null;
  chatId: string | null;
  launchMessageId: string | null;
  followUp: ResearchRunCreateRequest['follow_up'] | null;
};

type ConsoleState = {
  snapshot: ResearchRunSnapshot | null;
  artifactContents: Record<string, unknown>;
  bundle: ResearchBundle | null;
};

type ConsoleAction =
  | { type: 'clear' }
  | { type: 'invalidate-trust' }
  | { type: 'replace-run'; run: ResearchRun }
  | { type: 'replace-snapshot'; snapshot: ResearchRunSnapshot }
  | { type: 'apply-event'; event: ResearchRunStreamEvent }
  | { type: 'store-artifact'; artifactName: string; content: unknown }
  | { type: 'store-bundle'; bundle: ResearchBundle };

const INITIAL_STATE: ConsoleState = {
  snapshot: null,
  artifactContents: {},
  bundle: null,
};

function makeSnapshotFromRun(run: ResearchRun): ResearchRunSnapshot {
  return {
    run,
    latest_event_id: 0,
    checkpoint: null,
    artifacts: [],
  };
}

function mergeArtifactEntry(
  artifacts: ResearchArtifactManifestEntry[],
  nextArtifact: ResearchArtifactManifestEntry
): ResearchArtifactManifestEntry[] {
  const remaining = artifacts.filter((artifact) => artifact.artifact_name !== nextArtifact.artifact_name);
  return [...remaining, nextArtifact].sort((left, right) =>
    left.artifact_name.localeCompare(right.artifact_name)
  );
}

function updateLatestEventId(snapshot: ResearchRunSnapshot, eventId?: number): ResearchRunSnapshot {
  if (typeof eventId !== 'number' || !Number.isFinite(eventId) || eventId <= snapshot.latest_event_id) {
    return snapshot;
  }
  return {
    ...snapshot,
    latest_event_id: eventId,
  };
}

function isSnapshotPayload(payload: unknown): payload is ResearchRunSnapshot {
  if (!payload || typeof payload !== 'object') {
    return false;
  }
  const data = payload as Record<string, unknown>;
  return typeof data.latest_event_id === 'number' && typeof data.run === 'object' && data.run !== null;
}

function isTrustArtifactName(artifactName: string): boolean {
  return (TRUST_ARTIFACT_NAMES as readonly string[]).includes(artifactName);
}

function removeTrustArtifacts(artifactContents: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(artifactContents).filter(([artifactName]) => !isTrustArtifactName(artifactName))
  );
}

function reducer(state: ConsoleState, action: ConsoleAction): ConsoleState {
  switch (action.type) {
    case 'clear':
      return INITIAL_STATE;
    case 'invalidate-trust':
      return {
        ...state,
        artifactContents: removeTrustArtifacts(state.artifactContents),
        bundle: null,
      };
    case 'replace-run':
      if (state.snapshot) {
        const preserveLiveProgress =
          state.snapshot.run.status === action.run.status &&
          state.snapshot.run.phase === action.run.phase;

        return {
          ...state,
          snapshot: {
            ...state.snapshot,
            run: {
              ...state.snapshot.run,
              ...action.run,
              progress_percent: preserveLiveProgress
                ? state.snapshot.run.progress_percent
                : action.run.progress_percent,
              progress_message: preserveLiveProgress
                ? state.snapshot.run.progress_message
                : action.run.progress_message,
            },
          },
        };
      }
      return {
        ...state,
        snapshot: makeSnapshotFromRun(action.run),
      };
    case 'replace-snapshot':
      return {
        ...state,
        snapshot: action.snapshot,
      };
    case 'apply-event': {
      if (action.event.event === 'snapshot' && isSnapshotPayload(action.event.payload)) {
        return {
          ...state,
          snapshot: action.event.payload,
        };
      }
      const current = state.snapshot;
      if (!current) {
        return state;
      }
      if (!action.event.payload || typeof action.event.payload !== 'object') {
        return {
          ...state,
          snapshot: updateLatestEventId(current, action.event.id),
        };
      }

      const payload = action.event.payload as Record<string, unknown>;
      let nextSnapshot = current;

      if (action.event.event === 'status' || action.event.event === 'terminal') {
        nextSnapshot = {
          ...nextSnapshot,
          run: {
            ...nextSnapshot.run,
            ...payload,
          },
        };
      } else if (action.event.event === 'progress') {
        nextSnapshot = {
          ...nextSnapshot,
          run: {
            ...nextSnapshot.run,
            progress_percent:
              typeof payload.progress_percent === 'number'
                ? payload.progress_percent
                : nextSnapshot.run.progress_percent,
            progress_message:
              typeof payload.progress_message === 'string'
                ? payload.progress_message
                : nextSnapshot.run.progress_message,
          },
        };
      } else if (action.event.event === 'checkpoint') {
        if (payload.checkpoint_id === null) {
          nextSnapshot = {
            ...nextSnapshot,
            checkpoint: null,
            run: {
              ...nextSnapshot.run,
              latest_checkpoint_id: null,
            },
          };
        } else {
          const checkpoint = payload as unknown as ResearchCheckpointSummary;
          nextSnapshot = {
            ...nextSnapshot,
            checkpoint,
            run: {
              ...nextSnapshot.run,
              latest_checkpoint_id: checkpoint.checkpoint_id,
            },
          };
        }
      } else if (action.event.event === 'artifact') {
        const artifact = payload as unknown as ResearchArtifactManifestEntry;
        const previousArtifact = nextSnapshot.artifacts.find(
          (currentArtifact) => currentArtifact.artifact_name === artifact.artifact_name
        );
        nextSnapshot = {
          ...nextSnapshot,
          artifacts: mergeArtifactEntry(nextSnapshot.artifacts, artifact),
        };
        const nextState = {
          ...state,
          snapshot: updateLatestEventId(nextSnapshot, action.event.id),
        };
        if (
          isTrustArtifactName(artifact.artifact_name) &&
          previousArtifact &&
          artifact.artifact_version > previousArtifact.artifact_version
        ) {
          return {
            ...nextState,
            artifactContents: removeTrustArtifacts(state.artifactContents),
            bundle: null,
          };
        }
        return nextState;
      }

      return {
        ...state,
        snapshot: updateLatestEventId(nextSnapshot, action.event.id),
      };
    }
    case 'store-artifact':
      return {
        ...state,
        artifactContents: {
          ...state.artifactContents,
          [action.artifactName]: action.content,
        },
      };
    case 'store-bundle':
      return {
        ...state,
        bundle: action.bundle,
      };
    default:
      return state;
  }
}

function upsertListItem(
  items: ResearchRunListItem[] | undefined,
  run: ResearchRun,
  query: string
): ResearchRunListItem[] {
  const nextItem: ResearchRunListItem = {
    ...run,
    query,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
  const remaining = (items ?? []).filter((item) => item.id !== run.id);
  return [nextItem, ...remaining];
}

function formatArtifactContent(content: unknown): string {
  if (typeof content === 'string') {
    return content;
  }
  return JSON.stringify(content, null, 2);
}

function asObjectRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function normalizeVerificationSummary(value: unknown): ResearchVerificationSummary | null {
  const record = asObjectRecord(value);
  if (!record) {
    return null;
  }
  return {
    supported_claim_count:
      typeof record.supported_claim_count === 'number' ? record.supported_claim_count : undefined,
    unsupported_claim_count:
      typeof record.unsupported_claim_count === 'number' ? record.unsupported_claim_count : undefined,
    contradiction_count:
      typeof record.contradiction_count === 'number' ? record.contradiction_count : undefined,
    warnings: Array.isArray(record.warnings)
      ? record.warnings.filter((item): item is string => typeof item === 'string')
      : undefined,
  };
}

function normalizeUnsupportedClaims(value: unknown): ResearchUnsupportedClaim[] {
  if (Array.isArray(value)) {
    return value.filter((item): item is ResearchUnsupportedClaim => Boolean(asObjectRecord(item))) as ResearchUnsupportedClaim[];
  }
  const record = asObjectRecord(value);
  if (record && Array.isArray(record.claims)) {
    return record.claims.filter((item): item is ResearchUnsupportedClaim => Boolean(asObjectRecord(item))) as ResearchUnsupportedClaim[];
  }
  return [];
}

function normalizeContradictions(value: unknown): ResearchContradiction[] {
  if (Array.isArray(value)) {
    return value.filter((item): item is ResearchContradiction => Boolean(asObjectRecord(item))) as ResearchContradiction[];
  }
  const record = asObjectRecord(value);
  if (record && Array.isArray(record.contradictions)) {
    return record.contradictions.filter((item): item is ResearchContradiction => Boolean(asObjectRecord(item))) as ResearchContradiction[];
  }
  return [];
}

function normalizeSourceTrust(value: unknown): ResearchSourceTrust[] {
  if (Array.isArray(value)) {
    return value.filter((item): item is ResearchSourceTrust => Boolean(asObjectRecord(item))) as ResearchSourceTrust[];
  }
  const record = asObjectRecord(value);
  if (record && Array.isArray(record.sources)) {
    return record.sources.filter((item): item is ResearchSourceTrust => Boolean(asObjectRecord(item))) as ResearchSourceTrust[];
  }
  return [];
}

type ResearchTrustView = {
  verificationSummary: ResearchVerificationSummary | null;
  unsupportedClaims: ResearchUnsupportedClaim[];
  contradictions: ResearchContradiction[];
  sourceTrust: ResearchSourceTrust[];
};

function deriveTrustView(
  bundle: ResearchBundle | null,
  artifactContents: Record<string, unknown>
): ResearchTrustView | null {
  const verificationSummary = normalizeVerificationSummary(
    bundle?.verification_summary ?? artifactContents['verification_summary.json']
  );
  const unsupportedClaims = normalizeUnsupportedClaims(
    bundle?.unsupported_claims ?? artifactContents['unsupported_claims.json']
  );
  const contradictions = normalizeContradictions(
    bundle?.contradictions ?? artifactContents['contradictions.json']
  );
  const sourceTrust = normalizeSourceTrust(
    bundle?.source_trust ?? artifactContents['source_trust.json']
  );

  if (!verificationSummary && unsupportedClaims.length === 0 && contradictions.length === 0 && sourceTrust.length === 0) {
    return null;
  }

  return {
    verificationSummary,
    unsupportedClaims,
    contradictions,
    sourceTrust,
  };
}

function parseResearchLaunchParams(search: string): ResearchLaunchParams {
  const params = new URLSearchParams(search);
  const query = params.get('query')?.trim() || null;
  const sourcePolicy = params.get('source_policy')?.trim() || null;
  const autonomyMode = params.get('autonomy_mode')?.trim() || null;
  const runId = params.get('run')?.trim() || null;
  const chatId = params.get('chat_id')?.trim() || null;
  const launchMessageId = params.get('launch_message_id')?.trim() || null;
  const followUp = parseResearchLaunchFollowUp(params.get('follow_up'));
  return {
    query,
    sourcePolicy,
    autonomyMode,
    autorun: params.get('autorun') === '1',
    runId,
    chatId,
    launchMessageId,
    followUp,
  };
}

function replaceResearchLaunchUrl(runId: string | null) {
  if (typeof window === 'undefined') {
    return;
  }
  const nextUrl = new URL(window.location.href);
  nextUrl.searchParams.delete('query');
  nextUrl.searchParams.delete('source_policy');
  nextUrl.searchParams.delete('autonomy_mode');
  nextUrl.searchParams.delete('autorun');
  nextUrl.searchParams.delete('from');
  nextUrl.searchParams.delete('chat_id');
  nextUrl.searchParams.delete('launch_message_id');
  nextUrl.searchParams.delete('follow_up');
  if (runId) {
    nextUrl.searchParams.set('run', runId);
  } else {
    nextUrl.searchParams.delete('run');
  }
  const search = nextUrl.searchParams.toString();
  window.history.replaceState({}, '', `${nextUrl.pathname}${search ? `?${search}` : ''}${nextUrl.hash}`);
}

function normalizeLaunchString(value: unknown, maxLength: number): string {
  return String(value ?? '').trim().slice(0, maxLength).trim();
}

function normalizeLaunchCount(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0
    ? Math.floor(value)
    : 0;
}

function normalizeLaunchOutline(value: unknown): NonNullable<
  NonNullable<ResearchRunCreateRequest['follow_up']>['background']
>['outline'] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter((item): item is Record<string, unknown> => Boolean(asObjectRecord(item)))
    .map((item) => {
      const title = normalizeLaunchString(item.title, 500);
      const focusArea = normalizeLaunchString(item.focus_area, 200);
      return {
        title,
        ...(focusArea ? { focus_area: focusArea } : {}),
      };
    })
    .filter((item) => item.title.length > 0)
    .slice(0, 7);
}

function normalizeLaunchClaims(value: unknown): NonNullable<
  NonNullable<ResearchRunCreateRequest['follow_up']>['background']
>['key_claims'] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter((item): item is Record<string, unknown> => Boolean(asObjectRecord(item)))
    .map((item, index) => {
      const claimId =
        normalizeLaunchString(item.claim_id, 128) || `launch-follow-up-claim-${index + 1}`;
      const text = normalizeLaunchString(item.text, 4000);
      return {
        claim_id: claimId,
        text,
      };
    })
    .filter((item) => item.text.length > 0)
    .slice(0, 5);
}

function normalizeLaunchUnresolvedQuestions(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const normalized: string[] = [];
  value.forEach((item) => {
    const candidate = normalizeLaunchString(item, 1800);
    if (candidate && !normalized.includes(candidate)) {
      normalized.push(candidate);
    }
  });
  return normalized.slice(0, 5);
}

function parseResearchLaunchFollowUp(raw: string | null): ResearchRunCreateRequest['follow_up'] | null {
  if (!raw) {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  const record = asObjectRecord(parsed);
  if (!record) {
    return null;
  }
  const question = normalizeLaunchString(record.question, 4000);
  if (!question) {
    return null;
  }
  const backgroundRecord = asObjectRecord(record.background);
  if (!backgroundRecord) {
    return { question };
  }
  const verificationSummary = asObjectRecord(backgroundRecord.verification_summary);
  const sourceTrustSummary = asObjectRecord(backgroundRecord.source_trust_summary);
  return {
    question,
    background: {
      question: normalizeLaunchString(backgroundRecord.question, 4000) || question,
      outline: normalizeLaunchOutline(backgroundRecord.outline),
      key_claims: normalizeLaunchClaims(backgroundRecord.key_claims),
      unresolved_questions: normalizeLaunchUnresolvedQuestions(backgroundRecord.unresolved_questions),
      verification_summary: {
        supported_claim_count: normalizeLaunchCount(verificationSummary?.supported_claim_count),
        unsupported_claim_count: normalizeLaunchCount(verificationSummary?.unsupported_claim_count),
      },
      source_trust_summary: {
        high_trust_count: normalizeLaunchCount(sourceTrustSummary?.high_trust_count),
        low_trust_count: normalizeLaunchCount(sourceTrustSummary?.low_trust_count),
      },
    },
  };
}

function buildResearchRunCreatePayload(
  query: string,
  launchParams: ResearchLaunchParams | null
): ResearchRunCreateRequest {
  const followUp = launchParams?.followUp
    ? {
        ...launchParams.followUp,
        question: query,
        ...(launchParams.followUp.background
          ? {
              background: {
                ...launchParams.followUp.background,
                question: query,
              },
            }
          : {}),
      }
    : null;

  return {
    query,
    source_policy: launchParams?.sourcePolicy ?? 'balanced',
    autonomy_mode: launchParams?.autonomyMode ?? 'checkpointed',
    ...(launchParams?.chatId
      ? {
          chat_handoff: {
            chat_id: launchParams.chatId,
            ...(launchParams.launchMessageId ? { launch_message_id: launchParams.launchMessageId } : {}),
          },
        }
      : {}),
    ...(followUp
      ? {
          follow_up: followUp,
        }
      : {}),
  };
}

type PlanCheckpointEditorState = {
  checkpointId: string;
  checkpointType: 'plan_review';
  querySummary: string;
  original: {
    focusAreas: string[];
    constraints: string[];
    openQuestions: string[];
    minCitedSections: number | null;
    minSources: number | null;
  };
  draft: {
    focusAreasText: string;
    constraintsText: string;
    openQuestionsText: string;
    minCitedSections: string;
    minSources: string;
  };
};

type SourcesCheckpointEditorState = {
  checkpointId: string;
  checkpointType: 'sources_review';
  sourceInventory: ResearchSourceInventoryItem[];
  draft: {
    pinnedSourceIds: string[];
    droppedSourceIds: string[];
    prioritizedSourceIds: string[];
    recollectEnabled: boolean;
    needPrimarySources: boolean;
    needContradictions: boolean;
    guidance: string;
  };
};

type OutlineCheckpointEditorState = {
  checkpointId: string;
  checkpointType: 'outline_review';
  originalSections: ResearchOutlineSeedSection[];
  availableFocusAreas: string[];
  draft: {
    sections: ResearchOutlineSeedSection[];
  };
};

type CheckpointEditorState =
  | PlanCheckpointEditorState
  | SourcesCheckpointEditorState
  | OutlineCheckpointEditorState;

type CheckpointEditorEvaluation = {
  patch: Record<string, unknown>;
  errors: string[];
};

function normalizeStringList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const normalized: string[] = [];
  value.forEach((item) => {
    const candidate = String(item ?? '').trim();
    if (candidate && !normalized.includes(candidate)) {
      normalized.push(candidate);
    }
  });
  return normalized;
}

function stringListToText(items: string[]): string {
  return items.join('\n');
}

function textToStringList(value: string): string[] {
  return value
    .split('\n')
    .map((item) => item.trim())
    .filter((item, index, all) => item.length > 0 && all.indexOf(item) === index);
}

function arraysEqual(left: string[], right: string[]): boolean {
  return left.length === right.length && left.every((item, index) => item === right[index]);
}

function parseOptionalNumber(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  const parsed = Number.parseInt(trimmed, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function asPlanPayload(payload: Record<string, unknown>): ResearchPlanCheckpointPayload {
  const stopCriteria =
    payload.stop_criteria && typeof payload.stop_criteria === 'object'
      ? (payload.stop_criteria as ResearchPlanCheckpointPayload['stop_criteria'])
      : undefined;
  return {
    query: typeof payload.query === 'string' ? payload.query : undefined,
    focus_areas: normalizeStringList(payload.focus_areas),
    constraints: normalizeStringList(payload.constraints),
    open_questions: normalizeStringList(payload.open_questions),
    source_policy: typeof payload.source_policy === 'string' ? payload.source_policy : undefined,
    autonomy_mode: typeof payload.autonomy_mode === 'string' ? payload.autonomy_mode : undefined,
    stop_criteria: {
      min_cited_sections:
        typeof stopCriteria?.min_cited_sections === 'number' ? stopCriteria.min_cited_sections : undefined,
      min_sources: typeof stopCriteria?.min_sources === 'number' ? stopCriteria.min_sources : undefined,
    },
  };
}

function asSourcesPayload(payload: Record<string, unknown>): ResearchSourcesCheckpointPayload {
  const sourceInventory = Array.isArray(payload.source_inventory)
    ? payload.source_inventory
        .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object')
        .map((item) => ({
          source_id: String(item.source_id ?? '').trim(),
          title: typeof item.title === 'string' ? item.title : undefined,
          provider: typeof item.provider === 'string' ? item.provider : undefined,
          focus_area: typeof item.focus_area === 'string' ? item.focus_area : undefined,
        }))
        .filter((item) => item.source_id.length > 0)
    : [];
  return {
    query: typeof payload.query === 'string' ? payload.query : undefined,
    focus_areas: normalizeStringList(payload.focus_areas),
    source_inventory: sourceInventory,
    collection_summary:
      payload.collection_summary && typeof payload.collection_summary === 'object'
        ? (payload.collection_summary as Record<string, unknown>)
        : undefined,
  };
}

function asOutlinePayload(payload: Record<string, unknown>): ResearchOutlineCheckpointPayload {
  const outline = payload.outline && typeof payload.outline === 'object' ? payload.outline : undefined;
  const sections = Array.isArray((outline as { sections?: unknown[] } | undefined)?.sections)
    ? ((outline as { sections?: unknown[] }).sections ?? [])
        .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object')
        .map((item) => ({
          title: String(item.title ?? '').trim(),
          focus_area: String(item.focus_area ?? '').trim(),
        }))
        .filter((item) => item.title.length > 0 && item.focus_area.length > 0)
    : [];
  return {
    outline: { sections },
    claim_count: typeof payload.claim_count === 'number' ? payload.claim_count : undefined,
    report_preview: typeof payload.report_preview === 'string' ? payload.report_preview : undefined,
    focus_areas: normalizeStringList(payload.focus_areas),
  };
}

function buildCheckpointEditor(checkpoint: ResearchCheckpointSummary | null | undefined): CheckpointEditorState | null {
  if (!checkpoint) {
    return null;
  }
  const payload = checkpoint.proposed_payload ?? {};
  if (checkpoint.checkpoint_type === 'plan_review') {
    const plan = asPlanPayload(payload);
    const minCitedSections = plan.stop_criteria?.min_cited_sections ?? null;
    const minSources = plan.stop_criteria?.min_sources ?? null;
    return {
      checkpointId: checkpoint.checkpoint_id,
      checkpointType: 'plan_review',
      querySummary: plan.query ?? '',
      original: {
        focusAreas: plan.focus_areas ?? [],
        constraints: plan.constraints ?? [],
        openQuestions: plan.open_questions ?? [],
        minCitedSections,
        minSources,
      },
      draft: {
        focusAreasText: stringListToText(plan.focus_areas ?? []),
        constraintsText: stringListToText(plan.constraints ?? []),
        openQuestionsText: stringListToText(plan.open_questions ?? []),
        minCitedSections: minCitedSections === null ? '' : String(minCitedSections),
        minSources: minSources === null ? '' : String(minSources),
      },
    };
  }
  if (checkpoint.checkpoint_type === 'sources_review') {
    const sources = asSourcesPayload(payload);
    return {
      checkpointId: checkpoint.checkpoint_id,
      checkpointType: 'sources_review',
      sourceInventory: sources.source_inventory ?? [],
      draft: {
        pinnedSourceIds: [],
        droppedSourceIds: [],
        prioritizedSourceIds: [],
        recollectEnabled: false,
        needPrimarySources: false,
        needContradictions: false,
        guidance: '',
      },
    };
  }
  if (checkpoint.checkpoint_type === 'outline_review') {
    const outline = asOutlinePayload(payload);
    const sections = outline.outline?.sections ?? [];
    const availableFocusAreas = outline.focus_areas?.length
      ? outline.focus_areas
      : sections.map((section) => section.focus_area);
    return {
      checkpointId: checkpoint.checkpoint_id,
      checkpointType: 'outline_review',
      originalSections: sections,
      availableFocusAreas,
      draft: {
        sections,
      },
    };
  }
  return null;
}

function evaluateCheckpointEditor(editor: CheckpointEditorState | null): CheckpointEditorEvaluation {
  if (!editor) {
    return { patch: {}, errors: [] };
  }

  if (editor.checkpointType === 'plan_review') {
    const focusAreas = textToStringList(editor.draft.focusAreasText);
    const constraints = textToStringList(editor.draft.constraintsText);
    const openQuestions = textToStringList(editor.draft.openQuestionsText);
    const minCitedSections = parseOptionalNumber(editor.draft.minCitedSections);
    const minSources = parseOptionalNumber(editor.draft.minSources);
    const errors: string[] = [];
    if (focusAreas.length === 0) {
      errors.push('At least one focus area is required.');
    }
    if (minCitedSections !== null && minCitedSections < 0) {
      errors.push('Minimum cited sections must be zero or greater.');
    }
    if (minSources !== null && minSources < 0) {
      errors.push('Minimum sources must be zero or greater.');
    }
    const patch: Record<string, unknown> = {
      focus_areas: focusAreas,
      constraints,
      open_questions: openQuestions,
      stop_criteria: {
        ...(minCitedSections !== null ? { min_cited_sections: minCitedSections } : {}),
        ...(minSources !== null ? { min_sources: minSources } : {}),
      },
    };
    const unchanged =
      arraysEqual(focusAreas, editor.original.focusAreas) &&
      arraysEqual(constraints, editor.original.constraints) &&
      arraysEqual(openQuestions, editor.original.openQuestions) &&
      minCitedSections === editor.original.minCitedSections &&
      minSources === editor.original.minSources;
    return {
      patch: unchanged ? {} : patch,
      errors,
    };
  }

  if (editor.checkpointType === 'sources_review') {
    const pinned = editor.draft.pinnedSourceIds;
    const dropped = editor.draft.droppedSourceIds;
    const prioritized = editor.draft.prioritizedSourceIds;
    const errors: string[] = [];
    const overlap = pinned.filter((sourceId) => dropped.includes(sourceId));
    if (overlap.length > 0) {
      errors.push('Pinned and dropped sources must be distinct.');
    }
    const invalidPrioritized = prioritized.filter((sourceId) => dropped.includes(sourceId));
    if (invalidPrioritized.length > 0) {
      errors.push('Prioritized sources cannot also be dropped.');
    }
    const patch: Record<string, unknown> = {
      pinned_source_ids: pinned,
      dropped_source_ids: dropped,
      prioritized_source_ids: prioritized,
      recollect: {
        enabled: editor.draft.recollectEnabled,
        need_primary_sources: editor.draft.needPrimarySources,
        need_contradictions: editor.draft.needContradictions,
        guidance: editor.draft.guidance.trim(),
      },
    };
    const unchanged =
      pinned.length === 0 &&
      dropped.length === 0 &&
      prioritized.length === 0 &&
      !editor.draft.recollectEnabled &&
      !editor.draft.needPrimarySources &&
      !editor.draft.needContradictions &&
      editor.draft.guidance.trim().length === 0;
    return {
      patch: unchanged ? {} : patch,
      errors,
    };
  }

  const sections = editor.draft.sections.map((section) => ({
    title: section.title.trim(),
    focus_area: section.focus_area.trim(),
  }));
  const errors: string[] = [];
  if (sections.length === 0) {
    errors.push('At least one section is required.');
  }
  if (sections.some((section) => !section.title || !section.focus_area)) {
    errors.push('Every outline section needs a title and focus area.');
  }
  const focusAreas = sections.map((section) => section.focus_area);
  if (new Set(focusAreas).size !== focusAreas.length) {
    errors.push('Each focus area can only appear once.');
  }
  const patch = {
    sections,
  };
  const unchanged =
    sections.length === editor.originalSections.length &&
    sections.every(
      (section, index) =>
        section.title === editor.originalSections[index]?.title &&
        section.focus_area === editor.originalSections[index]?.focus_area
    );
  return {
    patch: unchanged ? {} : patch,
    errors,
  };
}

function toggleItem(items: string[], value: string): string[] {
  return items.includes(value) ? items.filter((item) => item !== value) : [...items, value];
}

function moveItem<T>(items: T[], fromIndex: number, direction: -1 | 1): T[] {
  const toIndex = fromIndex + direction;
  if (toIndex < 0 || toIndex >= items.length) {
    return items;
  }
  const nextItems = [...items];
  const [moved] = nextItems.splice(fromIndex, 1);
  nextItems.splice(toIndex, 0, moved);
  return nextItems;
}

export default function ResearchRunsPage() {
  const queryClient = useQueryClient();
  const { show } = useToast();
  const [question, setQuestion] = useState('');
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [launchParams, setLaunchParams] = useState<ResearchLaunchParams | null>(null);
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE);
  const [checkpointEditor, setCheckpointEditor] = useState<CheckpointEditorState | null>(null);
  const [isApprovingCheckpoint, setIsApprovingCheckpoint] = useState(false);
  const [isLoadingTrust, setIsLoadingTrust] = useState(false);
  const [trustError, setTrustError] = useState<string | null>(null);
  const autoLaunchKeyRef = React.useRef<string | null>(null);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    const nextLaunchParams = parseResearchLaunchParams(window.location.search);
    setLaunchParams(nextLaunchParams);
    if (nextLaunchParams.runId) {
      setSelectedRunId(nextLaunchParams.runId);
    }
    if (nextLaunchParams.query) {
      setQuestion((current) => (current.trim().length > 0 ? current : nextLaunchParams.query ?? ''));
    }
  }, []);

  const runsQuery = useQuery({
    queryKey: ['research-runs'],
    queryFn: () => listResearchRuns(),
    refetchInterval: 5000,
  });

  const effectiveSelectedRunId = selectedRunId ?? runsQuery.data?.[0]?.id ?? null;
  const selectedListItem =
    runsQuery.data?.find((run) => run.id === effectiveSelectedRunId) ?? runsQuery.data?.[0] ?? null;

  useEffect(() => {
    if (!selectedRunId && runsQuery.data && runsQuery.data.length > 0) {
      setSelectedRunId(runsQuery.data[0].id);
    }
  }, [runsQuery.data, selectedRunId]);

  useEffect(() => {
    if (!launchParams?.autorun || !launchParams.query) {
      return;
    }
    const autoLaunchKey = JSON.stringify(launchParams);
    if (autoLaunchKeyRef.current === autoLaunchKey) {
      return;
    }
    autoLaunchKeyRef.current = autoLaunchKey;
    let cancelled = false;
    void (async () => {
      try {
        const createdRun = await createResearchRun(
          buildResearchRunCreatePayload(launchParams.query!, launchParams)
        );
        if (cancelled) {
          return;
        }
        queryClient.setQueryData<ResearchRunListItem[]>(['research-runs'], (current) =>
          upsertListItem(current, createdRun, launchParams.query!)
        );
        setSelectedRunId(createdRun.id);
        dispatch({ type: 'replace-run', run: createdRun });
        setQuestion('');
        replaceResearchLaunchUrl(createdRun.id);
        setLaunchParams((current) =>
          current
            ? {
                ...current,
                autorun: false,
                query: null,
                runId: createdRun.id,
                chatId: null,
                launchMessageId: null,
                followUp: null,
              }
            : current
        );
      } catch (error) {
        if (cancelled) {
          return;
        }
        show({
          title: 'Run creation failed',
          description: error instanceof Error ? error.message : 'Unable to start research run',
          variant: 'danger',
        });
        autoLaunchKeyRef.current = null;
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [launchParams, queryClient, show]);

  useEffect(() => {
    dispatch({ type: 'clear' });
    setIsLoadingTrust(false);
    setTrustError(null);
  }, [effectiveSelectedRunId]);

  const selectedRunQuery = useQuery({
    queryKey: ['research-run', effectiveSelectedRunId],
    queryFn: () => getResearchRun(effectiveSelectedRunId!),
    enabled: Boolean(effectiveSelectedRunId),
  });

  useEffect(() => {
    if (selectedRunQuery.data) {
      dispatch({ type: 'replace-run', run: selectedRunQuery.data });
    }
  }, [selectedRunQuery.data]);

  useEffect(() => {
    if (!effectiveSelectedRunId) {
      return;
    }
    return subscribeResearchRunEvents({
      sessionId: effectiveSelectedRunId,
      afterId: 0,
      onEvent: (event) => {
        dispatch({ type: 'apply-event', event });
      },
      onError: (error) => {
        const message = error instanceof Error ? error.message : 'Research stream disconnected';
        show({
          title: 'Research stream warning',
          description: message,
          variant: 'warning',
        });
      },
    });
  }, [effectiveSelectedRunId, show]);

  const selectedSnapshot = state.snapshot;
  const selectedRun = selectedSnapshot?.run ?? selectedRunQuery.data ?? selectedListItem;
  const backToChatHref = selectedRun?.chat_id
    ? buildChatThreadPath({
        serverChatId: selectedRun.chat_id,
        researchReturnRunId: selectedRun.id,
      })
    : null;
  const trustView = deriveTrustView(state.bundle, state.artifactContents);
  const loadedTrustArtifactNames = TRUST_ARTIFACT_NAMES.filter(
    (artifactName) => state.artifactContents[artifactName] !== undefined
  );
  const allTrustArtifactsLoaded = loadedTrustArtifactNames.length === TRUST_ARTIFACT_NAMES.length;
  const trustArtifacts =
    selectedSnapshot?.artifacts.filter((artifact) => isTrustArtifactName(artifact.artifact_name)) ?? [];
  const canLoadTrust =
    Boolean(selectedRun) &&
    !state.bundle &&
    !allTrustArtifactsLoaded &&
    trustArtifacts.length > 0 &&
    TRUST_READY_PHASES.has(selectedRun.phase) &&
    !isLoadingTrust;
  const checkpointEvaluation = evaluateCheckpointEditor(checkpointEditor);
  const selectedRunTitle =
    selectedListItem?.query ||
    (selectedSnapshot?.run as ResearchRun & { query?: string } | undefined)?.query ||
    selectedRun?.id ||
    'No run selected';

  useEffect(() => {
    setCheckpointEditor(buildCheckpointEditor(selectedSnapshot?.checkpoint));
  }, [selectedSnapshot?.checkpoint]);

  useEffect(() => {
    if (
      selectedRun &&
      TRUST_INVALIDATION_PHASES.has(selectedRun.phase) &&
      (state.bundle !== null || loadedTrustArtifactNames.length > 0)
    ) {
      dispatch({ type: 'invalidate-trust' });
      setIsLoadingTrust(false);
      setTrustError(null);
    }
  }, [loadedTrustArtifactNames.length, selectedRun, state.bundle]);

  async function handleCreateRun(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) {
      return;
    }
    try {
      const createdRun = await createResearchRun(
        buildResearchRunCreatePayload(trimmed, launchParams)
      );
      queryClient.setQueryData<ResearchRunListItem[]>(['research-runs'], (current) =>
        upsertListItem(current, createdRun, trimmed)
      );
      setSelectedRunId(createdRun.id);
      dispatch({ type: 'replace-run', run: createdRun });
      setQuestion('');
      replaceResearchLaunchUrl(createdRun.id);
      setLaunchParams((current) =>
        current
          ? {
              ...current,
              autorun: false,
              query: null,
              runId: createdRun.id,
              chatId: null,
              launchMessageId: null,
              followUp: null,
            }
          : current
      );
    } catch (error) {
      show({
        title: 'Run creation failed',
        description: error instanceof Error ? error.message : 'Unable to start research run',
        variant: 'danger',
      });
    }
  }

  async function handleRefreshSelectedRun() {
    if (!effectiveSelectedRunId) {
      return;
    }
    const refreshed = await selectedRunQuery.refetch();
    if (refreshed.data) {
      dispatch({ type: 'replace-run', run: refreshed.data });
    }
  }

  async function handlePauseRun() {
    if (!effectiveSelectedRunId) {
      return;
    }
    const updated = await pauseResearchRun(effectiveSelectedRunId);
    dispatch({ type: 'replace-run', run: updated });
  }

  async function handleResumeRun() {
    if (!effectiveSelectedRunId) {
      return;
    }
    const updated = await resumeResearchRun(effectiveSelectedRunId);
    dispatch({ type: 'replace-run', run: updated });
  }

  async function handleCancelRun() {
    if (!effectiveSelectedRunId) {
      return;
    }
    const updated = await cancelResearchRun(effectiveSelectedRunId);
    dispatch({ type: 'replace-run', run: updated });
  }

  async function handleApproveCheckpoint() {
    if (!effectiveSelectedRunId || !selectedSnapshot?.checkpoint || isApprovingCheckpoint) {
      return;
    }
    setIsApprovingCheckpoint(true);
    try {
      const updated = await approveResearchCheckpoint(
        effectiveSelectedRunId,
        selectedSnapshot.checkpoint.checkpoint_id,
        checkpointEvaluation.patch
      );
      dispatch({ type: 'replace-run', run: updated });
      await queryClient.invalidateQueries({ queryKey: ['research-runs'] });
      await handleRefreshSelectedRun();
    } catch (error) {
      show({
        title: 'Checkpoint approval failed',
        description: error instanceof Error ? error.message : 'Unable to approve checkpoint',
        variant: 'danger',
      });
    } finally {
      setIsApprovingCheckpoint(false);
    }
  }

  async function handleLoadArtifact(artifactName: string) {
    if (!effectiveSelectedRunId) {
      return;
    }
    const artifact = await getResearchArtifact(effectiveSelectedRunId, artifactName);
    dispatch({
      type: 'store-artifact',
      artifactName,
      content: artifact.content,
    });
  }

  async function handleLoadBundle() {
    if (!effectiveSelectedRunId) {
      return;
    }
    const bundle = await getResearchBundle(effectiveSelectedRunId);
    dispatch({
      type: 'store-bundle',
      bundle,
    });
  }

  async function handleLoadTrustDetails() {
    if (!effectiveSelectedRunId) {
      return;
    }
    const missingArtifacts = TRUST_ARTIFACT_NAMES.filter(
      (artifactName) => state.artifactContents[artifactName] === undefined
    );
    if (missingArtifacts.length === 0) {
      return;
    }
    setIsLoadingTrust(true);
    setTrustError(null);
    try {
      const loadedArtifacts = await Promise.all(
        missingArtifacts.map(async (artifactName) => {
          const artifact = await getResearchArtifact(effectiveSelectedRunId, artifactName);
          return { artifactName, content: artifact.content };
        })
      );
      loadedArtifacts.forEach(({ artifactName, content }) => {
        dispatch({
          type: 'store-artifact',
          artifactName,
          content,
        });
      });
    } catch (error) {
      setTrustError(error instanceof Error ? error.message : 'Unable to load trust details');
    } finally {
      setIsLoadingTrust(false);
    }
  }

  const canPause = selectedRun && selectedRun.control_state === 'running' && selectedRun.status !== 'completed';
  const canResume = selectedRun?.control_state === 'paused';
  const canCancel = selectedRun && !['completed', 'failed', 'cancelled'].includes(selectedRun.status);
  const canLoadBundle = selectedRun?.status === 'completed';
  const canApproveCheckpoint =
    Boolean(selectedSnapshot?.checkpoint) &&
    checkpointEvaluation.errors.length === 0 &&
    !isApprovingCheckpoint;

  function updatePlanDraft(
    field: keyof PlanCheckpointEditorState['draft'],
    value: string
  ) {
    setCheckpointEditor((current) => {
      if (!current || current.checkpointType !== 'plan_review') {
        return current;
      }
      return {
        ...current,
        draft: {
          ...current.draft,
          [field]: value,
        },
      };
    });
  }

  function toggleSourceState(kind: 'pinnedSourceIds' | 'droppedSourceIds' | 'prioritizedSourceIds', sourceId: string) {
    setCheckpointEditor((current) => {
      if (!current || current.checkpointType !== 'sources_review') {
        return current;
      }
      const nextItems = toggleItem(current.draft[kind], sourceId);
      return {
        ...current,
        draft: {
          ...current.draft,
          [kind]: nextItems,
        },
      };
    });
  }

  function updateSourcesDraft(
    field:
      | 'recollectEnabled'
      | 'needPrimarySources'
      | 'needContradictions'
      | 'guidance',
    value: boolean | string
  ) {
    setCheckpointEditor((current) => {
      if (!current || current.checkpointType !== 'sources_review') {
        return current;
      }
      return {
        ...current,
        draft: {
          ...current.draft,
          [field]: value,
        },
      };
    });
  }

  function updateOutlineSectionTitle(index: number, title: string) {
    setCheckpointEditor((current) => {
      if (!current || current.checkpointType !== 'outline_review') {
        return current;
      }
      const sections = current.draft.sections.map((section, sectionIndex) =>
        sectionIndex === index ? { ...section, title } : section
      );
      return {
        ...current,
        draft: {
          sections,
        },
      };
    });
  }

  function moveOutlineSection(index: number, direction: -1 | 1) {
    setCheckpointEditor((current) => {
      if (!current || current.checkpointType !== 'outline_review') {
        return current;
      }
      return {
        ...current,
        draft: {
          sections: moveItem(current.draft.sections, index, direction),
        },
      };
    });
  }

  function removeOutlineSection(index: number) {
    setCheckpointEditor((current) => {
      if (!current || current.checkpointType !== 'outline_review') {
        return current;
      }
      return {
        ...current,
        draft: {
          sections: current.draft.sections.filter((_, sectionIndex) => sectionIndex !== index),
        },
      };
    });
  }

  function addOutlineFocusArea(focusArea: string) {
    setCheckpointEditor((current) => {
      if (!current || current.checkpointType !== 'outline_review') {
        return current;
      }
      if (current.draft.sections.some((section) => section.focus_area === focusArea)) {
        return current;
      }
      return {
        ...current,
        draft: {
          sections: [
            ...current.draft.sections,
            {
              title: focusArea.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()),
              focus_area: focusArea,
            },
          ],
        },
      };
    });
  }

  return (
    <div className="min-h-screen bg-bg text-foreground">
      <main className="mx-auto flex max-w-7xl flex-col gap-6 px-4 py-6 lg:flex-row lg:px-8">
        <section className="w-full rounded-2xl border border-border bg-card/95 p-5 shadow-sm lg:max-w-md">
          <div className="space-y-1">
            <p className="text-xs uppercase tracking-[0.24em] text-muted-foreground">Deep Research</p>
            <h1 className="text-2xl font-semibold">Run console</h1>
            <p className="text-sm text-muted-foreground">
              Create long-running research runs and inspect them live.
            </p>
          </div>

          <form className="mt-5 space-y-3" onSubmit={handleCreateRun}>
            <label className="block text-sm font-medium" htmlFor="research-question">
              Research question
            </label>
            <textarea
              id="research-question"
              className="min-h-28 w-full rounded-xl border border-border bg-bg px-3 py-2 text-sm outline-none transition focus:border-primary"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
            />
            <button
              type="submit"
              className="inline-flex rounded-full bg-primary px-4 py-2 text-sm font-medium text-white transition hover:opacity-90"
            >
              Start run
            </button>
          </form>

          <div className="mt-6 space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                Newly created runs
              </h2>
              {runsQuery.isFetching && <span className="text-xs text-muted-foreground">Refreshing…</span>}
            </div>
            <div className="space-y-2">
              {runsQuery.data?.map((run) => {
                const selected = run.id === effectiveSelectedRunId;
                return (
                  <button
                    key={run.id}
                    type="button"
                    className={`w-full rounded-2xl border px-4 py-3 text-left transition ${
                      selected
                        ? 'border-primary bg-primary/5 shadow-sm'
                        : 'border-border bg-bg/70 hover:border-primary/40'
                    }`}
                    onClick={() => setSelectedRunId(run.id)}
                  >
                    <div className="text-sm font-medium">{run.query}</div>
                    <div className="mt-1 text-xs text-muted-foreground">
                      {run.status} · {run.phase} · {run.control_state}
                    </div>
                    {run.progress_message && (
                      <div className="mt-2 text-sm text-muted-foreground">{run.progress_message}</div>
                    )}
                  </button>
                );
              })}
              {!runsQuery.isLoading && (!runsQuery.data || runsQuery.data.length === 0) && (
                <div className="rounded-2xl border border-dashed border-border px-4 py-6 text-sm text-muted-foreground">
                  No research runs yet.
                </div>
              )}
            </div>
          </div>
        </section>

        <section className="flex-1 rounded-2xl border border-border bg-card p-5 shadow-sm">
          <div className="flex flex-col gap-3 border-b border-border pb-4 md:flex-row md:items-start md:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-muted-foreground">Selected run</p>
              <h2 className="mt-1 text-2xl font-semibold">Selected run: {selectedRunTitle}</h2>
              {selectedRun && (
                <p className="mt-2 text-sm text-muted-foreground">
                  {selectedRun.status} · {selectedRun.phase} · {selectedRun.control_state}
                </p>
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              {backToChatHref && (
                <a
                  className="rounded-full border border-border px-3 py-2 text-sm hover:bg-muted"
                  href={backToChatHref}
                >
                  Back to Chat
                </a>
              )}
              <button
                type="button"
                className="rounded-full border border-border px-3 py-2 text-sm hover:bg-muted"
                onClick={handleRefreshSelectedRun}
              >
                Refresh selected run
              </button>
              <button
                type="button"
                className="rounded-full border border-border px-3 py-2 text-sm hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
                onClick={handlePauseRun}
                disabled={!canPause}
              >
                Pause
              </button>
              <button
                type="button"
                className="rounded-full border border-border px-3 py-2 text-sm hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
                onClick={handleResumeRun}
                disabled={!canResume}
              >
                Resume
              </button>
              <button
                type="button"
                className="rounded-full border border-danger/30 px-3 py-2 text-sm text-danger hover:bg-danger/10 disabled:cursor-not-allowed disabled:opacity-50"
                onClick={handleCancelRun}
                disabled={!canCancel}
              >
                Cancel
              </button>
            </div>
          </div>

          {!selectedRun && (
            <div className="py-10 text-sm text-muted-foreground">Select a research run to inspect it.</div>
          )}

          {selectedRun && (
            <div className="space-y-6 py-5">
              <section className="rounded-2xl border border-border bg-bg/70 p-4">
                <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                  Live status
                </h3>
                <div className="mt-3 grid gap-3 md:grid-cols-3">
                  <div className="rounded-xl border border-border bg-card px-3 py-3">
                    <div className="text-xs text-muted-foreground">Phase</div>
                    <div className="mt-1 text-sm font-medium">{selectedRun.phase}</div>
                  </div>
                  <div className="rounded-xl border border-border bg-card px-3 py-3">
                    <div className="text-xs text-muted-foreground">Progress</div>
                    <div className="mt-1 text-sm font-medium">
                      {selectedRun.progress_percent ?? 0}%
                    </div>
                  </div>
                  <div className="rounded-xl border border-border bg-card px-3 py-3">
                    <div className="text-xs text-muted-foreground">Message</div>
                    <div className="mt-1 text-sm font-medium">
                      {selectedRun.progress_message || 'Waiting for updates'}
                    </div>
                  </div>
                </div>
              </section>

              <section className="rounded-2xl border border-border bg-bg/70 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                      Checkpoint
                    </h3>
                    {selectedSnapshot?.checkpoint ? (
                      <p className="mt-2 text-sm font-medium">
                        {selectedSnapshot.checkpoint.checkpoint_type}
                      </p>
                    ) : (
                      <p className="mt-2 text-sm text-muted-foreground">No checkpoint awaiting review.</p>
                    )}
                  </div>
                  <button
                    type="button"
                    className="rounded-full bg-primary px-4 py-2 text-sm font-medium text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
                    onClick={handleApproveCheckpoint}
                    disabled={!canApproveCheckpoint}
                  >
                    {isApprovingCheckpoint ? 'Approving…' : 'Approve checkpoint'}
                  </button>
                </div>
                {selectedSnapshot?.checkpoint && (
                  <div className="mt-4 space-y-4">
                    {checkpointEvaluation.errors.length > 0 && (
                      <div className="rounded-xl border border-danger/30 bg-danger/5 px-3 py-3 text-sm text-danger">
                        {checkpointEvaluation.errors.map((error) => (
                          <div key={error}>{error}</div>
                        ))}
                      </div>
                    )}

                    {checkpointEditor?.checkpointType === 'plan_review' && (
                      <div className="space-y-4">
                        <div className="rounded-xl border border-border bg-card px-3 py-3 text-sm text-muted-foreground">
                          Query: {checkpointEditor.querySummary || 'No query summary'}
                        </div>
                        <label className="block text-sm font-medium" htmlFor="plan-focus-areas">
                          Focus areas
                        </label>
                        <textarea
                          id="plan-focus-areas"
                          className="min-h-24 w-full rounded-xl border border-border bg-card px-3 py-2 text-sm outline-none transition focus:border-primary"
                          value={checkpointEditor.draft.focusAreasText}
                          onChange={(event) => updatePlanDraft('focusAreasText', event.target.value)}
                        />
                        <label className="block text-sm font-medium" htmlFor="plan-constraints">
                          Constraints
                        </label>
                        <textarea
                          id="plan-constraints"
                          className="min-h-20 w-full rounded-xl border border-border bg-card px-3 py-2 text-sm outline-none transition focus:border-primary"
                          value={checkpointEditor.draft.constraintsText}
                          onChange={(event) => updatePlanDraft('constraintsText', event.target.value)}
                        />
                        <label className="block text-sm font-medium" htmlFor="plan-open-questions">
                          Open questions
                        </label>
                        <textarea
                          id="plan-open-questions"
                          className="min-h-20 w-full rounded-xl border border-border bg-card px-3 py-2 text-sm outline-none transition focus:border-primary"
                          value={checkpointEditor.draft.openQuestionsText}
                          onChange={(event) => updatePlanDraft('openQuestionsText', event.target.value)}
                        />
                        <div className="grid gap-3 md:grid-cols-2">
                          <label className="block text-sm font-medium" htmlFor="plan-min-cited-sections">
                            Minimum cited sections
                          </label>
                          <input
                            id="plan-min-cited-sections"
                            type="number"
                            min={0}
                            className="w-full rounded-xl border border-border bg-card px-3 py-2 text-sm outline-none transition focus:border-primary"
                            value={checkpointEditor.draft.minCitedSections}
                            onChange={(event) => updatePlanDraft('minCitedSections', event.target.value)}
                          />
                          <label className="block text-sm font-medium" htmlFor="plan-min-sources">
                            Minimum sources
                          </label>
                          <input
                            id="plan-min-sources"
                            type="number"
                            min={0}
                            className="w-full rounded-xl border border-border bg-card px-3 py-2 text-sm outline-none transition focus:border-primary"
                            value={checkpointEditor.draft.minSources}
                            onChange={(event) => updatePlanDraft('minSources', event.target.value)}
                          />
                        </div>
                      </div>
                    )}

                    {checkpointEditor?.checkpointType === 'sources_review' && (
                      <div className="space-y-3">
                        {checkpointEditor.sourceInventory.map((source) => {
                          const pinned = checkpointEditor.draft.pinnedSourceIds.includes(source.source_id);
                          const dropped = checkpointEditor.draft.droppedSourceIds.includes(source.source_id);
                          const prioritized = checkpointEditor.draft.prioritizedSourceIds.includes(source.source_id);
                          const sourceLabel = source.title || source.source_id;
                          return (
                            <div
                              key={source.source_id}
                              className="rounded-xl border border-border bg-card px-3 py-3"
                            >
                              <div className="text-sm font-medium">{sourceLabel}</div>
                              <div className="mt-1 text-xs text-muted-foreground">
                                {source.provider || 'unknown provider'}
                                {source.focus_area ? ` · ${source.focus_area}` : ''}
                              </div>
                              <div className="mt-3 flex flex-wrap gap-2">
                                <button
                                  type="button"
                                  className={`rounded-full border px-3 py-2 text-sm ${
                                    pinned ? 'border-primary bg-primary/10 text-primary' : 'border-border hover:bg-muted'
                                  }`}
                                  onClick={() => toggleSourceState('pinnedSourceIds', source.source_id)}
                                  aria-pressed={pinned}
                                  aria-label={`Pin ${sourceLabel}`}
                                >
                                  Pin
                                </button>
                                <button
                                  type="button"
                                  className={`rounded-full border px-3 py-2 text-sm ${
                                    dropped ? 'border-danger/40 bg-danger/10 text-danger' : 'border-border hover:bg-muted'
                                  }`}
                                  onClick={() => toggleSourceState('droppedSourceIds', source.source_id)}
                                  aria-pressed={dropped}
                                  aria-label={`Drop ${sourceLabel}`}
                                >
                                  Drop
                                </button>
                                <button
                                  type="button"
                                  className={`rounded-full border px-3 py-2 text-sm ${
                                    prioritized
                                      ? 'border-primary bg-primary/10 text-primary'
                                      : 'border-border hover:bg-muted'
                                  }`}
                                  onClick={() => toggleSourceState('prioritizedSourceIds', source.source_id)}
                                  aria-pressed={prioritized}
                                  aria-label={`Prioritize ${sourceLabel}`}
                                >
                                  Prioritize
                                </button>
                              </div>
                            </div>
                          );
                        })}
                        <label className="flex items-center gap-2 text-sm font-medium">
                          <input
                            type="checkbox"
                            checked={checkpointEditor.draft.recollectEnabled}
                            onChange={(event) => updateSourcesDraft('recollectEnabled', event.target.checked)}
                          />
                          Recollect sources
                        </label>
                        <label className="flex items-center gap-2 text-sm font-medium">
                          <input
                            type="checkbox"
                            checked={checkpointEditor.draft.needPrimarySources}
                            onChange={(event) => updateSourcesDraft('needPrimarySources', event.target.checked)}
                          />
                          Need more primary sources
                        </label>
                        <label className="flex items-center gap-2 text-sm font-medium">
                          <input
                            type="checkbox"
                            checked={checkpointEditor.draft.needContradictions}
                            onChange={(event) => updateSourcesDraft('needContradictions', event.target.checked)}
                          />
                          Need more contradictions
                        </label>
                        <label className="block text-sm font-medium" htmlFor="sources-guidance">
                          Recollection guidance
                        </label>
                        <textarea
                          id="sources-guidance"
                          className="min-h-24 w-full rounded-xl border border-border bg-card px-3 py-2 text-sm outline-none transition focus:border-primary"
                          value={checkpointEditor.draft.guidance}
                          onChange={(event) => updateSourcesDraft('guidance', event.target.value)}
                        />
                      </div>
                    )}

                    {checkpointEditor?.checkpointType === 'outline_review' && (
                      <div className="space-y-3">
                        {checkpointEditor.draft.sections.map((section, index) => (
                          <div
                            key={`${section.focus_area}-${index}`}
                            className="rounded-xl border border-border bg-card px-3 py-3"
                          >
                            <label
                              className="block text-sm font-medium"
                              htmlFor={`outline-section-title-${index}`}
                            >
                              Section title {index + 1}
                            </label>
                            <input
                              id={`outline-section-title-${index}`}
                              className="mt-2 w-full rounded-xl border border-border bg-bg px-3 py-2 text-sm outline-none transition focus:border-primary"
                              value={section.title}
                              onChange={(event) => updateOutlineSectionTitle(index, event.target.value)}
                            />
                            <div className="mt-2 text-xs text-muted-foreground">
                              Focus area: {section.focus_area}
                            </div>
                            <div className="mt-3 flex flex-wrap gap-2">
                              <button
                                type="button"
                                className="rounded-full border border-border px-3 py-2 text-sm hover:bg-muted"
                                onClick={() => moveOutlineSection(index, -1)}
                                aria-label={`Move section up: ${section.focus_area}`}
                              >
                                Move up
                              </button>
                              <button
                                type="button"
                                className="rounded-full border border-border px-3 py-2 text-sm hover:bg-muted"
                                onClick={() => moveOutlineSection(index, 1)}
                                aria-label={`Move section down: ${section.focus_area}`}
                              >
                                Move down
                              </button>
                              <button
                                type="button"
                                className="rounded-full border border-danger/30 px-3 py-2 text-sm text-danger hover:bg-danger/10"
                                onClick={() => removeOutlineSection(index)}
                                aria-label={`Remove section: ${section.focus_area}`}
                              >
                                Remove
                              </button>
                            </div>
                          </div>
                        ))}
                        <div className="flex flex-wrap gap-2">
                          {checkpointEditor.availableFocusAreas
                            .filter(
                              (focusArea) =>
                                !checkpointEditor.draft.sections.some(
                                  (section) => section.focus_area === focusArea
                                )
                            )
                            .map((focusArea) => (
                              <button
                                key={focusArea}
                                type="button"
                                className="rounded-full border border-border px-3 py-2 text-sm hover:bg-muted"
                                onClick={() => addOutlineFocusArea(focusArea)}
                                aria-label={`Add focus area: ${focusArea}`}
                              >
                                Add {focusArea}
                              </button>
                            ))}
                        </div>
                      </div>
                    )}

                    {!checkpointEditor && (
                      <pre className="overflow-x-auto rounded-xl border border-border bg-card p-3 text-xs text-muted-foreground">
                        {JSON.stringify(selectedSnapshot.checkpoint.proposed_payload, null, 2)}
                      </pre>
                    )}
                  </div>
                )}
              </section>

              <section className="rounded-2xl border border-border bg-bg/70 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                      Research Trust
                    </h3>
                    {!trustView && !trustError && !canLoadTrust && (
                      <p className="mt-2 text-sm text-muted-foreground">
                        Trust signals will appear after synthesis
                      </p>
                    )}
                  </div>
                  {canLoadTrust && (
                    <button
                      type="button"
                      className="rounded-full border border-border px-3 py-2 text-sm hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
                      onClick={handleLoadTrustDetails}
                      disabled={isLoadingTrust}
                    >
                      {isLoadingTrust ? 'Loading trust details…' : 'Load trust details'}
                    </button>
                  )}
                </div>

                {trustError && (
                  <div className="mt-3 rounded-xl border border-danger/30 bg-danger/5 px-3 py-3 text-sm text-danger">
                    Unable to load trust details: {trustError}
                  </div>
                )}

                {trustView && (
                  <div className="mt-4 space-y-4">
                    <div className="rounded-xl border border-border bg-card px-3 py-3">
                      <div className="text-sm font-medium">Verification</div>
                      <div className="mt-3 grid gap-3 md:grid-cols-3">
                        <div className="text-sm text-muted-foreground">
                          Supported claims: {trustView.verificationSummary?.supported_claim_count ?? 0}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          Unsupported claims: {trustView.verificationSummary?.unsupported_claim_count ?? 0}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          Contradictions: {trustView.verificationSummary?.contradiction_count ?? 0}
                        </div>
                      </div>
                      {trustView.verificationSummary?.warnings?.length ? (
                        <div className="mt-3 space-y-1">
                          {trustView.verificationSummary.warnings.map((warning) => (
                            <div key={warning} className="text-sm text-danger">
                              {warning}
                            </div>
                          ))}
                        </div>
                      ) : null}
                    </div>

                    <div className="rounded-xl border border-border bg-card px-3 py-3">
                      <div className="text-sm font-medium">Unsupported Claims</div>
                      {trustView.unsupportedClaims.length ? (
                        <div className="mt-3 space-y-2">
                          {trustView.unsupportedClaims.map((claim, index) => (
                            <div
                              key={`${claim.claim_id ?? claim.text}-${index}`}
                              className="rounded-lg border border-border bg-bg px-3 py-2"
                            >
                              <div className="text-sm font-medium">{claim.text}</div>
                              <div className="mt-1 text-xs text-muted-foreground">
                                {claim.focus_area ?? 'unknown focus area'}
                                {claim.reason ? ` · ${claim.reason}` : ''}
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="mt-2 text-sm text-muted-foreground">No unsupported claims flagged.</p>
                      )}
                    </div>

                    <div className="rounded-xl border border-border bg-card px-3 py-3">
                      <div className="text-sm font-medium">Contradictions</div>
                      {trustView.contradictions.length ? (
                        <div className="mt-3 space-y-2">
                          {trustView.contradictions.map((contradiction, index) => (
                            <div
                              key={`${contradiction.note_id ?? contradiction.text}-${index}`}
                              className="rounded-lg border border-border bg-bg px-3 py-2"
                            >
                              <div className="text-sm font-medium">{contradiction.text}</div>
                              <div className="mt-1 text-xs text-muted-foreground">
                                {contradiction.focus_area ?? 'unknown focus area'}
                                {contradiction.source_id ? ` · ${contradiction.source_id}` : ''}
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="mt-2 text-sm text-muted-foreground">No contradictions surfaced.</p>
                      )}
                    </div>

                    <div className="rounded-xl border border-border bg-card px-3 py-3">
                      <div className="text-sm font-medium">Source Trust</div>
                      {trustView.sourceTrust.length ? (
                        <div className="mt-3 space-y-2">
                          {trustView.sourceTrust.map((source) => (
                            <div
                              key={source.source_id}
                              className="rounded-lg border border-border bg-bg px-3 py-2"
                            >
                              <div className="text-sm font-medium">{source.title ?? source.source_id}</div>
                              <div className="mt-1 text-xs text-muted-foreground">
                                {[source.provider, source.trust_tier, source.snapshot_policy]
                                  .filter((value): value is string => typeof value === 'string' && value.length > 0)
                                  .join(' · ')}
                              </div>
                              {source.trust_labels?.length ? (
                                <div className="mt-2 text-xs text-muted-foreground">
                                  {source.trust_labels.join(', ')}
                                </div>
                              ) : null}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="mt-2 text-sm text-muted-foreground">No source trust metadata available.</p>
                      )}
                    </div>
                  </div>
                )}
              </section>

              <section className="rounded-2xl border border-border bg-bg/70 p-4">
                <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                  Artifacts
                </h3>
                <div className="mt-3 space-y-3">
                  {selectedSnapshot?.artifacts.length ? (
                    selectedSnapshot.artifacts.map((artifact) => (
                      <div
                        key={artifact.artifact_name}
                        className="rounded-xl border border-border bg-card px-3 py-3"
                      >
                        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                          <div>
                            <div className="text-sm font-medium">{artifact.artifact_name}</div>
                            <div className="mt-1 text-xs text-muted-foreground">
                              v{artifact.artifact_version} · {artifact.phase}
                            </div>
                          </div>
                          <button
                            type="button"
                            className="rounded-full border border-border px-3 py-2 text-sm hover:bg-muted"
                            onClick={() => handleLoadArtifact(artifact.artifact_name)}
                          >
                            Load {artifact.artifact_name}
                          </button>
                        </div>
                        {state.artifactContents[artifact.artifact_name] !== undefined && (
                          <pre className="mt-3 overflow-x-auto rounded-xl border border-border bg-bg p-3 text-xs text-muted-foreground">
                            {formatArtifactContent(state.artifactContents[artifact.artifact_name])}
                          </pre>
                        )}
                      </div>
                    ))
                  ) : (
                    <div className="rounded-xl border border-dashed border-border px-3 py-4 text-sm text-muted-foreground">
                      No artifact metadata yet.
                    </div>
                  )}
                </div>
              </section>

              <section className="rounded-2xl border border-border bg-bg/70 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                      Bundle
                    </h3>
                    <p className="mt-2 text-sm text-muted-foreground">
                      Load the final package once the run is completed.
                    </p>
                  </div>
                  <button
                    type="button"
                    className="rounded-full border border-border px-3 py-2 text-sm hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
                    onClick={handleLoadBundle}
                    disabled={!canLoadBundle}
                  >
                    Load bundle
                  </button>
                </div>
                {state.bundle && (
                  <pre className="mt-4 overflow-x-auto rounded-xl border border-border bg-card p-3 text-xs text-muted-foreground">
                    {formatArtifactContent(state.bundle)}
                  </pre>
                )}
              </section>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
