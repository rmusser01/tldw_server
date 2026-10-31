import {
  createAudioStudioGeneration,
  createAudioStudioRender,
  createAudioStudioExport,
  listAudioStudioJobs,
  type AudioStudioAdmissionKind,
  type AudioStudioAdmissionBody,
  type AudioStudioAdmissionIntent,
  type AudioStudioGenerationCreateRequest,
  type AudioStudioRenderCreateRequest,
  type AudioStudioExportCreateRequest,
  type AudioStudioRequestOptions,
  type AudioStudioJobResponse,
} from '@/services/audio-studio';
import { useAudioStudioStore } from '@/store/audio-studio';

const inFlight = new Set<string>();
const storageKey = (ownerKey: string) => `tldw:audio-studio:admissions:${ownerKey}`;
export const isTerminalAudioStudioJob = (status: string) =>
  ['completed', 'failed', 'cancelled', 'quarantined', 'skipped'].includes(status);

export const restoreAudioStudioIntents = (ownerKey: string) => {
  const text = sessionStorage.getItem(storageKey(ownerKey));
  const intents: Record<string, AudioStudioAdmissionIntent> = text ? JSON.parse(text) : {};
  if (
    !intents ||
    typeof intents !== 'object' ||
    Array.isArray(intents) ||
    Object.values(intents).some(
      (intent) =>
        !intent ||
        !['generation', 'render', 'export'].includes(intent.kind) ||
        !intent.body ||
        typeof intent.body.idempotency_key !== 'string' ||
        typeof intent.body.target_resource_id !== 'string' ||
        typeof intent.body.target_revision_id !== 'string' ||
        (intent.job_id !== undefined && typeof intent.job_id !== 'string'),
    )
  )
    throw new Error('Pending Audio Studio job identity could not be restored.');
  useAudioStudioStore.setState({ jobIntents: intents });
};

export const saveAudioStudioIntent = (
  ownerKey: string,
  projectId: string,
  intent?: AudioStudioAdmissionIntent,
) => {
  const intents = { ...useAudioStudioStore.getState().jobIntents };
  if (intent) intents[projectId] = intent;
  else delete intents[projectId];
  // Persist before dispatch so a reload or lost response cannot silently create another job.
  sessionStorage.setItem(storageKey(ownerKey), JSON.stringify(intents));
  useAudioStudioStore.setState({ jobIntents: intents });
};

const sameInput = (
  kind: AudioStudioAdmissionKind,
  a: AudioStudioAdmissionBody,
  b: AudioStudioAdmissionBody,
) => {
  const comparable = (body: AudioStudioAdmissionBody) => ({
    ...body,
    idempotency_key: undefined,
    ...(kind === 'generation' ? {} : { target_resource_id: undefined }),
  });
  return JSON.stringify(comparable(a)) === JSON.stringify(comparable(b));
};

export const admitAudioStudioJob = async (
  ownerKey: string,
  projectId: string,
  kind: AudioStudioAdmissionKind,
  body: AudioStudioAdmissionBody,
  options?: AudioStudioRequestOptions,
  retryStored = false,
): Promise<AudioStudioJobResponse> => {
  options?.assertCurrent?.();
  options?.assertCanWrite?.();
  const lock = `${ownerKey}:${projectId}`;
  if (inFlight.has(lock)) throw new Error('A project job admission is already in progress.');
  inFlight.add(lock);
  let dispatched = false;
  try {
    const jobs = await listAudioStudioJobs(projectId, options);
    options?.assertCurrent?.();
    let retained = useAudioStudioStore.getState().jobIntents[projectId];
    if (retained?.job_id) {
      const known = jobs.find((job) => job.job_id === retained.job_id);
      if (known && isTerminalAudioStudioJob(known.status)) {
        saveAudioStudioIntent(ownerKey, projectId);
        if (retryStored) return known;
        retained = undefined;
      } else {
        if (retryStored && known) return known;
        throw new Error('A project job is pending. Check its status before generating again.');
      }
    }
    if (retained && (retained.kind !== kind || !sameInput(kind, retained.body, body))) {
      throw new Error(
        'A previous admission is uncertain. Retry the pending admission before changing the request.',
      );
    }
    if (!retained && jobs.some((job) => !isTerminalAudioStudioJob(job.status))) {
      throw new Error('A project job is already queued or processing.');
    }
    const intent = retained ?? { kind, body };
    saveAudioStudioIntent(ownerKey, projectId, intent);
    dispatched = true;
    let result: AudioStudioJobResponse;
    if (intent.kind === 'generation')
      result = await createAudioStudioGeneration(
        projectId,
        intent.body as AudioStudioGenerationCreateRequest,
        options,
      );
    else if (intent.kind === 'render')
      result = await createAudioStudioRender(
        projectId,
        intent.body as AudioStudioRenderCreateRequest,
        options,
      );
    else
      result = await createAudioStudioExport(
        projectId,
        intent.body as AudioStudioExportCreateRequest,
        options,
      );
    options?.assertCurrent?.();
    saveAudioStudioIntent(ownerKey, projectId, { ...intent, job_id: result.job_id });
    return result;
  } catch (error) {
    const status = (error as { status?: number })?.status;
    if (dispatched && [400, 401, 403, 404, 409, 422].includes(status ?? 0)) {
      options?.assertCurrent?.();
      saveAudioStudioIntent(ownerKey, projectId);
    }
    throw error;
  } finally {
    inFlight.delete(lock);
  }
};
