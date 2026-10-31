import { beforeEach, describe, expect, it, vi } from 'vitest';
import { modelsAudioMethods } from '../models-audio';
import { presentationsMethods } from '../presentations';
import { workspaceApiMethods } from '../workspace-api';
import { generateResearchWorkspaceArtifact } from '@/services/researchWorkspaceArtifacts';
import { generateQuiz } from '@/services/quizzes';
import { isServicePromptRequestPath } from '../../service-prompt-scope-error';

const wire = vi.hoisted(() => vi.fn());
vi.mock('@/services/background-proxy', () => ({
  bgRequest: wire, sanitizeResponseData: (value: unknown) => value,
}));
const scope = { config: { serverUrl: 'https://original.test', authMode: 'multi-user' as const }, userId: 'alice' };
beforeEach(() => { wire.mockReset(); wire.mockResolvedValue({}); });

describe('Studio generation captured request authority', () => {
  it.each(['submit', 'poll', 'slides', 'quiz', 'verified-draft'] as const)('keeps %s on the captured scope and cancellation signal', async (operation) => {
    const controller = new AbortController();
    const options = { requestScope: scope, signal: controller.signal };
    if (operation === 'submit') await workspaceApiMethods.submitWorkspaceOutput('owned', {
      artifact_type: 'infographic', source_ids: ['source-1'], settings: {},
    }, options);
    if (operation === 'poll') await workspaceApiMethods.getWorkspaceOutputStatus('owned', 7, options);
    if (operation === 'slides') await presentationsMethods.generateSlidesFromMedia.call(
      { request: vi.fn(() => { throw new Error('Unscoped transport'); }) }, 101, options);
    if (operation === 'quiz') await generateQuiz({ media_id: 101 }, options);
    if (operation === 'verified-draft') await generateResearchWorkspaceArtifact({
      artifact_type: 'mindmap', media_ids: [101], model: 'local-model',
    }, options);
    expect(wire).toHaveBeenCalledTimes(1);
    const request = wire.mock.calls[0][0];
    expect(request).toMatchObject({
      abortSignal: controller.signal,
      servicePromptConfig: { ...scope.config, expectedUserId: 'alice' },
      headers: { 'X-TLDW-Expected-User-ID': 'alice' },
    });
    expect(isServicePromptRequestPath(request.path, request.method)).toBe(true);
  });

  it('scopes the TTS capability read and speech POST, retaining Accept and audio decoding', async () => {
    const controller = new AbortController();
    wire.mockResolvedValueOnce({ supports_explicit_backend: true })
      .mockResolvedValueOnce({
        ok: true, data: Uint8Array.from([1, 2, 3]).buffer,
        headers: { 'X-TLDW-TTS-Backend': 'pocket' }
      });
    const unscoped = vi.fn(() => { throw new Error('Unscoped transport'); });
    const result = await modelsAudioMethods.synthesizeSpeechDetailed.call({
      ensureConfigForRequest: unscoped, requestWithCurrentConfig: unscoped,
    }, 'Rowan Observatory', {
      requestScope: scope, signal: controller.signal,
      backend: 'pocket', responseFormat: 'wav', allowFallback: false
    });
    expect(unscoped).not.toHaveBeenCalled();
    expect(wire.mock.calls.map(([request]) => request.path)).toEqual([
      '/api/v1/audio/providers', '/api/v1/audio/speech',
    ]);
    expect(wire.mock.calls.every(([request]) => request.servicePromptConfig.expectedUserId === 'alice' &&
      request.abortSignal === controller.signal)).toBe(true);
    expect(wire.mock.calls[1][0]).toMatchObject({
      headers: { Accept: 'audio/wav', 'X-TLDW-Expected-User-ID': 'alice' },
      body: { input: 'Rowan Observatory', backend: 'pocket', allow_fallback: false },
    });
    expect(new Uint8Array(result.buffer)).toEqual(Uint8Array.from([1, 2, 3]));
    expect(result.actualBackend).toBe('pocket');
  });

  it('retires a held TTS capability lookup before any speech request', async () => {
    let finish!: (value: unknown) => void;
    wire.mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
    const controller = new AbortController();
    const request = modelsAudioMethods.synthesizeSpeechDetailed.call({}, 'Private source text', {
      requestScope: scope, signal: controller.signal, backend: 'pocket',
    });
    controller.abort();
    finish({ supports_explicit_backend: true });
    await expect(request).rejects.toBeDefined();
    expect(wire).toHaveBeenCalledTimes(1);
    expect(wire.mock.calls[0][0].path).toBe('/api/v1/audio/providers');
  });

  it.each([
    ['/api/v1/workspaces/owned/outputs/7', 'POST'],
    ['/api/v1/workspaces/owned/outputs/7/cancel', 'POST'],
    ['/api/v1/workspaces/owned%2Fother/outputs', 'POST'],
    ['/api/v1/slides/generate/from-media/other', 'POST'],
    ['/api/v1/research-workspace/artifacts/generate', 'GET'],
  ])('refuses unrelated scoped generation route %s %s', (path, method) => {
    expect(isServicePromptRequestPath(path, method)).toBe(false);
  });
});
