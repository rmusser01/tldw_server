import type { WorkspaceArtifactApiResponse } from "@/services/tldw/domains/workspace-api";
import type { MessageInstance } from 'antd/es/message/interface';
import type { TFunction } from 'i18next';
import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { GeneratedArtifact, WorkspaceSource } from '@/types/workspace';
import { rememberCanonicalArtifactWorkspace } from '@/services/workspace-artifact-persistence';
import { bindWorkspaceArtifactOrigin } from '@/store/workspace-artifact-codec';
import { mapServerArtifactToLocal } from '@/store/workspace-api';
import { useArtifactGeneration, type UseArtifactGenerationDeps } from '../useArtifactGeneration';

const io = vi.hoisted(() => ({
  migrated: true,
  state: { workspaceId: 'owned', generatedArtifacts: [] as GeneratedArtifact[] },
  request: vi.fn(), snapshot: vi.fn(), verified: vi.fn(), speech: vi.fn(), complete: vi.fn(), media: vi.fn(), sources: vi.fn(), submit: vi.fn(), output: vi.fn(),
}));
vi.mock('@/store/workspace', () => ({
  useWorkspaceStore: { getState: () => io.state },
  hasResearchWorkspaceMigrationTombstone: () => io.migrated,
}));
vi.mock('@/services/background-proxy', () => ({ bgRequest: io.request }));
vi.mock('@/services/service-prompts', () => ({ loadServicePromptSnapshot: io.snapshot }));
vi.mock('@/services/researchWorkspaceArtifacts', () => ({ generateResearchWorkspaceArtifact: io.verified }));
vi.mock('@/services/tldw/TldwApiClient', () => ({
  tldwClient: {
    listVisualStyles: async () => [],
    synthesizeSpeechDetailed: io.speech,
    getMediaDetails: io.media,
    getWorkspaceSources: io.sources,
    createChatCompletion: io.complete,
    submitWorkspaceOutput: io.submit, getWorkspaceOutputStatus: io.output,
  }
}));
vi.mock('@/services/tldw', () => ({
  tldwModels: {
    getChatModels: async () => [{ id: 'local-model', provider: 'llamacpp' }],
    getModel: async () => null, getProviderDisplayName: (value: string) => value,
  }
}));
vi.mock('@/services/flashcards', () => ({
  listDecks: async () => [], createDeck: vi.fn(), createFlashcard: vi.fn(),
  createFlashcardsBulk: vi.fn(), generateFlashcards: vi.fn(),
}));
vi.mock('@/utils/research-workspace-telemetry', () => ({ trackResearchWorkspaceTelemetry: vi.fn() }));

let stored: WorkspaceArtifactApiResponse[];
let controller: AbortController;
const setGenerating = vi.fn();
const messages = { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() };
const source: WorkspaceSource = {
  id: 'source-1', mediaId: 101, title: 'Cached title', type: 'text', status: 'ready', addedAt: new Date(),
};
const update = vi.fn((id: string, status: GeneratedArtifact['status'], patch?: Partial<GeneratedArtifact>) => {
  io.state.generatedArtifacts = io.state.generatedArtifacts.map((artifact) =>
    artifact.id === id ? { ...artifact, ...patch, status } : artifact);
});
const deps = (): UseArtifactGenerationDeps => ({
  workspaceId: io.state.workspaceId, workspaceName: 'Rowan', workspaceTag: 'workspace:owned',
  selectedMediaIds: [101], selectedSources: [source], selectedMediaCount: 1, hasSelectedSources: true,
  audioSettings: { provider: 'browser', model: 'kokoro', voice: 'af_heart', speed: 1, format: 'mp3' },
  outputButtons: [{ type: 'summary', label: 'Summary' }],
  generatedArtifacts: io.state.generatedArtifacts, isGeneratingOutput: false, generatingOutputType: null,
  addArtifact: (input) => {
    const artifact = { ...input, id: 'stable-draft', createdAt: new Date() } as GeneratedArtifact;
    io.state.generatedArtifacts.push(artifact);
    return artifact;
  },
  updateArtifactStatus: update, setIsGeneratingOutput: setGenerating,
  messageApi: messages as unknown as MessageInstance,
  selectedModel: 'local-model', normalizedApiProvider: 'llamacpp',
  resolvedTemperature: 0, resolvedTopP: 1, resolvedNumPredict: 512,
  resolvedSummaryInstruction: 'Summarize the source.',
  slidesVisualStyleValue: '', selectedFlashcardDeck: 'auto', ragAdvancedOptions: {},
  t: ((_key: string, fallback: unknown) => fallback) as unknown as TFunction,
});

beforeEach(() => {
  vi.clearAllMocks(); io.request.mockReset(); io.complete.mockReset(); localStorage.clear();
  stored = []; controller = new AbortController();
  io.speech.mockReset(); io.verified.mockReset();
  io.speech.mockResolvedValue({ buffer: new ArrayBuffer(8), generatedFileId: 47, actualBackend: 'pocket_tts', fallbackUsed: false });
  io.verified.mockResolvedValue({ content: 'Selene tours start Friday at 18:00.', data: {} });
  io.state = { workspaceId: 'owned', generatedArtifacts: [] };
  io.migrated = true;
  io.snapshot.mockImplementation(async () => ({
    scopeKey: 'server:alice',
    requestScope: { config: { serverUrl: 'https://owned.test' }, userId: 'alice' },
    scopeSignal: controller.signal, scopeInvalidatedSignal: controller.signal, release: vi.fn(),
  }));
  io.sources.mockResolvedValue([{ id: 'source-1', workspace_id: 'owned', media_id: 101, title: 'Rowan Observatory' }]);
  io.media.mockResolvedValue({ content: 'Selene is at Rowan Observatory. Tours start Friday at 18:00.' });
  io.complete.mockResolvedValue({ text: async () => JSON.stringify({ choices: [{ message: { content: 'Selene tours start Friday at 18:00.' } }] }) });
  io.request.mockImplementation(async (request) => {
    if (request.method === 'GET') return stored;
    if (request.method === 'PUT') {
      const id = decodeURIComponent(request.path.split('/').at(-1));
      const index = stored.findIndex(row => row.id === id);
      if (index < 0) throw new Error('Missing canonical artifact');
      stored[index] = { ...stored[index], ...request.body, version: stored[index].version + 1 };
      return stored[index];
    }
    if (request.method !== 'POST') throw new Error('Unexpected artifact write');
    const row = { ...request.body, workspace_id: 'owned', version: 1, created_at: new Date().toISOString() };
    stored.push(row);
    return row;
  });
});

describe('migrated Studio result durability', () => {

  it("persists a stored-audio reference that survives cold restoration", async () => {
    const original = Object.getOwnPropertyDescriptor(URL, "createObjectURL")
    const transient = vi.fn(() => "blob:transient-audio")
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: transient })
    try {
      const input = { ...deps(), audioSettings: { provider: "tldw" as const, model: "pocket", voice: "owned-voice", speed: 1, format: "wav" as const } }
      const hook = renderHook(() => useArtifactGeneration(input))
      await act(async () => { await hook.result.current.handleGenerateOutput("audio_overview") })
      expect(stored).toHaveLength(1)
      expect(io.speech.mock.calls[0][1]).toMatchObject({ returnDownloadLink: true, requestScope: { userId: "alice" } })
      const restored = mapServerArtifactToLocal(stored[0])
      expect(restored).toMatchObject({
        id: "stable-draft", workspaceArtifactId: "stable-draft", audioFormat: "wav",
        content: "Selene tours start Friday at 18:00.",
        exportRefs: [{ sourceType: "storage_file", fileId: 47, format: "wav", status: "ready" }],
      })
      expect(restored.audioUrl).toBeUndefined()
      expect(transient).not.toHaveBeenCalled()
    } finally {
      if (original) Object.defineProperty(URL, "createObjectURL", original)
      else Reflect.deleteProperty(URL, "createObjectURL")
    }
  })

  it("refuses canonical audio without a stored-file acknowledgement", async () => {
    io.speech.mockResolvedValueOnce({ buffer: new ArrayBuffer(8), fallbackUsed: false })
    const original = Object.getOwnPropertyDescriptor(URL, "createObjectURL")
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: vi.fn(() => "blob:transient-audio") })
    try {
      const hook = renderHook(() => useArtifactGeneration({ ...deps(), audioSettings: { provider: "tldw", model: "pocket", voice: "owned-voice", speed: 1, format: "wav" } }))
      await act(async () => { await hook.result.current.handleGenerateOutput("audio_overview") })
      expect(stored).toHaveLength(0)
      expect(messages.success).not.toHaveBeenCalled()
      expect(io.state.generatedArtifacts[0].status).toBe("failed")
    } finally {
      if (original) Object.defineProperty(URL, "createObjectURL", original)
      else Reflect.deleteProperty(URL, "createObjectURL")
    }
  })

  it("retains the stored-audio file for save-only Retry without another synthesis", async () => {
    const original = Object.getOwnPropertyDescriptor(URL, "createObjectURL")
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: vi.fn(() => "blob:transient-audio") })
    io.request.mockImplementationOnce(async () => []).mockRejectedValueOnce(new Error("Storage unavailable"))
      .mockImplementationOnce(async () => [])
    try {
      const hook = renderHook(() => useArtifactGeneration({ ...deps(), audioSettings: { provider: "tldw", model: "pocket", voice: "owned-voice", speed: 1, format: "wav" } }))
      await act(async () => { await hook.result.current.handleGenerateOutput("audio_overview") })
      expect(hook.result.current.unsavedArtifact?.exportRefs).toEqual([{ sourceType: "storage_file", fileId: 47, format: "wav", status: "ready" }])
      await act(async () => { await hook.result.current.handleRetryArtifactSave() })
      expect(io.speech).toHaveBeenCalledTimes(1)
      expect(stored).toHaveLength(1)
      expect(mapServerArtifactToLocal(stored[0]).exportRefs?.[0].fileId).toBe(47)
    } finally {
      if (original) Object.defineProperty(URL, "createObjectURL", original)
      else Reflect.deleteProperty(URL, "createObjectURL")
    }
  })

  it("does not save canonical audio after a storage-quota refusal", async () => {
    io.speech.mockRejectedValueOnce(Object.assign(new Error("Storage quota exceeded"), { status: 507 }))
    const hook = renderHook(() => useArtifactGeneration({ ...deps(), audioSettings: { provider: "tldw", model: "pocket", voice: "owned-voice", speed: 1, format: "wav" } }))
    await act(async () => { await hook.result.current.handleGenerateOutput("audio_overview") })
    expect(stored).toHaveLength(0)
    expect(messages.success).not.toHaveBeenCalled()
    expect(io.state.generatedArtifacts[0].status).toBe("failed")
  })


  it('saves a real generated result before success and cold-restores its stable identity and source lineage', async () => {
    const hook = renderHook(() => useArtifactGeneration(deps()));
    await act(async () => { await hook.result.current.handleGenerateOutput('summary'); });
    expect(stored).toHaveLength(1);
    io.state.generatedArtifacts = [];
    const restored = mapServerArtifactToLocal(stored[0]);
    expect(restored).toMatchObject({
      id: 'stable-draft', workspaceArtifactId: 'stable-draft', workspaceArtifactVersion: 1,
      content: 'Selene tours start Friday at 18:00.',
      sourceLineage: [{ sourceId: 'source-1', mediaId: 101, title: 'Rowan Observatory' }],
    });
    expect(messages.success).toHaveBeenCalledTimes(1);
    expect(io.complete.mock.calls[0][1]).toMatchObject({ requestScope: { userId: 'alice' } });
  });

  it('retains generated content after a failed save and retries only saving the same draft', async () => {
    io.request.mockImplementationOnce(async () => []).mockRejectedValueOnce(new Error('Storage unavailable'))
      .mockImplementationOnce(async () => []);
    const hook = renderHook(() => useArtifactGeneration(deps()));
    await act(async () => { await hook.result.current.handleGenerateOutput('summary'); });
    expect(io.state.generatedArtifacts[0]).toMatchObject({
      id: 'stable-draft', status: 'completed', content: 'Selene tours start Friday at 18:00.',
    });
    expect(messages.success).not.toHaveBeenCalled();
    expect(hook.result.current.unsavedArtifact).toMatchObject({ id: 'stable-draft' });
    await act(async () => { await hook.result.current.handleRetryArtifactSave(); });
    expect(io.complete).toHaveBeenCalledTimes(1);
    expect(stored).toHaveLength(1);
    expect(hook.result.current.unsavedArtifact).toBeNull();
  });

  it('retires a held generation before it can publish or save into a different workspace', async () => {
    let finish!: (value: unknown) => void;
    io.complete.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
    const hook = renderHook(() => useArtifactGeneration(deps()));
    let generation!: Promise<void>;
    act(() => { generation = hook.result.current.handleGenerateOutput('summary'); });
    await waitFor(() => expect(io.complete).toHaveBeenCalledTimes(1));
    io.state = { workspaceId: 'other', generatedArtifacts: [] };
    hook.rerender();
    await act(async () => {
      finish({ text: async () => JSON.stringify({ choices: [{ message: { content: 'Retired private result' } }] }) });
      await generation;
    });
    expect(io.request.mock.calls.filter(([request]) => request.method !== 'GET')).toHaveLength(0);
    expect(io.state.generatedArtifacts).toEqual([]);
    expect(messages.success).not.toHaveBeenCalled();
  });

  it('keeps legacy local generation available without requiring a canonical account', async () => {
    io.migrated = false;
    const hook = renderHook(() => useArtifactGeneration(deps()));
    await act(async () => { await hook.result.current.handleGenerateOutput('summary'); });
    expect(io.state.generatedArtifacts[0]).toMatchObject({ status: 'completed', content: 'Selene tours start Friday at 18:00.' });
    expect(io.snapshot).not.toHaveBeenCalled();
    expect(io.request).not.toHaveBeenCalled();
  });
  it('retires a held result when the captured authenticated account changes', async () => {
    let finish!: (value: unknown) => void;
    io.complete.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
    const hook = renderHook(() => useArtifactGeneration(deps()));
    let generation!: Promise<void>;
    act(() => { generation = hook.result.current.handleGenerateOutput('summary'); });
    await waitFor(() => expect(io.complete).toHaveBeenCalledTimes(1));
    await act(async () => {
      controller.abort();
      io.state.generatedArtifacts = [];
      finish({ text: async () => JSON.stringify({ choices: [{ message: { content: 'Alice private result' } }] }) });
      await generation;
    });
    expect(stored).toEqual([]);
    expect(io.state.generatedArtifacts).toEqual([]);
    expect(messages.success).not.toHaveBeenCalled();
  });

  it('reconciles an unknown POST acknowledgement without duplicate writes or another generation', async () => {
    io.request.mockImplementationOnce(async () => []).mockImplementationOnce(async (request) => {
      stored.push({ ...request.body, workspace_id: 'owned', version: 1, created_at: new Date().toISOString() });
      throw new Error('Acknowledgement lost');
    });
    const hook = renderHook(() => useArtifactGeneration(deps()));
    await act(async () => { await hook.result.current.handleGenerateOutput('summary'); });
    expect(stored).toHaveLength(1);
    expect(io.request.mock.calls.filter(([request]) => request.method === 'POST')).toHaveLength(1);
    expect(io.complete).toHaveBeenCalledTimes(1);
    expect(hook.result.current.unsavedArtifact).toBeNull();
    expect(messages.success).toHaveBeenCalledTimes(1);
  });

  it('does not bless a cached source whose canonical ID belongs to different Media', async () => {
    io.sources.mockResolvedValueOnce([{ id: 'source-1', workspace_id: 'owned', media_id: 202, title: 'Other source' }]);
    const hook = renderHook(() => useArtifactGeneration(deps()));
    await act(async () => { await hook.result.current.handleGenerateOutput('summary'); });
    expect(io.media).not.toHaveBeenCalled();
    expect(io.complete).not.toHaveBeenCalled();
    expect(stored).toEqual([]);
    expect(messages.success).not.toHaveBeenCalled();
  });

  it('enrolls a worker-created canonical output without posting another artifact or confusing its export ID', async () => {
    io.submit.mockResolvedValueOnce({ job_id: 7 });
    const canonical = {
      id: 'canonical-output', workspace_id: 'owned', version: 1, artifact_type: 'infographic',
      title: 'Infographic', status: 'completed', content: 'Rowan evidence infographic',
      created_at: new Date().toISOString(),
      completed_at: null, total_tokens: null, total_cost_usd: null,
      export_refs: [{ id: 'export-99', format: 'png', status: 'ready' }],
      source_lineage: {
        media_ids: [101], selected_source_ids: ['source-1'],
        sources: [{ sourceId: 'source-1', mediaId: 101, title: 'Rowan Observatory' }]
      },
    };
    stored.push(canonical);
    io.output.mockResolvedValueOnce({ status: 'completed', artifact: canonical });
    const hook = renderHook(() => useArtifactGeneration(deps()));
    await act(async () => { await hook.result.current.handleGenerateOutput('infographic'); });
    expect(io.request.mock.calls.filter(([request]) => request.method === 'POST')).toHaveLength(0);
    expect(stored).toHaveLength(1);
    expect(io.state.generatedArtifacts[0]).toMatchObject({
      id: 'stable-draft', workspaceArtifactId: 'canonical-output', serverId: 'export-99',
    });
    expect(io.submit.mock.calls[0][2]).toMatchObject({ requestScope: { userId: 'alice' } });
    expect(io.output.mock.calls[0][2]).toMatchObject({ requestScope: { userId: 'alice' } });
  });

  it('retires a pending save-only retry after account change', async () => {
    io.request.mockImplementationOnce(async () => []).mockRejectedValueOnce(new Error('Storage unavailable'))
      .mockImplementationOnce(async () => []);
    const hook = renderHook(() => useArtifactGeneration(deps()));
    await act(async () => { await hook.result.current.handleGenerateOutput('summary'); });
    expect(hook.result.current.unsavedArtifact).not.toBeNull();
    await act(async () => { controller.abort(); });
    const count = io.request.mock.calls.length;
    await act(async () => { await hook.result.current.handleRetryArtifactSave(); });
    expect(io.request.mock.calls).toHaveLength(count);
    expect(hook.result.current.unsavedArtifact).toBeNull();
    expect(stored).toEqual([]);
  });

  it('retires legacy local generation on a same-workspace authenticated-account event without requiring canonical access', async () => {
    io.migrated = false;
    let finish!: (value: unknown) => void;
    io.complete.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
    const hook = renderHook(() => useArtifactGeneration(deps()));
    let generation!: Promise<void>;
    act(() => { generation = hook.result.current.handleGenerateOutput('summary'); });
    await waitFor(() => expect(io.complete).toHaveBeenCalledTimes(1));
    await act(async () => {
      window.dispatchEvent(new Event('tldw:auth-principal-changed'));
      io.state.generatedArtifacts = [];
      finish({ text: async () => JSON.stringify({ choices: [{ message: { content: 'Retired legacy result' } }] }) });
      await generation;
    });
    expect(io.snapshot).not.toHaveBeenCalled();
    expect(io.state.generatedArtifacts).toEqual([]);
    expect(messages.success).not.toHaveBeenCalled();
  });

  it('refuses replacement of a serialized foreign cached card before reading Media or generating', async () => {
    io.state.generatedArtifacts = [{
      id: 'foreign-card', type: 'summary', title: 'Private other-account title',
      ownerId: 'bob', projectId: 'foreign-project', status: 'completed',
      content: 'Bob private content', createdAt: new Date(),
    }];
    const hook = renderHook(() => useArtifactGeneration(deps()));
    await act(async () => {
      await hook.result.current.handleGenerateOutput('summary',
        { mode: 'replace', targetArtifactId: 'foreign-card' });
    });
    expect(io.media).not.toHaveBeenCalled();
    expect(io.complete).not.toHaveBeenCalled();
    expect(stored).toEqual([]);
    expect(io.state.generatedArtifacts[0].content).toBe('Bob private content');
  });

  it.each(['edit', 'delete'])('does not overwrite a newer local %s while a canonical acknowledgement is held', async (change) => {
    let acknowledge!: (value: unknown) => void;
    io.request.mockImplementationOnce(async () => []).mockImplementationOnce((request) =>
      new Promise((resolve) => {
        acknowledge = () => resolve({
          ...request.body, workspace_id: 'owned', version: 1, created_at: new Date().toISOString(),
        });
      }));
    const hook = renderHook(() => useArtifactGeneration(deps()));
    let generation!: Promise<void>;
    act(() => { generation = hook.result.current.handleGenerateOutput('summary'); });
    await waitFor(() => expect(acknowledge).toBeTypeOf('function'));
    const edited = { ...io.state.generatedArtifacts[0], content: 'Newer local edit' };
    io.state.generatedArtifacts = change === 'edit' ? [edited] : [];
    await act(async () => { acknowledge(undefined); await generation; });
    expect(io.state.generatedArtifacts).toEqual(change === 'edit' ? [edited] : []);
    expect(messages.success).not.toHaveBeenCalled();
  });

  it('keeps the completed result in its bounded save stage when generation Cancel is pressed', async () => {
    let acknowledge!: (value: unknown) => void;
    io.request.mockImplementationOnce(async () => []).mockImplementationOnce((request) =>
      new Promise((resolve) => {
        acknowledge = () => resolve({
          ...request.body, workspace_id: 'owned', version: 1, created_at: new Date().toISOString(),
        });
      }));
    const hook = renderHook(() => useArtifactGeneration(deps()));
    let generation!: Promise<void>;
    act(() => { generation = hook.result.current.handleGenerateOutput('summary'); });
    await waitFor(() => expect(acknowledge).toBeTypeOf('function'));
    act(() => { hook.result.current.handleCancelGeneration(); });
    expect(io.request.mock.calls[1][0].abortSignal.aborted).toBe(false);
    await act(async () => { acknowledge(undefined); await generation; });
    expect(io.state.generatedArtifacts[0]).toMatchObject({ status: 'completed', workspaceArtifactVersion: 1 });
    expect(io.complete).toHaveBeenCalledTimes(1);
    expect(messages.success).toHaveBeenCalledTimes(1);
  });

  it.each(['edit', 'delete'])('retires a held save-only Retry after a newer %s without another write', async (change) => {
    io.request.mockImplementationOnce(async () => []).mockRejectedValueOnce(new Error('Storage unavailable'))
      .mockImplementationOnce(async () => []);
    const hook = renderHook(() => useArtifactGeneration(deps()));
    await act(async () => { await hook.result.current.handleGenerateOutput('summary'); });
    let release!: (value: unknown) => void;
    io.request.mockImplementationOnce(() => new Promise(resolve => { release = resolve; }));
    let retry!: Promise<void>;
    act(() => { retry = hook.result.current.handleRetryArtifactSave(); });
    await waitFor(() => expect(release).toBeTypeOf('function'));
    const edited = { ...io.state.generatedArtifacts[0], content: 'Keep latest edit during Retry' };
    io.state.generatedArtifacts = change === 'edit' ? [edited] : [];
    await act(async () => { release([]); await retry; });
    expect(io.request.mock.calls.filter(([request]) => request.method === 'POST')).toHaveLength(1);
    expect(io.state.generatedArtifacts).toEqual(change === 'edit' ? [edited] : []);
    expect(hook.result.current.unsavedArtifact).toBeNull();
    expect(messages.success).not.toHaveBeenCalled();
  });

  it('saves new outputs in a freshly restored canonical workspace even without a migration tombstone', async () => {
    io.migrated = false;
    rememberCanonicalArtifactWorkspace('owned', 'server:alice');
    const hook = renderHook(() => useArtifactGeneration(deps()));
    await act(async () => { await hook.result.current.handleGenerateOutput('summary'); });
    expect(stored).toHaveLength(1);
    expect(mapServerArtifactToLocal(stored[0])).toMatchObject({
      id: 'stable-draft', workspaceArtifactId: 'stable-draft', content: 'Selene tours start Friday at 18:00.',
    });
    expect(messages.success).toHaveBeenCalledTimes(1);
  });

  it('keeps distinct canonical and display identities when backend media creates a new version', async () => {
    const previous = {
      id: 'canonical-prior', workspace_id: 'owned', version: 1, artifact_type: 'infographic',
      title: 'Prior infographic', status: 'completed', content: 'Previous output',
      completed_at: null, total_tokens: null, total_cost_usd: null,
      created_at: new Date().toISOString(), version_metadata: {
        workspaceClient: { schemaVersion: 1, id: 'prior-display' },
      },
    };
    stored.push(previous);
    io.state.generatedArtifacts = [bindWorkspaceArtifactOrigin(mapServerArtifactToLocal(previous),
      'server:alice', 'owned')];
    const canonical = {
      ...previous, id: 'canonical-new', content: 'New infographic',
      version_metadata: {}, export_refs: [{ id: 'export-new', format: 'png', status: 'ready' }]
    };
    io.submit.mockImplementationOnce(async () => { stored.push(canonical); return { job_id: 8 }; });
    io.output.mockResolvedValueOnce({ status: 'completed', artifact: canonical });
    const hook = renderHook(() => useArtifactGeneration(deps()));
    await act(async () => {
      await hook.result.current.handleGenerateOutput('infographic',
        { mode: 'replace', targetArtifactId: 'prior-display' });
    });
    const restored = stored.map(mapServerArtifactToLocal);
    expect(restored.map(artifact => artifact.id).sort()).toEqual(['prior-display', 'stable-draft']);
    expect(new Set(restored.map(artifact => artifact.workspaceArtifactId))).toHaveProperty('size', 2);
    expect(restored.find(artifact => artifact.id === 'stable-draft')).toMatchObject({
      workspaceArtifactId: 'canonical-new', previousVersionId: 'prior-display', serverId: 'export-new',
    });
  });

  it.each([
    ['corpus_gap_finder', 'data_table', false],
    ['evidence_bound_hypotheses', 'report', false],
    ['research_proposal_pack', 'report', false],
    ['evidence_bound_hypotheses', 'report', true],
  ] as const)('uses only witnessed prior work for %s (owned=%s)', async (templateId, type, owned) => {
    const previous = {
      id: 'cached-matrix', type: 'data_table', templateId: 'literature_matrix', status: 'completed',
      title: 'Prior matrix', content: 'PRIOR_PRIVATE_MATRIX_SENTINEL', createdAt: new Date(),
      sourceLineage: [{ sourceId: 'source-1', mediaId: 101 }, { sourceId: 'source-2', mediaId: 102 }],
    } as GeneratedArtifact;
    io.state.generatedArtifacts = [owned ? bindWorkspaceArtifactOrigin(previous, 'server:alice', 'owned') : previous];
    const selectedSources = [source, { ...source, id: 'source-2', mediaId: 102 }];
    io.sources.mockResolvedValueOnce(selectedSources.map(item => ({
      id: item.id, workspace_id: 'owned', media_id: item.mediaId, title: 'Owned source',
    })));
    const hook = renderHook(() => useArtifactGeneration({
      ...deps(), selectedSources, selectedMediaIds: [101, 102], selectedMediaCount: 2,
    }));
    await act(async () => { await hook.result.current.handleGenerateOutput(type, { templateId }); });
    expect(io.complete).toHaveBeenCalledTimes(1);
    const sent = JSON.stringify(io.complete.mock.calls[0][0].messages);
    expect(sent.includes('PRIOR_PRIVATE_MATRIX_SENTINEL')).toBe(owned);
    expect(sent).toContain('Owned source');
  });

  it('retires an obsolete save warning when an explicit edit saves a newer canonical revision', async () => {
    io.request.mockImplementationOnce(async () => []).mockRejectedValueOnce(new Error('Storage unavailable'))
      .mockImplementationOnce(async () => []);
    const hook = renderHook(() => useArtifactGeneration(deps()));
    await act(async () => { await hook.result.current.handleGenerateOutput('summary'); });
    expect(hook.result.current.unsavedArtifact).not.toBeNull();
    io.state.generatedArtifacts = [{
      ...io.state.generatedArtifacts[0],
      workspaceArtifactId: 'stable-draft', workspaceArtifactVersion: 2, content: 'Saved newer explicit edit'
    }];
    hook.rerender();
    await waitFor(() => expect(hook.result.current.unsavedArtifact).toBeNull());
    const count = io.request.mock.calls.length;
    await act(async () => { await hook.result.current.handleRetryArtifactSave(); });
    expect(io.request.mock.calls).toHaveLength(count);
    expect(io.state.generatedArtifacts[0].content).toBe('Saved newer explicit edit');
  });

  it('retains canonical saving when a cached workspace is revisited after one hundred other restores', async () => {
    io.migrated = false;
    rememberCanonicalArtifactWorkspace('owned', 'server:alice');
    for (let index = 0; index < 101; index++)
      rememberCanonicalArtifactWorkspace('visited-workspace-' + index, 'server:alice');
    const hook = renderHook(() => useArtifactGeneration(deps()));
    await act(async () => { await hook.result.current.handleGenerateOutput('summary'); });
    expect(stored).toHaveLength(1);
    expect(mapServerArtifactToLocal(stored[0])).toMatchObject({
      id: 'stable-draft', content: 'Selene tours start Friday at 18:00.',
    });
  });

});
