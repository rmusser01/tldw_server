import type { MessageInstance } from 'antd/es/message/interface';
import type { TFunction } from 'i18next';
import React from 'react';
import { act, fireEvent, render, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Modal } from 'antd';
import type { GeneratedArtifact } from '@/types/workspace';
import { bindWorkspaceArtifactOrigin } from '@/store/workspace-artifact-codec';
import { useArtifactExport } from '../hooks/useArtifactExport';
import {
  clearWorkspaceUndoActionsForTests,
  getWorkspaceUndoPendingCount,
} from '../../undo-manager';

const runtime = vi.hoisted(() => ({
  state: { workspaceId: 'owned', generatedArtifacts: [] as GeneratedArtifact[] },
  save: vi.fn(),
  release: vi.fn(),
}));
vi.mock('@/store/workspace', () => ({
  useWorkspaceStore: { getState: () => runtime.state },
  hasResearchWorkspaceMigrationTombstone: () => true,
}));
vi.mock('@/services/workspace-artifact-persistence', () => ({
  persistWorkspaceArtifact: runtime.save,
}));
vi.mock('@/services/service-prompts', () => ({
  loadServicePromptSnapshot: async (_ids: unknown, options: { signal: AbortSignal }) => ({
    scopeKey: 'server:alice',
    requestScope: { config: { serverUrl: 'https://owned.test' }, userId: 'alice' },
    scopeSignal: options.signal,
    scopeInvalidatedSignal: options.signal,
    release: runtime.release,
  }),
}));

const messages = { open: vi.fn(), success: vi.fn(), error: vi.fn(), destroy: vi.fn() };
const draft = (): GeneratedArtifact =>
  bindWorkspaceArtifactOrigin(
    {
      id: 'report-owned',
      workspaceArtifactId: 'report-owned',
      workspaceArtifactVersion: 1,
      title: 'Saved report',
      type: 'report',
      status: 'completed',
      reviewStatus: 'draft',
      content: 'Keep the cited report',
      createdAt: new Date(),
    },
    'server:alice',
    'owned'
  );

beforeEach(() => {
  vi.clearAllMocks();
  runtime.state = { workspaceId: 'owned', generatedArtifacts: [] };
  runtime.save.mockImplementation(async (artifact) => ({
    ...artifact,
    workspaceArtifactVersion: artifact.workspaceArtifactVersion + 1,
  }));
});
afterEach(() => clearWorkspaceUndoActionsForTests());

const setup = (artifact = draft()) => {
  runtime.state.generatedArtifacts = [artifact];
  const remove = vi.fn((id) => {
    runtime.state.generatedArtifacts = runtime.state.generatedArtifacts.filter(
      (item) => item.id !== id
    );
  });
  const restore = vi.fn((artifact) => {
    runtime.state.generatedArtifacts = [artifact];
  });
  let confirmation: Parameters<typeof Modal.confirm>[0] | undefined;
  vi.spyOn(Modal, 'confirm').mockImplementation((config) => {
    confirmation = config;
    return { destroy: vi.fn(), update: vi.fn() };
  });
  const hook = renderHook(() =>
    useArtifactExport({
      workspaceId: 'owned',
      generatedArtifacts: [artifact],
      isMobile: false,
      messageApi: messages as unknown as MessageInstance,
      removeArtifact: remove,
      restoreArtifact: restore,
      captureToCurrentNote: vi.fn(),
      t: ((_key: string, fallback: unknown) => fallback) as unknown as TFunction,
    })
  );
  act(() => hook.result.current.handleDeleteArtifact(artifact));
  return { artifact, remove, restore, confirmation };
};

describe('saved Workspace output removal', () => {
  it('keeps unenrolled Studio-style outputs locally removable in a migrated workspace', async () => {
    const local: GeneratedArtifact = {
      id: 'local-summary',
      type: 'summary',
      title: 'Local output',
      status: 'completed',
      content: 'New normal Studio result',
      createdAt: new Date(),
    };
    const { confirmation, remove } = setup(local);
    await act(async () => {
      await confirmation.onOk();
    });
    expect(remove).toHaveBeenCalledWith(local.id);
    expect(runtime.save).not.toHaveBeenCalled();
  });
  it('waits for canonical removal and Undo acknowledgements before publishing local changes', async () => {
    let resolveSave!: (value: unknown) => void;
    runtime.save.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveSave = resolve;
        })
    );
    const { artifact, confirmation, remove, restore } = setup();
    let operation!: Promise<void>;
    act(() => {
      operation = confirmation.onOk();
    });
    await waitFor(() => expect(runtime.save).toHaveBeenCalledTimes(1));
    expect(remove).not.toHaveBeenCalled();
    expect(runtime.save.mock.calls[0][0]).toMatchObject({
      removedFromStudio: true,
      reviewStatus: 'archived',
    });
    await act(async () => {
      resolveSave({
        ...artifact,
        removedFromStudio: true,
        reviewStatus: 'archived',
        workspaceArtifactVersion: 2,
      });
      await operation;
    });
    expect(remove).toHaveBeenCalledWith(artifact.id);
    const toast = render(<>{messages.open.mock.calls[0][0].content}</>);
    fireEvent.click(toast.getByRole('button', { name: 'Undo' }));
    await waitFor(() =>
      expect(restore).toHaveBeenCalledWith(
        expect.objectContaining({
          id: artifact.id,
          content: artifact.content,
          removedFromStudio: false,
          reviewStatus: 'draft',
          workspaceArtifactVersion: 3,
        }),
        { index: 0 }
      )
    );
    expect(runtime.save).toHaveBeenCalledTimes(2);
    expect(messages.success).toHaveBeenCalledTimes(1);
  });

  it.each(['account', 'workspace', 'newer-record'])(
    'refuses a late removal acknowledgement after %s retirement',
    async (kind) => {
      let resolveSave!: (value: unknown) => void;
      runtime.save.mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveSave = resolve;
          })
      );
      const { artifact, confirmation, remove } = setup();
      const operation = confirmation.onOk() as Promise<void>;
      const rejected = expect(operation).rejects.toThrow('changed');
      await waitFor(() => expect(runtime.save).toHaveBeenCalledTimes(1));
      if (kind === 'account')
        act(() => window.dispatchEvent(new Event('tldw:auth-principal-changed')));
      else if (kind === 'workspace') runtime.state.workspaceId = 'another';
      else runtime.state.generatedArtifacts = [{ ...artifact, content: 'Newer local edit' }];
      await act(async () => {
        resolveSave({
          ...artifact,
          removedFromStudio: true,
          reviewStatus: 'archived',
          workspaceArtifactVersion: 2,
        });
        await rejected;
      });
      expect(remove).not.toHaveBeenCalled();
      expect(getWorkspaceUndoPendingCount()).toBe(0);
    }
  );

  it('retains the visible result when canonical removal fails', async () => {
    runtime.save.mockRejectedValueOnce(new Error('storage unavailable'));
    const { artifact, confirmation, remove } = setup();
    await expect(confirmation.onOk()).rejects.toThrow('storage unavailable');
    expect(remove).not.toHaveBeenCalled();
    expect(runtime.state.generatedArtifacts[0]).toBe(artifact);
    expect(getWorkspaceUndoPendingCount()).toBe(0);
  });
});
