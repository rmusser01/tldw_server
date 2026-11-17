import React from 'react';
import type { MessageInstance } from 'antd/es/message/interface';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { message } from 'antd';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { FlashcardArtifactEditor, QuizArtifactEditor } from '../ArtifactModalContent';
import {
  clearWorkspaceUndoActionsForTests, getWorkspaceUndoPendingCount,
  scheduleWorkspaceUndoAction, undoWorkspaceAction,
} from '../../undo-manager';

vi.mock('@/utils/research-workspace-telemetry', () => ({ trackResearchWorkspaceTelemetry: vi.fn() }));
const api = { open: vi.fn(), success: vi.fn(), destroy: vi.fn() };
beforeEach(() => {
  vi.clearAllMocks();
  clearWorkspaceUndoActionsForTests();
  vi.spyOn(message, 'useMessage').mockReturnValue([api as unknown as MessageInstance, <></>]);
});
afterEach(() => clearWorkspaceUndoActionsForTests());

describe('artifact editor removal Undo save boundary', () => {
  it.each(['flashcards', 'quiz'] as const)('retires only this %s editor removal before a held save and ignores its stale portal action', async (type) => {
    const unrelatedUndo = vi.fn();
    const unrelated = scheduleWorkspaceUndoAction({ apply: vi.fn(), undo: unrelatedUndo });
    let finish!: () => void;
    const save = vi.fn(() => new Promise<void>(resolve => { finish = resolve; }));
    render(type === 'flashcards'
      ? <FlashcardArtifactEditor cards={[{ front: 'Removed first', back: 'First answer' },
      { front: 'Kept second', back: 'Second answer' }]} onSave={save} />
      : <QuizArtifactEditor questions={[{ question: 'Removed first', options: ['yes'], answer: 'yes' },
      { question: 'Kept second', options: ['yes'], answer: 'yes' }]} onSave={save} />);
    fireEvent.click(screen.getAllByRole('button', { name: 'Remove' })[0]);
    const stalePortal = render(<>{api.open.mock.calls[0][0].content}</>);
    expect(getWorkspaceUndoPendingCount()).toBe(2);
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await waitFor(() => expect(save).toHaveBeenCalledTimes(1));
    expect(getWorkspaceUndoPendingCount()).toBe(1);
    fireEvent.click(stalePortal.getByRole('button', { name: 'Undo' }));
    expect(screen.queryByDisplayValue('Removed first')).not.toBeInTheDocument();
    expect(screen.getByDisplayValue('Kept second')).toBeInTheDocument();
    expect(api.success).not.toHaveBeenCalled();
    await act(async () => { finish(); });
    expect(undoWorkspaceAction(unrelated.id)).toBe(true);
    expect(unrelatedUndo).toHaveBeenCalledTimes(1);
  });
});
