import React from 'react';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AssistantSelect } from '@/components/Common/AssistantSelect';
import { useCharacterGreeting } from '@/hooks/useCharacterGreeting';
import { useSelectedCharacter } from '@/hooks/useSelectedCharacter';
import { useSelectedAssistant } from '@/hooks/useSelectedAssistant';
import { useServerChatLoader } from '@/hooks/chat/useServerChatLoader';
import { useStoreMessageOption } from '@/store/option';
import {
  characterToAssistantSelection,
  getAssistantSelectionMode,
} from '@/types/assistant-selection';
import { selectedAssistantStorage } from '@/utils/selected-assistant-storage';
import { selectedCharacterStorage } from '@/utils/selected-character-storage';
import type { Character } from '@/types/character';

const mocks = vi.hoisted(() => ({
  getCharacter: vi.fn(),
  getChat: vi.fn(),
  listChatMessages: vi.fn(),
  listAllCharacters: vi.fn(),
  notification: { error: vi.fn(), warning: vi.fn(), success: vi.fn() },
}));
vi.mock('@/services/tldw/TldwApiClient', () => ({
  tldwClient: {
    initialize: async () => undefined,
    getCharacter: mocks.getCharacter,
    getChat: mocks.getChat,
    listChatMessages: mocks.listChatMessages,
    listAllCharacters: mocks.listAllCharacters,
    listPersonaProfiles: async () => [],
    ensureConfigForRequest: async () => ({
      serverUrl: 'http://127.0.0.1:8000',
      authMode: 'single-user',
      apiKey: 'test-key',
    }),
  },
}));
vi.mock('@/db/dexie/helpers', () => ({
  generateID: () => crypto.randomUUID(),
  getHistoriesWithMetadata: async () => new Map(),
  saveMessage: async () => undefined,
}));
vi.mock('@/hooks/chat/useChatSettingsRecord', () => ({
  useChatSettingsRecord: () => ({ settings: null, updateSettings }),
}));
vi.mock('@/services/chat-settings', () => ({
  syncChatSettingsForServerChat: async () => null,
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t }) }));

const t = (key: string, fallback?: string | { defaultValue?: string }) =>
  typeof fallback === 'string' ? fallback : (fallback?.defaultValue ?? key);
const updateSettings = async () => null;
const ensureServerChatHistoryId = async () => 'local-cedar';
const cedar: Character = {
  id: '4',
  name: 'Cycle3 Cedar Guide',
  greeting: 'Welcome to Cedar.',
  system_prompt: 'Answer using Cedar facts.',
  avatar_url: '',
  extensions: {},
};
const robot: Character = {
  id: '5',
  name: 'Cycle3 BEEP Robot',
  greeting: 'BEEP BOOP.',
  system_prompt: 'Always respond with exactly: BEEP BOOP.',
  avatar_url: '',
  extensions: {},
};
const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
};

// Keep the actual picker, canonical selection, storage notifications, greeting
// hydration, clear-chat and server loader together. Only I/O is controlled.
function SelectionWorkspace() {
  const [character, , meta] = useSelectedCharacter<Character>();
  const [assistant, setAssistant] = useSelectedAssistant();
  const state = useStoreMessageOption();
  useServerChatLoader({
    ensureServerChatHistoryId,
    notification: mocks.notification,
    t: t as never,
  });
  useCharacterGreeting({
    playgroundReady: !meta?.isLoading,
    selectedCharacter: character,
    selectedCharacterMode: getAssistantSelectionMode(assistant),
    serverChatId: state.serverChatId,
    historyId: state.historyId,
    messagesLength: state.messages.length,
    setMessages: state.setMessages,
    setHistory: state.setHistory,
  });
  return (
    <>
      <AssistantSelect variant="dropdown" />
      <button
        onClick={async () => {
          await setAssistant(null);
          state.setMessages([]);
        }}
      >
        Clear selection
      </button>
      <output aria-label="Selected assistant">{assistant?.id ?? 'none'}</output>
      <output aria-label="Request system prompt">{assistant?.system_prompt}</output>
      <output aria-label="Conversation messages">
        {state.messages.map((m) => m.message).join('\n')}
      </output>
    </>
  );
}

describe('character picker with greeting and server hydration', () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    localStorage.clear();
    useStoreMessageOption.setState(useStoreMessageOption.getInitialState(), true);
    await selectedAssistantStorage.set('selectedAssistant', characterToAssistantSelection(cedar));
    await selectedCharacterStorage.set('selectedCharacter', cedar);
    mocks.listAllCharacters.mockResolvedValue([cedar, robot]);
    mocks.getCharacter.mockImplementation(async (id) => (String(id) === '4' ? cedar : robot));
    mocks.getChat.mockResolvedValue({ id: 'cedar-chat', character_id: 4, name: 'Cedar chat' });
    mocks.listChatMessages.mockResolvedValue([]);
  });
  afterEach(() => vi.restoreAllMocks());

  it('keeps an explicit Robot pick when the old Cedar legacy read finishes later', async () => {
    useStoreMessageOption.setState({
      serverChatId: 'cedar-chat',
      serverChatCharacterId: 4,
      serverChatAssistantKind: 'character',
      serverChatMetaLoaded: true,
      messages: [{ id: 'cedar-answer', isBot: true, message: 'Saved Cedar answer', sources: [] }],
    });
    const user = userEvent.setup();
    render(<SelectionWorkspace />);
    await waitFor(() => expect(screen.getByLabelText('Selected assistant')).toHaveTextContent('4'));
    expect(screen.getByLabelText('Conversation messages')).toHaveTextContent('Saved Cedar answer');
    await user.click(screen.getByRole('button', { name: 'Cycle3 Cedar Guide', exact: true }));

    const legacyRead = deferred<Character>();
    vi.spyOn(selectedCharacterStorage, 'get').mockImplementation(() => legacyRead.promise as never);
    await user.click(await screen.findByRole('button', { name: 'Cycle3 BEEP Robot', exact: true }));
    await act(async () => {
      legacyRead.resolve(cedar);
      await legacyRead.promise;
    });

    await waitFor(() => {
      expect(screen.getByLabelText('Selected assistant')).toHaveTextContent('5');
      expect(screen.getByLabelText('Request system prompt')).toHaveTextContent(
        'Always respond with exactly: BEEP BOOP.'
      );
      expect(screen.getByLabelText('Conversation messages')).toHaveTextContent('BEEP BOOP.');
    });
    expect(screen.getByLabelText('Conversation messages')).not.toHaveTextContent(
      'Welcome to Cedar.'
    );
  });

  it('ignores the first Cedar profile after choosing Robot and then Cedar again', async () => {
    const oldProfile = deferred<Character>();
    let cedarReads = 0;
    mocks.getCharacter.mockImplementation(async (id) => {
      if (String(id) !== '4') return robot;
      cedarReads += 1;
      return cedarReads === 1 ? oldProfile.promise : cedar;
    });
    const user = userEvent.setup();
    render(<SelectionWorkspace />);
    await waitFor(() => expect(cedarReads).toBe(1));
    await user.click(screen.getByRole('button', { name: 'Cycle3 Cedar Guide', exact: true }));
    await user.click(await screen.findByRole('button', { name: 'Cycle3 BEEP Robot', exact: true }));
    await waitFor(() => expect(screen.getByLabelText('Selected assistant')).toHaveTextContent('5'));
    await user.click(screen.getByTestId('character-select'));
    await user.click(
      await screen.findByRole('button', { name: 'Cycle3 Cedar Guide', exact: true })
    );
    await waitFor(() => expect(cedarReads).toBe(2));
    await act(async () => {
      oldProfile.resolve({
        ...cedar,
        greeting: 'Obsolete Cedar greeting',
        system_prompt: 'Obsolete Cedar instructions',
      });
      await oldProfile.promise;
    });
    expect(screen.getByLabelText('Selected assistant')).toHaveTextContent('4');
    expect(screen.getByLabelText('Conversation messages')).toHaveTextContent('Welcome to Cedar.');
    expect(screen.getByLabelText('Request system prompt')).toHaveTextContent(
      'Answer using Cedar facts.'
    );
  });

  it.each([
    ['tldw:auth-principal-changed', { kind: 'logout' }],
    ['tldw:config-updated', { authorityChanged: true }],
  ])('ignores a pending profile after the %s account boundary', async (eventName, detail) => {
    const oldProfile = deferred<Character>();
    mocks.getCharacter.mockReturnValue(oldProfile.promise);
    render(<SelectionWorkspace />);
    await waitFor(() => expect(mocks.getCharacter).toHaveBeenCalledWith('4'));
    await act(async () => {
      window.dispatchEvent(new CustomEvent(eventName, { detail }));
      oldProfile.resolve({ ...cedar, greeting: 'Previous account private greeting' });
      await oldProfile.promise;
    });
    expect(screen.getByLabelText('Conversation messages')).not.toHaveTextContent(
      'Previous account private greeting'
    );
  });

  it('keeps Robot selected when the abandoned saved Cedar loader finishes', async () => {
    const oldProfile = deferred<Character>();
    useStoreMessageOption.setState({
      serverChatId: 'cedar-chat',
      serverChatCharacterId: 4,
      serverChatAssistantKind: 'character',
      // Leave saved metadata unresolved so the canonical loader owns the
      // pending Character read that the picker must supersede.
      serverChatMetaLoaded: false,
    });
    mocks.getCharacter.mockImplementation(async (id) =>
      String(id) === '4' ? oldProfile.promise : robot
    );
    const user = userEvent.setup();
    render(<SelectionWorkspace />);
    await waitFor(() => expect(mocks.getCharacter).toHaveBeenCalledWith('4'));
    await user.click(screen.getByTestId('character-select'));
    await user.click(await screen.findByRole('button', { name: 'Cycle3 BEEP Robot', exact: true }));
    await waitFor(() => expect(screen.getByLabelText('Selected assistant')).toHaveTextContent('5'));
    await act(async () => {
      oldProfile.resolve(cedar);
      await oldProfile.promise;
    });
    expect(screen.getByLabelText('Selected assistant')).toHaveTextContent('5');
    expect(useStoreMessageOption.getState().serverChatId).toBeNull();
    expect(screen.getByLabelText('Conversation messages')).toHaveTextContent('BEEP BOOP.');
  });

  it('does not resurrect a cleared selection from its pending greeting', async () => {
    const oldProfile = deferred<Character>();
    mocks.getCharacter.mockReturnValue(oldProfile.promise);
    const user = userEvent.setup();
    render(<SelectionWorkspace />);
    await waitFor(() => expect(mocks.getCharacter).toHaveBeenCalledWith('4'));
    await user.click(screen.getByRole('button', { name: 'Clear selection' }));
    await waitFor(() =>
      expect(screen.getByLabelText('Selected assistant')).toHaveTextContent('none')
    );
    await act(async () => {
      oldProfile.resolve({ ...cedar, greeting: 'Late cleared greeting' });
      await oldProfile.promise;
    });
    expect(screen.getByLabelText('Selected assistant')).toHaveTextContent('none');
    expect(screen.getByLabelText('Conversation messages')).not.toHaveTextContent(
      'Late cleared greeting'
    );
  });

  it('retains greeting hydration through a benign configuration update', async () => {
    const profile = deferred<Character>();
    mocks.getCharacter.mockReturnValue(profile.promise);
    render(<SelectionWorkspace />);
    await waitFor(() => expect(mocks.getCharacter).toHaveBeenCalledWith('4'));
    await act(async () => {
      window.dispatchEvent(
        new CustomEvent('tldw:config-updated', { detail: { authorityChanged: false } })
      );
      profile.resolve({ ...cedar, greeting: 'Current hydrated greeting' });
      await profile.promise;
    });
    expect(screen.getByLabelText('Conversation messages')).toHaveTextContent(
      'Current hydrated greeting'
    );
  });
});
