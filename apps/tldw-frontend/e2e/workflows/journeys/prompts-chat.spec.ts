/**
 * C-01 engineering regression: saved Prompt -> system instruction -> saved Chat.
 * Seeded auth and a controlled downstream model do not certify fresh-install UAT.
 * Export/import, failed sync, variables and account-switch variants remain separate.
 */
import { randomUUID } from 'node:crypto';
import { test, expect } from '../../utils/fixtures';
import { PromptsWorkspacePage, ChatPage } from '../../utils/page-objects';
import { waitForStreamComplete } from '../../utils/journey-helpers';
import { TEST_CONFIG, fetchWithApiKey, waitForConnection } from '../../utils/helpers';
import { assertSavedTurn, readSavedMessages } from './uat390-grounding';
import { readStore } from '../../../../extension/tests/e2e/utils/history-selection';
import type { HistoryBookmark } from '../../../../packages/ui/src/db/dexie/types';

const SYSTEM = 'You are a pirate. Respond to everything in pirate speak. Always say ARRR at least once.';
const QUESTION = 'Tell me about the weather today.';
// The provider selects this only for the exact system/user pair. The ordinary
// default response cannot satisfy this oracle if applying the prompt is broken.
const ANSWER = "ARRR, matey! I need yer location and current weather data before I can give ye today's forecast.";

type CanonicalTurnMessage = {
  id: string; conversation_id: string; content: string; sender: string; parent_message_id: string | null;
};

test.describe('Prompts -> Chat journey', () => {
  test('applies a saved system prompt and preserves its successful Chat turn on reload', async ({
    authedPage: page,
    serverInfo,
  }, testInfo) => {
    test.setTimeout(180_000);
    expect(serverInfo.available, 'The real application backend is required').toBe(true);
    expect(process.env.UAT_PROMPT_MODE, 'This regression requires its declared controlled provider').toBe('deterministic');
    const provider = process.env.UAT_PROMPT_PROVIDER;
    const model = process.env.UAT_PROMPT_MODEL;
    expect(provider).toBeTruthy();
    expect(model).toBeTruthy();
    await page.unrouteAll({ behavior: 'wait' });
    const promptName = `E2E-Pirate-${randomUUID()}`;
    const evidence: Record<string, unknown> = { promptName, mode: 'deterministic', auth: 'seeded' };
    const apiGet = async (path: string) => {
      const response = await fetchWithApiKey(`${TEST_CONFIG.serverUrl}${path}`);
      expect(response.ok, `Canonical GET ${path}: HTTP ${response.status}`).toBe(true);
      return response.json();
    };
    let promptId = 0;
    let promptRowId = '';
    const prompts = new PromptsWorkspacePage(page);
    const inspector = page.getByTestId('prompts-inspector-panel-scaffold');
    try {
      await test.step('Save and reopen the exact synced prompt through the UI', async () => {
        await prompts.goto();
        await prompts.assertPageReady();
        const synced = page.waitForResponse(response =>
          response.request().method() === 'POST' &&
          /\/api\/v1\/prompt-studio\/prompts\/create$/.test(response.url()) &&
          response.request().postDataJSON().name === promptName
        );
        const [response] = await Promise.all([
          synced,
          prompts.createPrompt({ name: promptName, template: SYSTEM }),
        ]);
        expect(response.ok()).toBe(true);
        expect(response.request().postDataJSON()).toMatchObject({ name: promptName, system_prompt: SYSTEM });
        const body = await response.json();
        expect(body.success).toBe(true);
        promptId = body.data.id;
        expect(Number.isInteger(promptId) && promptId > 0).toBe(true);
        expect(body.data).toMatchObject({ id: promptId, name: promptName, system_prompt: SYSTEM });
        evidence.serverPromptId = promptId;
        const row = page.locator('[data-testid^="prompt-row-"], [data-testid^="prompt-gallery-card-"]')
          .filter({ has: page.getByText(promptName, { exact: true }) });
        await expect(row).toHaveCount(1);
        promptRowId = (await row.getAttribute('data-testid'))!;
        expect(promptRowId).toMatch(/^prompt-(?:row|gallery-card)-.+/);
        evidence.localPromptRowId = promptRowId;
        await page.reload({ waitUntil: 'domcontentloaded' });
        await waitForConnection(page);
        await prompts.assertPromptVisible(promptName);
        await page.getByTestId(promptRowId).getByText(promptName, { exact: true }).click();
        await expect(inspector).toBeVisible();
        await expect(inspector.getByText(promptName, { exact: true })).toBeVisible();
        await expect(inspector.getByText(SYSTEM, { exact: true })).toBeVisible();
        const savedPrompt = await apiGet(`/api/v1/prompt-studio/prompts/get/${promptId}`);
        expect(savedPrompt.success).toBe(true);
        expect(savedPrompt.data).toMatchObject({ id: promptId, name: promptName, system_prompt: SYSTEM });
      });
      const chat = new ChatPage(page);
      await test.step('Use the saved prompt as the actual Chat system instruction', async () => {
        await inspector.getByRole('button', { name: 'Use', exact: true }).click();
        const choice = page.getByTestId('prompt-insert-system');
        await expect(choice).toContainText(SYSTEM);
        await choice.click();
        await expect(page).toHaveURL(/\/chat(?:\?|$)/);
        await waitForConnection(page);
        await chat.waitForReady();
        await chat.selectModel(model!);
        const createdChat = page.waitForResponse(response =>
          response.request().method() === 'POST' && /\/api\/v1\/chats\/?$/.test(response.url())
        );
        const admittedUser = page.waitForResponse(response =>
          response.request().method() === 'POST' && /\/api\/v1\/chats\/[^/]+\/messages(?:\?|$)/.test(response.url()) &&
          response.request().postDataJSON().role === 'user'
        );
        const settledAssistant = page.waitForResponse(response =>
          response.request().method() === 'POST' && /\/api\/v1\/chats\/[^/]+\/messages(?:\?|$)/.test(response.url()) &&
          response.request().postDataJSON().role === 'assistant',
          { timeout: 90_000 }
        );
        const completed = page.waitForResponse(response =>
          response.request().method() === 'POST' && /\/api\/v1\/chat\/completions(?:\?|$)/.test(response.url())
        );
        const [creation, userWrite, completion, assistantWrite] = await Promise.all([
          createdChat, admittedUser, completed, settledAssistant, chat.sendMessage(QUESTION),
        ]);
        expect(creation.status()).toBe(201);
        const chatId = (await creation.json()).id;
        expect(chatId).toEqual(expect.any(String));
        expect(chatId).not.toMatch(/^(?:local[-_]|$)/);
        expect(completion.status()).toBe(200);
        const sent = completion.request().postDataJSON();
        expect(sent).toMatchObject({ model, save_to_db: false });
        expect(sent).not.toHaveProperty('conversation_id');
        expect(sent.provider ?? sent.api_provider).toBe(provider);
        expect(sent.messages.map(({ role, content }: { role: string; content: unknown }) => ({ role, content })))
          .toEqual([{ role: 'system', content: SYSTEM }, { role: 'user', content: QUESTION }]);
        evidence.completionRequest = sent;
        await waitForStreamComplete(page, 90_000);
        await chat.waitForResponse();
        const visible = (await chat.getMessages()).filter(message => message.role === 'assistant');
        expect(visible.map(message => message.content)).toEqual([ANSWER]);
        expect(userWrite.status()).toBe(201);
        expect(assistantWrite.status()).toBe(201);
        for (const write of [userWrite, assistantWrite])
          expect(new URL(write.url()).pathname).toBe(`/api/v1/chats/${chatId}/messages`);
        const userReceipt = await userWrite.json();
        const assistantReceipt = await assistantWrite.json();
        const userRequest = userWrite.request().postDataJSON();
        const selection = userRequest.tldw_history_selection_v1;
        expect(selection).toMatchObject({ version: 1, conversation_id: chatId,
          owner_key: expect.stringMatching(/^native-history-v1:sha256:/), cursor: { kind: 'empty' }, messages: [] });
        const admission = userReceipt.tldw_history_admission_v1;
        expect(admission).toMatchObject({ version: 1, conversation_id: chatId,
          owner_key: selection.owner_key, input_message_id: userReceipt.id,
          selection_digest: selection.selection_digest, messages: selection.messages,
          originating_selection_revision: selection.selection_revision });
        expect(admission.input_message_revision).toEqual(expect.any(String));
        expect(admission.input_message_revision).not.toBe('');
        expect(userRequest).toMatchObject({ id: userReceipt.id, role: 'user', content: QUESTION });
        expect(userRequest.parent_message_id ?? null).toBeNull();
        const assistantRequest = assistantWrite.request().postDataJSON();
        expect(assistantRequest).toMatchObject({ id: assistantReceipt.id, role: 'assistant',
          content: ANSWER, parent_message_id: userReceipt.id });
        expect(assistantRequest.tldw_history_admission_v1).toEqual({ version: 1,
          owner_key: admission.owner_key, conversation_id: chatId, input_message_id: userReceipt.id,
          input_message_revision: admission.input_message_revision, selection_digest: admission.selection_digest });
        let saved: CanonicalTurnMessage[] = [];
        await expect.poll(async () => {
          saved = (await apiGet(`/api/v1/chats/${chatId}/messages?render_placeholders=false`)).messages;
          return saved.length;
        }).toBe(2);
        const user = saved.find(row => row.id === userReceipt.id)!;
        const assistant = saved.find(row => row.id === assistantReceipt.id)!;
        expect(user).toMatchObject({ sender: 'user', content: QUESTION, conversation_id: chatId, parent_message_id: null });
        expect(assistant).toMatchObject({ sender: 'assistant', content: ANSWER, conversation_id: chatId, parent_message_id: user.id });
        assertSavedTurn(readSavedMessages(saved, chatId), QUESTION, ANSWER);
        let bookmarks: HistoryBookmark[] = [];
        await expect.poll(async () => {
          bookmarks = ((await readStore(page, 'historySelections')) as HistoryBookmark[])
            .filter(row => row.conversation_id === chatId && row.owner_key === admission.owner_key &&
              row.view.cursor.kind === 'after_message' && row.view.cursor.message_id === assistant.id);
          return bookmarks.length;
        }).toBe(1);
        const bookmark = bookmarks[0];
        expect(bookmark.view.owner_key).toBe(admission.owner_key);
        expect(bookmark.view.conversation_id).toBe(chatId);
        expect(bookmark.view.interpretation).toEqual({ kind: 'parent_graph_v1' });
        const reference = { profile_id: bookmark.profile_id, client_session_id: bookmark.client_session_id,
          owner_key: bookmark.owner_key, conversation_id: chatId, owner_kind: 'native' };
        evidence.chatConversationId = chatId;
        evidence.chatAdmission = admission;
        evidence.chatSettlement = assistantReceipt;
        evidence.chatMessages = saved;
        evidence.chatSelection = reference;
        const visiblePair = [{ role: 'user', content: QUESTION }, { role: 'assistant', content: ANSWER }];
        await page.goto('/chat?historySelection=' + encodeURIComponent(JSON.stringify(reference)), { waitUntil: 'domcontentloaded' });
        await waitForConnection(page);
        await chat.waitForReady();
        await expect.poll(async () => (await chat.getMessages())
          .map(({ role, content }) => ({ role, content }))).toEqual(visiblePair);
        await expect(page).toHaveURL(/\/chat$/);
        await page.reload({ waitUntil: 'domcontentloaded' });
        await waitForConnection(page);
        await chat.waitForReady();
        await expect.poll(async () => (await chat.getMessages())
          .map(({ role, content }) => ({ role, content }))).toEqual(visiblePair);
        expect((await apiGet(`/api/v1/chats/${chatId}/messages?render_placeholders=false`)).messages).toEqual(saved);
      });
      await test.step('The source Prompt still exists with the same identity and instructions', async () => {
        await prompts.goto();
        await prompts.assertPromptVisible(promptName);
        await page.getByTestId(promptRowId).getByText(promptName, { exact: true }).click();
        await expect(inspector.getByText(SYSTEM, { exact: true })).toBeVisible();
        const savedPrompt = await apiGet(`/api/v1/prompt-studio/prompts/get/${promptId}`);
        expect(savedPrompt.success).toBe(true);
        expect(savedPrompt.data).toMatchObject({ id: promptId, name: promptName, system_prompt: SYSTEM });
      });
    } finally {
      await testInfo.attach('c01-prompt-chat-evidence.json', {
        body: JSON.stringify(evidence, null, 2), contentType: 'application/json',
      });
    }
  });
});
