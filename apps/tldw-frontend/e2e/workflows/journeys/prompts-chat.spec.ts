/**
 * C-01 engineering regression: saved Prompt -> system instruction -> saved Chat.
 * Seeded auth and a controlled downstream model do not certify fresh-install UAT.
 * Export/import, failed sync, variables and account-switch variants remain separate.
 */
import { createHash, randomUUID } from 'node:crypto';
import { test, expect } from '../../utils/fixtures';
import { PromptsWorkspacePage, ChatPage } from '../../utils/page-objects';
import { waitForStreamComplete } from '../../utils/journey-helpers';
import { TEST_CONFIG, fetchWithApiKey, waitForConnection } from '../../utils/helpers';
import { assertSavedTurn } from './uat390-grounding';
import { readStore } from '../../../../extension/tests/e2e/utils/history-selection';
import type { HistoryBookmark, HistoryInfo, Message } from '../../../../packages/ui/src/db/dexie/types';

const SYSTEM = 'You are a pirate. Respond to everything in pirate speak. Always say ARRR at least once.';
const QUESTION = 'Tell me about the weather today.';
// The provider selects this only for the exact system/user pair. The ordinary
// default response cannot satisfy this oracle if applying the prompt is broken.
const ANSWER = "ARRR, matey! I need yer location and current weather data before I can give ye today's forecast.";

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
        const nativeWrites: string[] = [];
        page.on('request', request => {
          if (request.method() === 'POST' && /\/api\/v1\/(?:chats(?:\/|$)|chat\/conversations(?:\/|$))/.test(request.url()))
            nativeWrites.push(request.url());
        });
        const completed = page.waitForResponse(response =>
          response.request().method() === 'POST' && /\/api\/v1\/chat\/completions(?:\?|$)/.test(response.url())
        );
        const [completion] = await Promise.all([completed, chat.sendMessage(QUESTION)]);
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
        let users: Message[] = [];
        await expect.poll(async () => {
          users = ((await readStore(page, 'messages')) as Message[])
            .filter(row => row.role === 'user' && row.content === QUESTION);
          return users.length;
        }).toBe(1);
        const user = users[0];
        const historyId = user.history_id;
        const history = ((await readStore(page, 'chatHistories')) as HistoryInfo[])
          .find(row => row.id === historyId)!;
        const keyScope = createHash('sha256').update('tldw:service-prompt-single-user-api-key:v1\0' + TEST_CONFIG.apiKey.trim()).digest('hex');
        expect(history.server_scope_key).toBe(JSON.stringify([TEST_CONFIG.serverUrl, 'single-user', 'manual', null, null, 'key:sha256:' + keyScope]));
        expect(history.server_chat_id).toBeUndefined();
        expect(history.local_owner_key).toMatch(/^local-history-v1:/);
        const saved = ((await readStore(page, 'messages')) as Message[]).filter(row => row.history_id === historyId);
        expect(saved).toHaveLength(2);
        const assistant = saved.find(row => row.role === 'assistant')!;
        assertSavedTurn([user, assistant], QUESTION, ANSWER);
        expect(user.parent_message_id).toBeNull();
        expect(assistant.parent_message_id).toBe(user.id);
        expect(user.history_admission).toMatchObject({ owner_key: history.local_owner_key, conversation_id: historyId, input_message_id: user.id });
        expect(assistant.history_settlement?.input_message_id).toBe(user.id);
        expect(saved.every(row => row.history_provenance?.owner_key === history.local_owner_key)).toBe(true);
        expect(saved.every(row => !row.serverMessageId)).toBe(true);
        let bookmarks: HistoryBookmark[] = [];
        await expect.poll(async () => {
          bookmarks = ((await readStore(page, 'historySelections')) as HistoryBookmark[])
            .filter(row => row.conversation_id === historyId && row.view.cursor.kind === 'after_message' && row.view.cursor.message_id === assistant.id);
          return bookmarks.length;
        }).toBe(1);
        const bookmark = bookmarks[0];
        expect(history.local_owner_key).toBe('local-history-v1:' + bookmark.profile_id);
        expect(bookmark.view.owner_key).toBe(history.local_owner_key);
        expect(bookmark.view.interpretation).toEqual({ kind: 'parent_graph_v1' });
        const reference = { profile_id: bookmark.profile_id, client_session_id: bookmark.client_session_id,
          owner_key: bookmark.owner_key, conversation_id: historyId, owner_kind: 'local' };
        evidence.chatConversationId = historyId;
        evidence.chatHistory = history;
        evidence.chatMessages = saved;
        evidence.chatSelection = reference;
        await page.goto('/chat?historySelection=' + encodeURIComponent(JSON.stringify(reference)), { waitUntil: 'domcontentloaded' });
        await waitForConnection(page);
        await chat.waitForReady();
        await expect.poll(async () => (await chat.getMessages())
          .filter(message => message.role === 'assistant').map(message => message.content)).toEqual([ANSWER]);
        await expect(page).toHaveURL(/\/chat$/);
        await page.reload({ waitUntil: 'domcontentloaded' });
        await waitForConnection(page);
        await chat.waitForReady();
        await expect.poll(async () => (await chat.getMessages())
          .filter(message => message.role === 'assistant').map(message => message.content)).toEqual([ANSWER]);
        expect(((await readStore(page, 'messages')) as Message[]).filter(row => row.history_id === historyId)).toEqual(saved);
        expect(((await readStore(page, 'chatHistories')) as HistoryInfo[]).find(row => row.id === historyId)).toEqual(history);
        expect(nativeWrites).toEqual([]);
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
