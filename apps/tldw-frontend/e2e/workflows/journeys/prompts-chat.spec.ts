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
import { assertSavedTurn, readSavedMessages, type SavedMessage } from './uat390-grounding';

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
        const created = page.waitForResponse(response =>
          response.request().method() === 'POST' && /\/api\/v1\/chats\/?$/.test(response.url())
        );
        const completed = page.waitForResponse(response =>
          response.request().method() === 'POST' && /\/api\/v1\/chat\/completions(?:\?|$)/.test(response.url())
        );
        const [[creation, completion]] = await Promise.all([
          Promise.all([created, completed]), chat.sendMessage(QUESTION),
        ]);
        expect(creation.ok()).toBe(true);
        const chatId = String((await creation.json()).id);
        expect(chatId).not.toMatch(/^(?:undefined|null|local[-_]|$)/);
        expect(completion.status()).toBe(200);
        const sent = completion.request().postDataJSON();
        expect(sent).toMatchObject({ model, conversation_id: chatId, save_to_db: true });
        expect(sent.provider ?? sent.api_provider).toBe(provider);
        expect(sent.messages.map(({ role, content }: { role: string; content: unknown }) => ({ role, content })))
          .toEqual([{ role: 'system', content: SYSTEM }, { role: 'user', content: QUESTION }]);
        evidence.chatConversationId = chatId;
        evidence.completionRequest = sent;
        await waitForStreamComplete(page, 90_000);
        await chat.waitForResponse();
        const visible = (await chat.getMessages()).filter(message => message.role === 'assistant');
        expect(visible.map(message => message.content)).toEqual([ANSWER]);
        let saved: SavedMessage[] = [];
        await expect.poll(async () => {
          saved = readSavedMessages(
            (await apiGet(`/api/v1/chats/${chatId}/messages?render_placeholders=false`)).messages, chatId
          );
          return saved.filter(message => message.role === 'assistant').length;
        }, { timeout: 30_000 }).toBe(1);
        const savedAnswer = saved.find(message => message.role === 'assistant')!.content;
        expect(savedAnswer.trim()).toBe(ANSWER);
        assertSavedTurn(saved, QUESTION, savedAnswer);
        expect(saved.filter(message => message.role === 'system').map(message => message.content)).toEqual([SYSTEM]);
        evidence.chatMessages = saved;
        await page.goto(`/chat?settingsServerChatId=${encodeURIComponent(chatId)}`, { waitUntil: 'domcontentloaded' });
        await waitForConnection(page);
        await chat.waitForReady();
        await expect.poll(async () => (await chat.getMessages())
          .filter(message => message.role === 'assistant').map(message => message.content)).toEqual([ANSWER]);
        expect(readSavedMessages(
          (await apiGet(`/api/v1/chats/${chatId}/messages?render_placeholders=false`)).messages, chatId
        )).toEqual(saved);
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
