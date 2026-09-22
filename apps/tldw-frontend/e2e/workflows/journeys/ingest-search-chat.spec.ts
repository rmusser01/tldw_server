/**
 * UAT390: A-05 TXT -> A-07 Media-only FTS/citations/loaded Media-to-Chat.
 * Seeded auth is not fresh-setup UAT. TLDW_UAT390_MODE=deterministic means a
 * real backend with controlled downstream inference; live means a real model.
 * No application API fulfillment. Wikipedia, loading-race, vector/hybrid and
 * A-09 reuse are separate gates. Retain IDs for linked follow-up and review.
 */
import { createHash, randomUUID } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
import { test, expect } from '../../utils/fixtures';
import { ChatPage, KnowledgeQAPage } from '../../utils/page-objects';
import { ingestAndWaitForReady, waitForStreamComplete } from '../../utils/journey-helpers';
import { TEST_CONFIG, fetchWithApiKey, waitForConnection } from '../../utils/helpers';
import {
  ROWAN_SOURCE,
  LARCH_SOURCE,
  ROWAN_QUESTION,
  PRICE_QUESTION,
  assertCitedRowanAnswer,
  assertFullSourceHandoff,
  assertPriceAbstention,
  assertRowanFacts,
  assertSavedTurn,
  readMediaRecord,
  readSavedMessages,
  type SavedMessage,
} from './uat390-grounding';

test.describe('Ingest -> Search -> Chat journey', () => {
  test('retains the exact owned Rowan source through cited QA, Chat and reload', async ({
    authedPage: page,
    serverInfo,
  }, testInfo) => {
    test.setTimeout(360_000);
    const mode = process.env.TLDW_UAT390_MODE;
    const provider = process.env.TLDW_UAT390_PROVIDER;
    const providerOption = process.env.TLDW_UAT390_PROVIDER_OPTION ?? provider;
    const model = process.env.TLDW_UAT390_MODEL;
    expect(['deterministic', 'live'], 'Declare the downstream provider test mode').toContain(mode);
    expect(provider, 'Set the actual Knowledge QA provider').toBeTruthy();
    expect(model, 'Set the actual Knowledge QA model').toBeTruthy();
    expect(serverInfo.available, 'Owned backend is required; unavailable is not a pass').toBe(true);
    // Remove the legacy title-settings fixture fulfillment before navigation.
    await page.unrouteAll({ behavior: 'wait' });
    const runId = `uat390-${randomUUID()}`;
    const evidence: Record<string, unknown> = {
      runId,
      mode,
      provider,
      providerOption,
      model,
      auth: 'seeded',
      variant: 'txt-fts-loaded-handoff',
    };
    const apiGet = async (path: string) => {
      const response = await fetchWithApiKey(`${TEST_CONFIG.serverUrl}${path}`);
      expect(response.ok, `Canonical GET ${path}: HTTP ${response.status}`).toBe(true);
      return response.json();
    };
    try {
      const owned: Array<{ id: string; title: string; text: string }> = [];
      for (const [fixture, fixtureText] of [
        ['F-SOURCE', ROWAN_SOURCE],
        ['F-DISTRACTOR', LARCH_SOURCE],
      ] as const) {
        await test.step(`Ingest and corroborate ${fixture}`, async () => {
          // Ingestion deduplicates content even when the filename changes.
          const text = `${fixtureText.trimEnd()}\n\nFixture run: ${runId}\n`;
          const fileName = `${runId}-${fixture}.txt`;
          const filePath = testInfo.outputPath(fileName);
          await writeFile(filePath, text, 'utf8');
          await ingestAndWaitForReady(page, { file: filePath });
          // Never accept the helper's job/batch/unknown media-ID fallback.
          await page
            .getByRole('dialog', { name: /quick ingest/i })
            .getByRole('button', { name: /^Open .+ in Media$/i })
            .click();
          await expect(page).toHaveURL(/\/media\?[^#]*\bid=\d+/);
          const id = new URL(page.url()).searchParams.get('id')!;
          const record = await apiGet(`/api/v1/media/${id}`);
          const savedSource = readMediaRecord(record, id, text, runId);
          owned.push(savedSource);
          evidence[fixture] = {
            id,
            title: savedSource.title,
            sha256: createHash('sha256').update(text).digest('hex'),
          };
        });
      }
      const [source, distractor] = owned;
      expect(source.id).not.toBe(distractor.id);
      await test.step('Media search resolves Rowan and excludes the distractor', async () => {
        await page.goto('/media', { waitUntil: 'domcontentloaded' });
        await waitForConnection(page);
        await page.getByTestId('media-search-input').fill('Cedar Ridge');
        const searched = page.waitForResponse(
          (response) =>
            /\/api\/v1\/media\/search(?:\?|$)/.test(response.url()) &&
            response.request().method() === 'POST'
        );
        await page.getByTestId('media-search-submit').click();
        const response = await searched;
        expect(response.ok()).toBe(true);
        const body = await response.json();
        const results = body.items;
        expect(Array.isArray(results), 'Media search must return actual records').toBe(true);
        const ids = results.map((item: { id: string | number }) => String(item.id));
        expect(ids).toContain(source.id);
        expect(ids).not.toContain(distractor.id);
        await expect(page.getByText(source.title, { exact: true }).first()).toBeVisible();
        await expect(page.getByText(distractor.title, { exact: true })).toHaveCount(0);
        evidence.mediaSearchIds = ids;
      });
      const qa = new KnowledgeQAPage(page);
      let qaId = '';
      let qaAnswer = '';
      let qaMessages: ReturnType<typeof readSavedMessages> = [];
      await test.step('Ask scoped Media-only FTS using the selected provider', async () => {
        await qa.goto();
        await qa.waitForReady();
        await qa.openSettings();
        const settings = qa.getSettingsDialog();
        const expert = qa.getExpertModeToggle();
        if ((await expert.getAttribute('aria-checked')) === 'true') await expert.click();
        // Native BasicSettings/ExpertSettings selects have no associated label.
        await settings
          .getByRole('combobox')
          .filter({ has: page.locator('option[value="fts"]') })
          .selectOption('fts');
        for (const label of [
          'Documents & Media',
          'Notes',
          'Chats',
          'Characters',
          'Task Boards',
          'Prompts',
          'World Books',
          'Dictionaries',
        ]) {
          await settings
            .getByRole('checkbox', { name: label, exact: true })
            .setChecked(label === 'Documents & Media');
        }
        for (const [name, enabled] of [
          ['Generate Answer', true],
          ['Include Citations', true],
          ['Web Search Fallback', false],
          ['Enable Reranking', false],
        ] as const) {
          const control = settings.getByRole('switch', { name, exact: true });
          if (((await control.getAttribute('aria-checked')) === 'true') !== enabled)
            await control.click();
        }
        await expert.click();
        await settings
          .getByRole('combobox')
          .filter({ has: page.locator('option[value="media"]') })
          .selectOption('media');
        await page.keyboard.press('Escape');
        // Both controls stay in scope; filtering out Larch here would hide a retrieval defect.
        await qa.selectSpecificSource('media', source.title);
        await qa.selectSpecificSource('media', distractor.title);
        await page.getByRole('button', { name: 'Choose answer model' }).click();
        const modelDialog = page.getByRole('dialog', { name: 'Answer model controls' });
        await modelDialog.getByLabel('Answer provider', { exact: true }).selectOption(providerOption!);
        await modelDialog.getByLabel('Answer model', { exact: true }).fill(model!);
        await page.getByRole('button', { name: 'Choose answer model' }).click();
        const created = page.waitForResponse(
          (response) =>
            response.request().method() === 'POST' && /\/api\/v1\/chats\/?$/.test(response.url())
        );
        const searched = qa.waitForRagSearch();
        await qa.search(ROWAN_QUESTION);
        const search = await searched;
        expect(search.status).toBe(200);
        expect(search.requestBody).toMatchObject({
          search_mode: 'fts',
          fts_level: 'media',
          sources: ['media_db'],
          enable_generation: true,
          enable_citations: true,
          enable_web_fallback: false,
          generation_model: model,
          generation_provider: provider,
        });
        expect(search.requestBody.include_media_ids.map(String).sort()).toEqual(
          [source.id, distractor.id].sort()
        );
        const response = await created;
        expect(response.ok()).toBe(true);
        qaId = String((await response.json()).id);
        expect(qaId).not.toMatch(/^(?:undefined|null|local[-_])/);
        evidence.qaConversationId = qaId;
        evidence.ragRequest = search.requestBody;
        await qa.waitForResults(90_000);
        qaAnswer = await qa.getAnswerText();
        assertRowanFacts(qaAnswer);
        await expect
          .poll(
            async () => {
              qaMessages = readSavedMessages(
                await apiGet(
                  `/api/v1/chat/conversations/${qaId}/messages-with-context?include_rag_context=true`
                ),
                qaId
              );
              return (
                qaMessages.find((message) => message.role === 'assistant')?.rag_context
                  ?.retrieved_documents?.length ?? 0
              );
            },
            { timeout: 30_000 }
          )
          .toBeGreaterThan(0);
        const assistant = qaMessages.find((message) => message.role === 'assistant')!;
        assertCitedRowanAnswer(
          assistant.content,
          assistant.rag_context!.retrieved_documents,
          source.id,
          distractor.id
        );
        expect(assistant.rag_context!.trust_state).toBe('cited_answer');
        assertSavedTurn(qaMessages, ROWAN_QUESTION, assistant.content);
        expect(assistant.rag_context!.generated_answer).toBe(assistant.content);
        evidence.qaMessages = qaMessages;
      });
      await test.step('Inspect every citation and its canonical Media target', async () => {
        const indices = [
          ...new Set(
            await qa
              .getCitationButtons()
              .evaluateAll((buttons) =>
                buttons.map((button) =>
                  Number(button.getAttribute('data-knowledge-citation-index'))
                )
              )
          ),
        ];
        expect(indices.length).toBeGreaterThan(0);
        for (const index of indices) {
          await qa
            .getCitationButtons()
            .filter({ hasText: `[${index}]` })
            .first()
            .click();
          const card = page.locator(`#source-card-${index - 1}`);
          await expect(card).toHaveAttribute('data-source-id', source.id);
          await card.getByRole('button', { name: `View source ${index}`, exact: true }).click();
          const preview = page.getByRole('dialog', { name: new RegExp(`^Source ${index}:`) });
          await expect(preview).toContainText(`Source ID ${source.id}`);
          assertRowanFacts(await preview.locator('pre').innerText());
          const [mediaPage] = await Promise.all([
            page.context().waitForEvent('page'),
            preview.getByRole('button', { name: 'Open in Media', exact: true }).click(),
          ]);
          try {
            await expect(mediaPage).toHaveURL(new RegExp(`/media\\?id=${source.id}$`));
            await expect(mediaPage.getByText(source.title, { exact: true }).first()).toBeVisible();
          } finally {
            await mediaPage.close();
          }
          await preview.getByRole('button', { name: 'Close source preview' }).click();
        }
      });
      await test.step('Reload saved QA with its original citations', async () => {
        await page.goto(`/knowledge/thread/${encodeURIComponent(qaId)}`, {
          waitUntil: 'domcontentloaded',
        });
        await waitForConnection(page);
        await qa.waitForReady();
        await qa.waitForResults();
        expect(await qa.getAnswerText()).toBe(qaAnswer);
        expect(
          readSavedMessages(
            await apiGet(
              `/api/v1/chat/conversations/${qaId}/messages-with-context?include_rag_context=true`
            ),
            qaId
          )
        ).toEqual(qaMessages);
      });
      await test.step('Hand off the full loaded source and save a grounded Chat turn', async () => {
        await page.goto(`/media?id=${source.id}`, { waitUntil: 'domcontentloaded' });
        await waitForConnection(page);
        const handoff = page.getByRole('button', { name: 'Chat with this media', exact: true });
        await expect(handoff).toBeEnabled();
        await handoff.click();
        await expect(page).toHaveURL(/\/chat(?:\?|$)/);
        const chat = new ChatPage(page);
        await chat.waitForReady();
        const input = await chat.getChatInput();
        await expect(input).toHaveValue(/ORBIT-742/);
        const draft = await input.inputValue();
        assertFullSourceHandoff(draft, source.text);
        expect(draft).toContain(source.title);
        await chat.selectModel(model!);
        const prompt = `${draft}\n\nUsing only this source, answer: ${ROWAN_QUESTION}`;
        const created = page.waitForResponse(
          (response) =>
            response.request().method() === 'POST' && /\/api\/v1\/chats\/?$/.test(response.url())
        );
        const completionRequest = page.waitForRequest(
          (request) =>
            request.method() === 'POST' &&
            /\/api\/v1\/chat\/completions(?:\?|$)/.test(request.url())
        );
        // Primary Chat persistence returns save receipts; a separate fallback
        // message POST is intentionally absent when the server saved both rows.
        const [[response, completion]] = await Promise.all([
          Promise.all([created, completionRequest]),
          chat.sendMessage(prompt),
        ]);
        const sent = completion.postDataJSON();
        expect(sent.model).toBe(model);
        expect(sent.provider ?? sent.api_provider).toBe(provider);
        expect(sent.save_to_db).toBe(true);
        expect(
          sent.messages
            .filter((message: { role: string }) => message.role === 'user')
            .map((message: { role: string; content: unknown }) => ({
              role: message.role,
              content: message.content,
            }))
        ).toEqual([{ role: 'user', content: prompt }]);
        evidence.chatProvider = { model: sent.model, provider: sent.provider ?? sent.api_provider };
        expect(response.ok()).toBe(true);
        const chatId = String((await response.json()).id);
        expect(chatId).not.toMatch(/^(?:undefined|null|local[-_])/);
        expect(sent.conversation_id).toBe(chatId);
        evidence.chatConversationId = chatId;
        await waitForStreamComplete(page, 90_000);
        await chat.waitForResponse();
        const visibleAnswer = (await chat.getMessages())
          .filter((message) => message.role === 'assistant')
          .at(-1)!.content;
        assertRowanFacts(visibleAnswer);
        let saved: SavedMessage[] = [];
        await expect
          .poll(
            async () => {
              saved = readSavedMessages(
                (await apiGet(`/api/v1/chats/${chatId}/messages?render_placeholders=false`))
                  .messages,
                chatId
              );
              return saved.filter((message) => message.role === 'assistant').length;
            },
            { timeout: 30_000 }
          )
          .toBe(1);
        const savedAnswer = saved.find((message) => message.role === 'assistant')!.content;
        assertRowanFacts(savedAnswer);
        assertSavedTurn(saved, prompt, savedAnswer);
        evidence.chatMessages = saved;
        await page.goto(`/chat?settingsServerChatId=${encodeURIComponent(chatId)}`, {
          waitUntil: 'domcontentloaded',
        });
        await waitForConnection(page);
        await chat.waitForReady();
        await expect
          .poll(
            async () =>
              (await chat.getMessages()).filter((message) => message.role === 'assistant').at(-1)
                ?.content
          )
          .toBe(visibleAnswer);
        const reloaded = readSavedMessages(
          (await apiGet(`/api/v1/chats/${chatId}/messages?render_placeholders=false`)).messages,
          chatId
        );
        expect(reloaded).toEqual(saved);
        assertFullSourceHandoff(
          reloaded.find((message: SavedMessage) => message.role === 'user').content,
          source.text
        );
      });
      await test.step('Ask an absent fact without inventing a price', async () => {
        await page.goto(`/knowledge/thread/${encodeURIComponent(qaId)}`, {
          waitUntil: 'domcontentloaded',
        });
        await waitForConnection(page);
        await qa.waitForReady();
        await qa.waitForResults();
        const searched = qa.waitForRagSearch();
        await qa.askFollowUp(PRICE_QUESTION);
        const followUp = await searched;
        expect(followUp.status).toBe(200);
        expect(followUp.requestBody).toMatchObject({
          generation_provider: provider!,
          generation_model: model!,
        });
        await qa.waitForResults(90_000);
        assertPriceAbstention(await qa.getAnswerText());
        evidence.priceAnswer = await qa.getAnswerText();
      });
    } finally {
      await testInfo.attach('uat390-linked-evidence.json', {
        body: JSON.stringify(evidence, null, 2),
        contentType: 'application/json',
      });
    }
  });
});
