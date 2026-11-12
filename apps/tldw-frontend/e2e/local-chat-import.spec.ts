import { randomUUID } from 'node:crypto';
import { test } from '@playwright/test';
import runLocalChatImport from './helpers/local-chat-import.mjs';

test('restores local chat data through its enclosing atomic transaction', async ({ page }) => {
  const historyId = randomUUID();
  const createdAt = Date.now();
  const input = {
    chat: [
      {
        history: {
          id: historyId,
          title: 'Local import transaction regression',
          is_rag: false,
          createdAt,
          message_source: 'web-ui',
        },
        messages: [
          {
            id: randomUUID(),
            history_id: historyId,
            name: 'You',
            role: 'user',
            content: 'Local import input; no answer generated.',
            createdAt,
            parent_message_id: null,
          },
        ],
      },
    ],
  };
  await page.goto('/settings/data');
  await runLocalChatImport(page, {
    name: 'local-chat-import-regression.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(input)),
  });
});
