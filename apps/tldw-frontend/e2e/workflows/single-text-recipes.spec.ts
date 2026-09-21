import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Locator, type Page, type Route } from '@playwright/test';

import { expectNoHorizontalOverflow, seedAuth } from '../utils/helpers';

const API_ORIGIN = 'http://127.0.0.1:18093';
const PROJECT_ID = 42;
const RUNTIME_SENTINEL = 'RUNTIME_ONLY_DO_NOT_PERSIST';
const ORIGINAL_USER_DRAFT = 'Unsent composer draft, byte for byte.';
const ORIGINAL_SYSTEM_DRAFT = 'Original selected system prompt.';

const LIMITS = {
  max_request_bytes: 64_000,
  max_draft_chars: 24_000,
  max_candidate_chars: 24_000,
  max_raw_output_chars: 32_000,
  max_findings: 5,
  max_finding_text_chars: 500,
  max_provider_chars: 100,
  max_model_chars: 500,
  max_meta_prompt_version_chars: 100,
  max_warning_chars: 100,
  max_warnings: 16,
  max_protected_tokens: 64,
  max_protected_token_kind_chars: 50,
  max_protected_token_chars: 500,
  max_protected_token_occurrences: 100,
  max_protected_token_total_chars: 4_000,
};

type CapabilityMode = 'supported' | 'old' | 'unknown';
type RequestBody = Record<string, unknown>;

type RecipeApi = {
  creates: RequestBody[];
  updates: RequestBody[];
};

const json = (route: Route, status: number, body: unknown) =>
  route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  });

const serverPrompt = (body: RequestBody, id: number) => ({
  id,
  project_id: PROJECT_ID,
  uuid: `recipe-e2e-${id}`,
  name: String(body.name || 'Untitled recipe'),
  system_prompt: String(body.system_prompt || ''),
  user_prompt: String(body.user_prompt || ''),
  prompt_format: body.prompt_format,
  prompt_schema_version: body.prompt_schema_version,
  prompt_definition: body.prompt_definition,
  few_shot_examples: null,
  modules_config: null,
  version_number: 1,
  change_description: 'recipe E2E',
  parent_version_id: null,
  updated_at: '2026-09-11T12:00:00Z',
});

async function installRecipeApi(page: Page, mode: CapabilityMode): Promise<RecipeApi> {
  const creates: RequestBody[] = [];
  const updates: RequestBody[] = [];
  let nextId = 700;

  await page.unroute('**/api/v1/**');
  await page.route('**/api/v1/**', async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() === 'OPTIONS') return route.fulfill({ status: 204 });
    if (path === '/api/v1/health') return json(route, 200, { status: 'ok' });
    if (path === '/api/v1/llm/models/metadata') {
      return json(route, 200, { models: [] });
    }
    if (path === '/api/v1/llm/models') return json(route, 200, []);
    if (path === '/api/v1/llm/providers') {
      return json(route, 200, { providers: [] });
    }
    if (path === '/api/v1/config/docs-info') return json(route, 200, {});
    if (path === '/api/v1/notifications/unread-count') {
      return json(route, 200, { count: 0 });
    }
    if (path === '/api/v1/users/me/profile') {
      return json(route, 200, { preferences: {} });
    }
    if (path === '/api/v1/prompts/capabilities') {
      if (mode === 'unknown') return json(route, 404, { detail: 'not found' });
      return json(route, 200, {
        prompt_improvement_v1: { supported: false, limits: LIMITS },
        single_text_recipe_v2: { supported: mode === 'supported' },
        prompt_persistence: {
          create_authorized: mode === 'supported',
          update_authorized: mode === 'supported',
        },
      });
    }
    if (path === '/api/v1/prompt-studio/prompts/create' && request.method() === 'POST') {
      const body = request.postDataJSON() as RequestBody;
      creates.push(body);
      return json(route, 200, {
        success: true,
        data: serverPrompt(body, ++nextId),
      });
    }
    if (path.startsWith('/api/v1/prompt-studio/prompts/update/') && request.method() === 'PUT') {
      const body = request.postDataJSON() as RequestBody;
      updates.push(body);
      const id = Number(path.split('/').at(-1));
      return json(route, 200, {
        success: true,
        data: serverPrompt(body, id),
      });
    }
    if (path === '/api/v1/prompts/search') {
      return json(route, 200, {
        items: [],
        total_matches: 0,
        page: 1,
        per_page: 20,
      });
    }
    if (path.startsWith('/api/v1/characters')) return json(route, 200, []);
    if (path.includes('/collections')) return json(route, 200, { collections: [] });
    return json(route, 404, { detail: 'not found' });
  });
  return { creates, updates };
}

async function prepareChat(page: Page, mode: CapabilityMode = 'supported') {
  await seedAuth(page, { serverUrl: API_ORIGIN, allowOffline: true });
  await page.addInitScript(
    ({ projectId }) => {
      localStorage.removeItem('tldw:nextgenComposerEnabled');
      localStorage.removeItem('tldw:composerVariant');
      localStorage.setItem('playgroundComposerOptionsExpanded', 'true');
      localStorage.setItem(
        'promptStudioDefaults',
        JSON.stringify({
          defaultProjectId: projectId,
          autoSyncWorkspacePrompts: true,
        })
      );
    },
    { projectId: PROJECT_ID }
  );
  const api = await installRecipeApi(page, mode);
  await page.goto('/chat', { waitUntil: 'domcontentloaded' });
  await expect(page.getByTestId('chat-input').first()).toBeVisible({
    timeout: 30_000,
  });
  return api;
}

async function openComposerRecipe(page: Page) {
  const trigger = page.getByRole('button', { name: 'Improve prompt' }).first();
  await trigger.click();
  const actions = page.getByRole('group', { name: 'Prompt improvement actions' }).first();
  await expect(actions).toBeVisible();
  await actions.getByRole('button', { name: /Build from recipe/ }).click();
  const builder = page.getByRole('region', {
    name: 'Structured recipe builder',
  });
  await expect(builder).toBeVisible();
  return builder;
}

async function fillCurrentValue(builder: Locator, label: string, value: string) {
  await builder
    .getByRole('textbox', {
      name: `Current value for ${label} (not saved)`,
    })
    .fill(value);
}

async function waitForRecipeDrawerContained(page: Page) {
  const dialog = page.getByRole('dialog', { name: 'Build from recipe' });
  await expect
    .poll(() =>
      dialog.evaluate((element) => {
        const rect = element.getBoundingClientRect();
        return Math.max(0, -rect.left, rect.right - window.innerWidth);
      })
    )
    .toBeLessThanOrEqual(1);
}

const clearTaskDefinition = (schemaVersion = 2) => ({
  schema_version: schemaVersion,
  format: 'structured',
  definition_kind: 'single_text_recipe',
  assembly_config: {
    assembly_mode: 'single_text',
    target_role: 'system',
    render_format: 'xml',
    block_separator: '\n\n',
  },
  variables: [
    {
      name: 'task',
      label: 'Task',
      description: 'The task to complete.',
      required: true,
      default_value: null,
      input_type: 'textarea',
      options: null,
      max_length: null,
    },
  ],
  blocks: [
    {
      id: 'objective',
      name: 'Objective',
      section_key: 'objective',
      role: 'system',
      kind: 'objective',
      content: 'Complete this task:\n\n{{task}}',
      enabled: true,
      order: 10,
      is_template: true,
    },
  ],
});

async function seedPromptRecords(page: Page) {
  await page.evaluate(
    ({ known, future }) =>
      new Promise<void>((resolve, reject) => {
        const open = indexedDB.open('PageAssistDatabase');
        open.onerror = () => reject(open.error);
        open.onsuccess = () => {
          const database = open.result;
          const transaction = database.transaction('prompts', 'readwrite');
          const store = transaction.objectStore('prompts');
          store.put(known);
          store.put(future);
          transaction.oncomplete = () => {
            database.close();
            resolve();
          };
          transaction.onerror = () => reject(transaction.error);
        };
      }),
    {
      known: {
        id: 'known-v2-recipe',
        title: 'E2E searchable system recipe',
        name: 'E2E searchable system recipe',
        content: 'Known recipe preview sentinel',
        is_system: true,
        system_prompt: 'Known recipe preview sentinel',
        user_prompt: '',
        promptFormat: 'structured',
        promptSchemaVersion: 2,
        structuredPromptDefinition: clearTaskDefinition(),
        keywords: ['recipe-e2e'],
        syncStatus: 'local',
        sourceSystem: 'workspace',
        createdAt: Date.now(),
        updatedAt: Date.now(),
        deletedAt: null,
      },
      future: {
        id: 'future-v3-recipe',
        title: 'FUTURE_V3_MUST_STAY_QUARANTINED',
        name: 'FUTURE_V3_MUST_STAY_QUARANTINED',
        content: 'FUTURE_V3_PAYLOAD_SENTINEL',
        is_system: true,
        promptFormat: 'structured',
        promptSchemaVersion: 3,
        structuredPromptDefinition: clearTaskDefinition(3),
        syncStatus: 'local',
        sourceSystem: 'workspace',
        createdAt: Date.now(),
        updatedAt: Date.now(),
        deletedAt: null,
      },
    }
  );
}

async function readPromptRecord(page: Page, id: string) {
  return page.evaluate(
    (promptId) =>
      new Promise<unknown>((resolve, reject) => {
        const open = indexedDB.open('PageAssistDatabase');
        open.onerror = () => reject(open.error);
        open.onsuccess = () => {
          const database = open.result;
          const transaction = database.transaction('prompts', 'readonly');
          const request = transaction.objectStore('prompts').get(promptId);
          request.onsuccess = () => {
            database.close();
            resolve(request.result);
          };
          request.onerror = () => reject(request.error);
        };
      }),
    id
  );
}

function expectNoRuntimePayload(body: RequestBody) {
  const serialized = JSON.stringify(body);
  expect(serialized).not.toContain(RUNTIME_SENTINEL);
  expect(serialized).not.toMatch(/runtimeValues|runtime_values|variable_values|resolved_values/);
  expect(body.prompt_schema_version).toBe(2);
  expect(body.prompt_format).toBe('structured');
}

test.describe('WebUI single-text structured recipes', () => {
  test('covers all starters and compiles exact edit, reorder, toggle, variable, and format output', async ({
    page,
  }) => {
    await prepareChat(page);
    const builder = await openComposerRecipe(page);
    const source = builder.getByRole('combobox', { name: 'Recipe source' });

    for (const [name, blockCount, variableLabel] of [
      ['Clear task', 4, 'Task'],
      ['Research and analysis', 5, 'Research question'],
      ['Agent workflow', 5, 'Objective'],
    ] as const) {
      await source.selectOption({ label: name });
      await expect(
        builder.getByTestId('structured-block-list').locator(':scope > div')
      ).toHaveCount(blockCount);
      await fillCurrentValue(builder, variableLabel, `${name} input`);
      await expect(builder.getByRole('textbox', { name: 'Compiled prompt preview' })).toContainText(
        `${name} input`
      );
    }
    await source.selectOption({ label: 'Blank' });
    await expect(builder.getByText('No variables yet.')).toBeVisible();
    await expect(builder.getByTestId('structured-block-list').locator(':scope > div')).toHaveCount(
      0
    );

    await source.selectOption({ label: 'Clear task' });
    await fillCurrentValue(builder, 'Task', 'Exact output task');
    await builder.getByRole('button', { name: 'Edit Context / inputs block' }).click();
    await builder.getByRole('checkbox', { name: 'Block enabled' }).uncheck();
    await builder.getByRole('button', { name: 'Edit Constraints block' }).click();
    await builder.getByRole('checkbox', { name: 'Block enabled' }).uncheck();
    await builder.getByRole('button', { name: 'Edit Output block' }).click();
    await builder.getByRole('checkbox', { name: 'Block enabled' }).uncheck();

    const preview = builder.getByRole('textbox', {
      name: 'Compiled prompt preview',
    });
    await expect(preview).toHaveValue(
      '<objective>Complete this task:\n\nExact output task</objective>'
    );
    await builder.getByRole('combobox', { name: 'Output format' }).selectOption('markdown');
    await expect(preview).toHaveValue('## Objective\n\nComplete this task:\n\nExact output task');
    await builder.getByRole('combobox', { name: 'Output format' }).selectOption('freeform');
    await expect(preview).toHaveValue('Complete this task:\n\nExact output task');

    await builder.getByRole('button', { name: 'Edit Output block' }).click();
    await builder.getByRole('checkbox', { name: 'Block enabled' }).check();
    await builder.getByRole('textbox', { name: 'Block content' }).fill('Final answer only.');
    await builder.getByRole('button', { name: 'Edit Constraints block' }).click();
    await builder.getByRole('checkbox', { name: 'Block enabled' }).check();
    await expect(preview).toHaveValue(
      'Complete this task:\n\nExact output task\n\nFollow every explicit constraint. Preserve supplied names, facts, code, and required formatting; do not invent requirements.\n\nFinal answer only.'
    );
    await builder.getByRole('button', { name: 'Move Output up' }).press('Enter');
    await expect(preview).toHaveValue(
      'Complete this task:\n\nExact output task\n\nFinal answer only.\n\nFollow every explicit constraint. Preserve supplied names, facts, code, and required formatting; do not invent requirements.'
    );
  });

  test('saves, reopens, clones, updates, applies to the user draft, and undoes exactly', async ({
    page,
  }) => {
    const api = await prepareChat(page);
    const composer = page.getByTestId('chat-input').first();
    await composer.fill(ORIGINAL_USER_DRAFT);
    const builder = await openComposerRecipe(page);
    await fillCurrentValue(builder, 'Task', RUNTIME_SENTINEL);
    await builder
      .getByRole('checkbox', {
        name: 'Use a saved starter default for Task',
      })
      .check();
    await builder
      .getByRole('textbox', { name: 'Starter default for Task (saved)' })
      .fill('Saved starter default');

    const exactApplied = await builder
      .getByRole('textbox', { name: 'Compiled prompt preview' })
      .inputValue();
    const save = builder.getByRole('button', { name: 'Save as new recipe' });
    await expect(save).toBeEnabled();
    await save.click();
    await expect.poll(() => api.creates.length).toBe(1);
    expectNoRuntimePayload(api.creates[0]);
    expect(JSON.stringify(api.creates[0])).toContain('Saved starter default');

    const source = builder.getByRole('combobox', { name: 'Recipe source' });
    await expect(source.locator('optgroup[label="Saved recipes"] option')).toHaveCount(1);
    await source.selectOption({ label: 'Untitled recipe' });
    await fillCurrentValue(builder, 'Task', 'Reopened runtime');
    await builder.getByRole('button', { name: 'Edit Objective block' }).click();
    await builder
      .getByRole('textbox', { name: 'Block content' })
      .fill('Updated objective: {{task}}');
    await builder.getByRole('button', { name: 'Update recipe' }).click();
    await expect.poll(() => api.updates.length).toBe(1);
    expectNoRuntimePayload(api.updates[0]);

    await builder.getByRole('button', { name: 'Save as new recipe' }).click();
    await expect.poll(() => api.creates.length).toBe(2);
    expectNoRuntimePayload(api.creates[1]);

    await source.selectOption({ label: 'Clear task' });
    await fillCurrentValue(builder, 'Task', RUNTIME_SENTINEL);
    await builder.getByRole('button', { name: 'Apply to user message' }).click();
    await expect(composer).toHaveValue(exactApplied);
    await page.getByRole('button', { name: 'Undo recipe' }).click();
    await expect(composer).toHaveValue(ORIGINAL_USER_DRAFT);
  });

  test('applies to the current system draft without changing template identity and supports exact Undo', async ({
    page,
  }) => {
    await prepareChat(page);
    await page.evaluate(
      ({ title, content }) =>
        new Promise<void>((resolve, reject) => {
          const open = indexedDB.open('PageAssistDatabase');
          open.onerror = () => reject(open.error);
          open.onsuccess = () => {
            const database = open.result;
            const transaction = database.transaction('prompts', 'readwrite');
            transaction.objectStore('prompts').put({
              id: 'recipe-system-template',
              title,
              name: title,
              content,
              is_system: true,
              createdAt: Date.now(),
              updatedAt: Date.now(),
              deletedAt: null,
            });
            transaction.oncomplete = () => {
              database.close();
              resolve();
            };
            transaction.onerror = () => reject(transaction.error);
          };
        }),
      { title: 'Recipe E2E System Template', content: ORIGINAL_SYSTEM_DRAFT }
    );
    await page.evaluate(() =>
      localStorage.setItem('selectedSystemPrompt', JSON.stringify('recipe-system-template'))
    );
    await page.reload({ waitUntil: 'domcontentloaded' });

    const promptTrigger = page.getByTestId('chat-prompt-select').first();
    await expect(promptTrigger).toContainText('Recipe E2E System Template');
    await promptTrigger.click();
    await page
      .getByRole('menuitem', { name: /Edit system prompt/i })
      .last()
      .click();
    const editor = page.getByPlaceholder('Enter system prompt');
    await expect(editor).toHaveValue(ORIGINAL_SYSTEM_DRAFT);
    const dialog = page.getByRole('dialog', { name: 'Edit system prompt' });
    await dialog.getByRole('button', { name: 'Improve prompt' }).click();
    await page.getByRole('button', { name: /Build from recipe/ }).click();
    const builder = page.getByRole('region', {
      name: 'Structured recipe builder',
    });
    await fillCurrentValue(builder, 'Task', 'System recipe task');
    const compiled = await builder
      .getByRole('textbox', { name: 'Compiled prompt preview' })
      .inputValue();
    await builder.getByRole('button', { name: 'Apply to system prompt' }).click();
    await expect(editor).toHaveValue(compiled);
    await expect(promptTrigger).toContainText('Recipe E2E System Template');
    await page.getByRole('button', { name: 'Undo recipe' }).click();
    await expect(editor).toHaveValue(ORIGINAL_SYSTEM_DRAFT);
    await expect(promptTrigger).toContainText('Recipe E2E System Template');
  });

  test('groups, badges, searches, and safely quarantines an unknown v3 record', async ({
    page,
  }) => {
    await prepareChat(page);
    await seedPromptRecords(page);
    const before = await readPromptRecord(page, 'future-v3-recipe');
    await page.goto('/prompts', { waitUntil: 'domcontentloaded' });
    await expect(page.getByTestId('prompts-custom')).toBeVisible({
      timeout: 30_000,
    });
    const search = page.getByTestId('prompts-search');
    await search.fill('searchable system recipe');
    const row = page.getByTestId('prompt-row-known-v2-recipe');
    await expect(row).toBeVisible();
    await expect(row.getByText('Recipe', { exact: true })).toBeVisible();
    await page.getByTestId('facet-type-recipe_system').click();
    await expect(row).toBeVisible();
    await expect(page.getByText('FUTURE_V3_MUST_STAY_QUARANTINED')).toHaveCount(0);
    await row.press('Enter');
    await expect(page.getByRole('region', { name: 'Structured recipe builder' })).toBeVisible();
    await expect(page.getByRole('combobox', { name: 'Recipe source' })).toHaveValue(
      'saved:known-v2-recipe'
    );
    await page.getByRole('button', { name: 'Back to Prompts' }).click();
    expect(await readPromptRecord(page, 'future-v3-recipe')).toEqual(before);
  });

  test('old and unknown servers disable persistence while local built-in apply remains available', async ({
    page,
  }) => {
    for (const mode of ['old', 'unknown'] as const) {
      await prepareChat(page, mode);
      const composer = page.getByTestId('chat-input').first();
      await composer.fill(`${mode} exact draft`);
      const builder = await openComposerRecipe(page);
      await fillCurrentValue(builder, 'Task', `${mode} local compile`);
      await expect(builder.getByRole('button', { name: 'Save as new recipe' })).toBeDisabled();
      await expect(builder).toContainText(
        mode === 'old'
          ? 'This server does not support recipe saving yet'
          : 'server capabilities could not be confirmed'
      );
      const compiled = await builder
        .getByRole('textbox', { name: 'Compiled prompt preview' })
        .inputValue();
      await builder.getByRole('button', { name: 'Apply to user message' }).click();
      await expect(composer).toHaveValue(compiled);
      await page.getByRole('button', { name: 'Undo recipe' }).click();
      await expect(composer).toHaveValue(`${mode} exact draft`);
    }
  });

  test('offline mobile recipe editing is keyboard reachable, local-only, responsive, and axe-clean in both themes', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 780 });
    await prepareChat(page);
    const visitedThemes = new Set<string>();
    for (let index = 0; index < 2; index += 1) {
      const trigger = page.getByRole('button', { name: 'Improve prompt' }).first();
      await trigger.focus();
      await trigger.press('Enter');
      const action = page.getByRole('button', { name: /Build from recipe/ });
      await action.focus();
      await action.press('Enter');
      const builder = page.getByRole('region', {
        name: 'Structured recipe builder',
      });
      await expect(builder).toBeVisible();
      await page.evaluate(() => {
        const store = (
          window as unknown as {
            __tldw_useConnectionStore?: {
              getState: () => { state: Record<string, unknown> };
              setState: (value: { state: Record<string, unknown> }) => void;
            };
          }
        ).__tldw_useConnectionStore;
        if (!store) throw new Error('Connection store is unavailable');
        store.setState({
          state: {
            ...store.getState().state,
            phase: 'error',
            isConnected: false,
            isChecking: false,
            offlineBypass: false,
            errorKind: 'unreachable',
            knowledgeStatus: 'offline',
          },
        });
      });
      await expect(builder).toContainText('Recipe saving is unavailable offline');
      await fillCurrentValue(builder, 'Task', 'Offline mobile task');
      await expect(builder.getByRole('button', { name: 'Save as new recipe' })).toBeDisabled();
      await expect(builder.getByRole('button', { name: 'Apply to user message' })).toBeEnabled();
      await waitForRecipeDrawerContained(page);
      await expectNoHorizontalOverflow(page, 'mobile structured recipe builder');
      const theme = await page.evaluate(() =>
        document.documentElement.classList.contains('dark') ? 'dark' : 'light'
      );
      expect(visitedThemes.has(theme), `theme ${theme} was already checked`).toBe(false);
      visitedThemes.add(theme);
      const results = await new AxeBuilder({ page })
        .include('[data-testid="single-field-recipe-editor"]')
        .analyze();
      expect(results.violations, `${theme} recipe accessibility violations`).toEqual([]);
      await page.keyboard.press('Escape');
      await expect(builder).not.toBeVisible();
      await expect(trigger).toBeFocused();
      if (index === 0) {
        const nextTheme = theme === 'dark' ? 'light' : 'dark';
        await page.evaluate((value) => localStorage.setItem('theme', value), nextTheme);
        await page.reload({ waitUntil: 'domcontentloaded' });
        await expect(page.getByTestId('chat-input').first()).toBeVisible();
        await expect
          .poll(() =>
            page.evaluate(() =>
              document.documentElement.classList.contains('dark') ? 'dark' : 'light'
            )
          )
          .toBe(nextTheme);
      }
    }
    expect(visitedThemes).toEqual(new Set(['dark', 'light']));
  });
});
