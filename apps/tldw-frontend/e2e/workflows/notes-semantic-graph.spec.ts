import type { Page, Route } from '@playwright/test';

import { expect, test } from '../utils/fixtures';

const NOW = '2026-08-29T12:00:00Z';
const SOURCE_ID = 'semantic-source';
const TARGET_ID = 'semantic-target';
const GENERATION_ID = 'semantic-generation-a';
const CAPABILITY_REVISION = `sha256:${'a'.repeat(64)}`;
const SOURCE_EXCERPT = 'Source matched passage for semantic inspection.';
const TARGET_EXCERPT = 'Target matched passage for semantic inspection.';

const sourceNote = {
  id: SOURCE_ID,
  title: 'Semantic source note',
  content: 'Source passage about durable evidence and retrieval boundaries.',
  version: 4,
  keywords: [],
  created_at: NOW,
  updated_at: NOW,
};

const targetNote = {
  ...sourceNote,
  id: TARGET_ID,
  title: 'Semantic target note with a long relationship title',
  content: 'Target passage about durable evidence and retrieval boundaries.',
  version: 7,
};

type SemanticState = 'off' | 'preparing' | 'ready' | 'updating';

class SemanticGraphFixture {
  readonly calls: string[] = [];
  readonly graphUrls: string[] = [];
  state: SemanticState;
  converted = false;
  completeActiveRun = false;
  failSemanticGraph = false;
  activeRunId: string | null = null;
  activeRunMode = 'build';
  manualLinkBody: unknown = null;

  constructor(initialState: SemanticState = 'off') {
    this.state = initialState;
  }

  private capabilities() {
    return {
      active_note_count: 2,
      estimated_chunk_count: 6,
      estimated_run_count: 1,
      provider_label: 'Local deterministic embedding provider',
      model: 'semantic-e2e-model-with-a-long-revision-label',
      endpoint_display: 'http://127.0.0.1:8099',
      execution_boundary: 'local',
      storage_boundary: 'local',
      storage_label: 'Local vector store',
      outbound_data_categories: ['note_content_chunks', 'note_title'],
      capability_revision: CAPABILITY_REVISION,
      indexing_available: true,
      unavailable_reason: null,
      metric: 'cosine',
      resolved_dimensions: 384,
      dimension_probe_required: false,
      renewal_requires_delete: false,
      manage_authorized: true,
    };
  }

  private run(status = 'processing') {
    const terminal = ['completed', 'cancelled'].includes(status);
    return {
      run_id: this.activeRunId ?? 'semantic-run-a',
      mode: this.activeRunMode,
      status,
      revision: terminal ? 2 : 1,
      indexed_notes: status === 'completed' ? 2 : 1,
      excluded_notes: 0,
      failed_notes: 0,
      pending_notes: status === 'completed' ? 0 : 1,
      published_chunks: status === 'completed' ? 6 : 3,
      cleanup_complete: status === 'completed' && this.activeRunMode === 'delete',
      error_code: null,
      link: `/api/v1/notes/graph/semantic-index/runs/${this.activeRunId ?? 'semantic-run-a'}`,
    };
  }

  private status() {
    const active = this.activeRunId ? this.run() : null;
    if (this.state === 'off') {
      return {
        state: 'off',
        detail_reason: null,
        desired_state: 'disabled',
        configuration_revision: 0,
        semantic_index_revision: 0,
        active_generation_id: null,
        active_generation_usable: false,
        indexed_notes: 0,
        excluded_notes: 0,
        failed_notes: 0,
        pending_notes: 0,
        published_chunks: 0,
        cleanup_pending: false,
        active_run: null,
      };
    }
    return {
      state: this.state,
      detail_reason: this.state === 'preparing' || this.state === 'updating' ? 'building' : null,
      desired_state: 'enabled',
      configuration_revision: 1,
      semantic_index_revision: this.state === 'preparing' ? 0 : 4,
      active_generation_id: this.state === 'preparing' ? null : GENERATION_ID,
      active_generation_usable: this.state !== 'preparing',
      indexed_notes: active ? 1 : 2,
      excluded_notes: 0,
      failed_notes: 0,
      pending_notes: active ? 1 : 0,
      published_chunks: active ? 3 : 6,
      cleanup_pending: false,
      active_run: active,
    };
  }

  private semanticGraphStatus() {
    return {
      available: true,
      state: 'ready',
      detail_reason: null,
      generation_id: GENERATION_ID,
      semantic_index_revision: 4,
      configuration_revision: 1,
      active_notes: 2,
      indexed_notes: 2,
      dirty_notes: 0,
      excluded_notes: 0,
      failed_notes: 0,
      effective_top_k: 10,
      effective_threshold: 0.75,
      max_top_k: 50,
      max_admission_nodes: 50,
      max_admission_edges: 50,
      max_evidence_pairs: 3,
      max_excerpt_code_points: 480,
      max_edge_evidence_code_points: 2880,
      max_response_evidence_bytes: 262144,
      truncated_by: [],
    };
  }

  private graph(includeSemantic: boolean) {
    const semanticEdge = {
      id: 'semantic:source:target',
      source: `note:${SOURCE_ID}`,
      target: `note:${TARGET_ID}`,
      type: 'semantic',
      directed: false,
      weight: 0.8765,
      label: null,
      evidence: {
        similarity: 0.8765,
        qualitative_band: 'high',
        source_note_id: `note:${SOURCE_ID}`,
        target_note_id: `note:${TARGET_ID}`,
        source_content_version: 4,
        target_content_version: 7,
        generation_id: GENERATION_ID,
        semantic_index_revision: 4,
        configuration_revision: 1,
        normalization_version: 'normalize-v1',
        chunker_version: 'chunk-v1',
        provider_label: 'Local deterministic embedding provider',
        model_label: 'semantic-e2e-model-with-a-long-revision-label',
        model_revision: 'model-r1',
        excerpt_pairs: [
          {
            source: {
              field: 'content',
              start_code_point: 0,
              end_code_point: SOURCE_EXCERPT.length,
              text: SOURCE_EXCERPT,
            },
            target: {
              field: 'content',
              start_code_point: 0,
              end_code_point: TARGET_EXCERPT.length,
              text: TARGET_EXCERPT,
            },
          },
        ],
      },
    };
    return {
      nodes: [
        {
          id: `note:${SOURCE_ID}`,
          type: 'note',
          label: sourceNote.title,
          created_at: NOW,
          deleted: false,
          degree: 1,
          tag_count: 0,
          primary_source_id: null,
        },
        {
          id: `note:${TARGET_ID}`,
          type: 'note',
          label: targetNote.title,
          created_at: NOW,
          deleted: false,
          degree: 1,
          tag_count: 0,
          primary_source_id: null,
        },
      ],
      edges: [
        ...(includeSemantic ? [semanticEdge] : []),
        ...(this.converted
          ? [
              {
                id: 'manual:source:target',
                source: `note:${SOURCE_ID}`,
                target: `note:${TARGET_ID}`,
                type: 'manual',
                directed: false,
                weight: 1,
                label: null,
              },
            ]
          : []),
      ],
      truncated: false,
      truncated_by: [],
      has_more: false,
      cursor: null,
      limits: { max_nodes: 120, max_edges: 480, max_degree: 40 },
      radius_cap_applied: false,
      active_note_count: 2,
      all_notes_note_cap: 100,
      all_notes_eligible: true,
      suggestions_authorized: false,
      manual_link_authorized: true,
      ...(includeSemantic ? { semantic_status: this.semanticGraphStatus() } : {}),
    };
  }

  private async fulfill(route: Route, body: unknown, status = 200) {
    await route.fulfill({
      status,
      contentType: 'application/json',
      headers: {
        'access-control-allow-origin': '*',
        'access-control-allow-headers': '*',
      },
      body: JSON.stringify(body),
    });
  }

  async handle(route: Route) {
    const request = route.request();
    const url = new URL(request.url());
    const method = request.method().toUpperCase();
    const requestPath = url.pathname;
    if (requestPath.startsWith('/api/v1/')) {
      this.calls.push(`${method} ${requestPath}`);
    }

    if (requestPath === '/api/v1/notes/graph/semantic-index/capabilities' && method === 'GET') {
      await this.fulfill(route, this.capabilities());
      return;
    }
    if (requestPath === '/api/v1/notes/graph/semantic-index' && method === 'GET') {
      await this.fulfill(route, this.status());
      return;
    }
    if (requestPath === '/api/v1/notes/graph/semantic-index' && method === 'PUT') {
      this.state = 'preparing';
      this.activeRunId = 'semantic-build-a';
      this.activeRunMode = 'build';
      this.completeActiveRun = false;
      await this.fulfill(route, {
        resource: this.status(),
        run: this.run(),
      });
      return;
    }
    if (requestPath === '/api/v1/notes/graph/semantic-index/runs' && method === 'POST') {
      this.state = 'updating';
      this.activeRunId = 'semantic-rebuild-a';
      this.activeRunMode = 'rebuild';
      this.completeActiveRun = false;
      await this.fulfill(route, this.run());
      return;
    }
    if (requestPath === '/api/v1/notes/graph/semantic-index' && method === 'DELETE') {
      this.state = 'off';
      this.activeRunId = 'semantic-delete-a';
      this.activeRunMode = 'delete';
      const mutation = {
        resource: this.status(),
        run: this.run('completed'),
      };
      this.activeRunId = null;
      await this.fulfill(route, mutation);
      return;
    }
    const cancelMatch = requestPath.match(
      /^\/api\/v1\/notes\/graph\/semantic-index\/runs\/([^/]+)\/cancel$/
    );
    if (cancelMatch && method === 'POST') {
      const cancelled = this.run('cancelled');
      this.state = 'ready';
      this.activeRunId = null;
      await this.fulfill(route, {
        resource: this.status(),
        run: cancelled,
      });
      return;
    }
    const runMatch = requestPath.match(/^\/api\/v1\/notes\/graph\/semantic-index\/runs\/([^/]+)$/);
    if (runMatch && method === 'GET') {
      if (this.completeActiveRun) {
        const completed = this.run('completed');
        this.state = 'ready';
        this.activeRunId = null;
        await this.fulfill(route, completed);
      } else {
        await this.fulfill(route, this.run());
      }
      return;
    }
    if (requestPath === '/api/v1/notes/graph' && method === 'GET') {
      this.graphUrls.push(url.toString());
      const includeSemantic =
        url.searchParams.get('edge_types')?.split(',').includes('semantic') ?? false;
      if (includeSemantic && this.failSemanticGraph) {
        await this.fulfill(
          route,
          {
            detail: {
              error_code: 'notes_semantic_provider_unavailable',
              message: 'semantic graph unavailable',
            },
          },
          503
        );
        return;
      }
      await this.fulfill(route, this.graph(includeSemantic));
      return;
    }
    if (requestPath === `/api/v1/notes/${SOURCE_ID}/links` && method === 'POST') {
      this.converted = true;
      this.manualLinkBody = request.postDataJSON();
      await this.fulfill(route, {
        status: 'created',
        edge: {
          edge_id: 'manual:source:target',
          from_note_id: SOURCE_ID,
          to_note_id: TARGET_ID,
        },
      });
      return;
    }
    if (
      requestPath.startsWith('/api/v1/notes/title-settings') ||
      requestPath.startsWith('/api/v1/admin/notes/title-settings')
    ) {
      await this.fulfill(route, {
        llm_enabled: false,
        default_strategy: 'heuristic',
      });
      return;
    }
    if (requestPath.startsWith('/api/v1/notes/search')) {
      await this.fulfill(route, { notes: [sourceNote, targetNote], total: 2 });
      return;
    }
    if (requestPath === '/api/v1/notes' || requestPath === '/api/v1/notes/') {
      await this.fulfill(route, {
        items: [sourceNote, targetNote],
        pagination: { total_items: 2 },
      });
      return;
    }
    if (requestPath === `/api/v1/notes/${SOURCE_ID}`) {
      await this.fulfill(route, { ...sourceNote, links: [] });
      return;
    }
    if (requestPath === `/api/v1/notes/${TARGET_ID}`) {
      await this.fulfill(route, { ...targetNote, links: [] });
      return;
    }
    if (requestPath === '/api/v1/auth/me') {
      await this.fulfill(route, {
        id: 1,
        username: 'semantic-e2e',
        role: 'user',
        is_active: true,
      });
      return;
    }
    if (requestPath.includes('/neighbors')) {
      await this.fulfill(route, { nodes: [], edges: [] });
      return;
    }
    const emptyResources: Array<[string, unknown]> = [
      ['/api/v1/notes/keywords', { keywords: [], total: 0 }],
      ['/api/v1/notes/collections', { collections: [], total: 0 }],
      ['/api/v1/notes/moodboards', { moodboards: [], total: 0 }],
      ['/api/v1/notes/trash', { notes: [], total: 0 }],
    ];
    const emptyResource = emptyResources.find(([prefix]) => requestPath.startsWith(prefix));
    if (emptyResource) {
      await this.fulfill(route, emptyResource[1]);
      return;
    }
    await route.continue();
  }
}

const openGraph = async (page: Page, fixture: SemanticGraphFixture) => {
  await page.route('**/api/v1/**', (route) => fixture.handle(route));
  await page.goto('/notes', { waitUntil: 'domcontentloaded' });
  const skipTour = page.getByText('Skip tour', { exact: true });
  if ((await skipTour.count()) > 0 && (await skipTour.isVisible())) {
    await skipTour.click();
  }
  const noteButton = page.getByRole('button', {
    name: 'Open note Semantic source note',
  });
  await expect
    .poll(
      async () =>
        JSON.stringify({
          visible: await noteButton.isVisible().catch(() => false),
          calls: fixture.calls,
        }),
      { timeout: 15_000 }
    )
    .toContain('"visible":true');
  await noteButton.click();
  await page.getByTestId('notes-view-mode-graph').click();
  await expect(page.getByTestId('notes-graph-workspace')).toBeVisible();
};

const confirm = async (page: Page, title: string, action: string) => {
  const dialog = page.getByRole('dialog', { name: title });
  await expect(dialog).toBeVisible();
  await dialog.getByRole('button', { name: action, exact: true }).click();
};

const closeMobileNotesList = async (page: Page) => {
  const backdrop = page.getByTestId('notes-mobile-sidebar-backdrop');
  await expect(page.getByTestId('notes-desktop-sidebar-toggle')).toHaveCount(0);
  if ((await backdrop.count()) > 0 && (await backdrop.isVisible())) {
    await backdrop.click();
  }
  const list = page.getByTestId('notes-list-region');
  await expect(list).toHaveClass(/-translate-x-full/);
  await expect
    .poll(async () => {
      const bounds = await list.boundingBox();
      return bounds ? bounds.x + bounds.width : 0;
    })
    .toBeLessThanOrEqual(1);
  await expect(backdrop).toHaveCount(0);
};

const hideNextDevPortal = async (page: Page) => {
  await page.addStyleTag({
    content: 'nextjs-portal { display: none !important; }',
  });
  await page.evaluate(() => {
    document.querySelectorAll('nextjs-portal').forEach((portal) => portal.remove());
  });
};

const measureSemanticContrast = async (page: Page, target: 'indicator' | 'evidenceText') =>
  page.evaluate((contrastTarget) => {
    const parseRgb = (value: string): [number, number, number] => {
      const channels = value
        .match(/[\d.]+/g)
        ?.slice(0, 3)
        .map(Number);
      if (!channels || channels.length !== 3) {
        throw new Error(`Unable to parse browser color: ${value}`);
      }
      return channels as [number, number, number];
    };
    const luminance = ([red, green, blue]: [number, number, number]) => {
      const linear = [red, green, blue].map((channel) => {
        const normalized = channel / 255;
        return normalized <= 0.04045 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4;
      });
      return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
    };
    const contrast = (foreground: string, background: string) => {
      const foregroundLuminance = luminance(parseRgb(foreground));
      const backgroundLuminance = luminance(parseRgb(background));
      return (
        (Math.max(foregroundLuminance, backgroundLuminance) + 0.05) /
        (Math.min(foregroundLuminance, backgroundLuminance) + 0.05)
      );
    };
    const effectiveBackground = (element: Element) => {
      let current: Element | null = element.parentElement;
      while (current) {
        const color = getComputedStyle(current).backgroundColor;
        if (color !== 'rgba(0, 0, 0, 0)' && color !== 'transparent') {
          return color;
        }
        current = current.parentElement;
      }
      return getComputedStyle(document.body).backgroundColor;
    };
    const element = document.querySelector(
      contrastTarget === 'indicator'
        ? '[data-testid="notes-graph-semantic-legend-swatch"]'
        : '[data-testid="notes-graph-semantic-treatment-label"]'
    );
    if (!element) {
      throw new Error(`Semantic ${contrastTarget} contrast target is missing.`);
    }
    const style = getComputedStyle(element);
    return contrast(
      contrastTarget === 'indicator' ? style.borderTopColor : style.color,
      effectiveBackground(element)
    );
  }, target);

test.describe('Notes semantic graph', () => {
  test('enables, inspects, converts, and keeps ordinary fallback available', async ({
    authedPage: page,
  }, testInfo) => {
    const fixture = new SemanticGraphFixture();
    await page.setViewportSize({ width: 1440, height: 1000 });
    await openGraph(page, fixture);

    await page.getByRole('tab', { name: 'Similar content' }).click();
    await expect(
      page.getByText('Local deterministic embedding provider', { exact: true })
    ).toBeVisible();
    await expect(page.getByText('Note body chunks', { exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Enable semantic index' }).click();
    await confirm(page, 'Enable semantic indexing?', 'Enable');
    await expect(page.getByText('Preparing', { exact: true })).toBeVisible();
    await expect(page.getByText('1 of 2 Notes indexed', { exact: true })).toBeVisible();

    fixture.completeActiveRun = true;
    await expect(page.getByText('Ready', { exact: true })).toBeVisible({
      timeout: 10_000,
    });

    const similarFilter = page.getByRole('checkbox', {
      name: 'Similar content',
      exact: true,
    });
    await expect(similarFilter).not.toBeChecked();
    await similarFilter.check();
    await expect.poll(() => fixture.graphUrls.at(-1) ?? '').toContain('semantic_top_k=10');
    const semanticUrl = new URL(fixture.graphUrls.at(-1) ?? 'http://invalid');
    expect(semanticUrl.searchParams.get('cursor')).toBeNull();
    expect(semanticUrl.searchParams.get('edge_types')).toContain('semantic');

    await page.getByRole('button', { name: 'Relationships', exact: true }).click();
    const relationships = page.getByTestId('notes-graph-relationships-view');
    const evidenceDisclosure = relationships.getByTestId('notes-graph-semantic-evidence-toggle');
    await expect(evidenceDisclosure).toContainText('Passage similarity: 0.8765');
    await expect(
      relationships.getByText('Source matched passage for semantic inspection.')
    ).not.toBeVisible();
    await evidenceDisclosure.click();
    await expect(
      relationships.getByText('Source matched passage for semantic inspection.')
    ).toBeVisible();
    await evidenceDisclosure.click();
    await relationships.getByRole('button', { name: 'Similar content', exact: true }).click();
    const inspector = page.getByTestId('notes-graph-inspector-region');
    await expect(
      inspector.getByText('Target matched passage for semantic inspection.')
    ).toBeVisible();
    await hideNextDevPortal(page);
    await page.screenshot({
      path: testInfo.outputPath('semantic-graph-desktop.png'),
      fullPage: false,
    });

    await page.setViewportSize({ width: 390, height: 844 });
    await closeMobileNotesList(page);
    const mobileGeometry = await page
      .getByTestId('notes-graph-workspace')
      .evaluate((workspace) => ({
        workspaceRight: workspace.getBoundingClientRect().right,
        viewportWidth: window.innerWidth,
        horizontalOverflow: document.documentElement.scrollWidth - window.innerWidth,
        nestedCards: workspace.querySelectorAll('[data-ui="card"] [data-ui="card"]').length,
      }));
    expect(mobileGeometry.workspaceRight).toBeLessThanOrEqual(mobileGeometry.viewportWidth + 1);
    expect(mobileGeometry.horizontalOverflow).toBeLessThanOrEqual(1);
    expect(mobileGeometry.nestedCards).toBe(0);
    await hideNextDevPortal(page);
    await page.screenshot({
      path: testInfo.outputPath('semantic-graph-mobile.png'),
      fullPage: false,
    });

    await page.setViewportSize({ width: 1440, height: 1000 });
    await inspector.getByRole('button', { name: 'Create manual link', exact: true }).click();
    await expect.poll(() => fixture.converted).toBe(true);
    expect(fixture.manualLinkBody).toEqual({
      to_note_id: TARGET_ID,
      directed: false,
      weight: 1,
      idempotency_key: expect.any(String),
      semantic_conversion: { generation_id: GENERATION_ID },
    });
    await expect(page.getByRole('button', { name: 'Create manual link', exact: true })).toHaveCount(
      0
    );

    fixture.failSemanticGraph = true;
    await page.getByRole('slider', { name: 'Minimum passage similarity' }).press('ArrowRight');
    await expect(page.getByTestId('notes-graph-degraded-state')).toBeVisible();
    await expect(page.getByTestId('notes-graph-primary-view')).toBeVisible();
    await expect(
      relationships.getByRole('button', { name: 'Manual links', exact: true })
    ).toBeVisible();
    expect(fixture.calls.some((call) => call.includes('/api/v1/jobs'))).toBe(false);
  });

  for (const theme of ['light', 'dark'] as const) {
    test(`meets semantic treatment contrast in ${theme} mode`, async ({ authedPage: page }) => {
      await page.addInitScript((requestedTheme) => {
        localStorage.setItem('theme', requestedTheme);
        localStorage.setItem('tldw:themePreset', 'default');
      }, theme);
      const fixture = new SemanticGraphFixture('ready');
      await page.setViewportSize({ width: 1440, height: 1000 });
      await openGraph(page, fixture);
      await page.getByRole('checkbox', { name: 'Similar content', exact: true }).check();
      const indicatorContrast = await measureSemanticContrast(page, 'indicator');
      await page.getByRole('button', { name: 'Relationships', exact: true }).click();
      await expect(page.getByTestId('notes-graph-semantic-evidence-toggle')).toBeVisible();
      const evidenceTextContrast = await measureSemanticContrast(page, 'evidenceText');

      expect(indicatorContrast).toBeGreaterThanOrEqual(3);
      expect(evidenceTextContrast).toBeGreaterThanOrEqual(4.5);
    });
  }

  test('cancels a rebuild and deletes the semantic index through nested routes', async ({
    authedPage: page,
  }) => {
    const fixture = new SemanticGraphFixture('ready');
    await openGraph(page, fixture);
    await page.getByRole('tab', { name: 'Similar content' }).click();
    await expect(page.getByText('Ready', { exact: true })).toBeVisible();

    await page.getByRole('button', { name: 'Rebuild index' }).click();
    await confirm(page, 'Rebuild the semantic index?', 'Rebuild');
    await expect(page.getByText('Updating', { exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Cancel indexing' }).click();
    await confirm(page, 'Cancel semantic indexing?', 'Cancel indexing');
    await expect(page.getByText('Ready', { exact: true })).toBeVisible();

    await page.getByRole('button', { name: 'Disable and delete index' }).click();
    await confirm(page, 'Disable and delete the semantic index?', 'Delete index');
    await expect(page.getByText('Off', { exact: true })).toBeVisible();
    expect(fixture.calls).toContain(
      'POST /api/v1/notes/graph/semantic-index/runs/semantic-rebuild-a/cancel'
    );
    expect(fixture.calls).toContain('DELETE /api/v1/notes/graph/semantic-index');
    expect(fixture.calls.some((call) => call.includes('/api/v1/jobs'))).toBe(false);
  });
});
