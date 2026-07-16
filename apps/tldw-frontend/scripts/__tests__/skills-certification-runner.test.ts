import { describe, expect, it, vi } from 'vitest';

import { runSkillsCertification } from '../skills-certification/run.mjs';

const webName = 'skills-cert-web';
const extensionName = 'skills-cert-extension';

function result(status: 'passed' | 'failed' | 'running', categories: string[] = []) {
  return { categories, status };
}

function report(stats = { expected: 1, flaky: 0, skipped: 0, unexpected: 0 }) {
  return { stats };
}

function harness(overrides: Record<string, unknown> = {}) {
  const calls: string[] = [];
  const files = new Map<string, unknown>();
  const registry = {
    spawn: vi.fn((command) => ({ command })),
    teardown: vi.fn(async () => undefined),
    wait: vi.fn(async () => ({ code: 0, signal: null })),
  };
  const evidence = {
    extensionDir: '/evidence/extension',
    frontendRoot: '/frontend',
    logsDir: '/evidence/logs',
    relayLedgerPath: '/evidence/extension/relay-ledger.json',
    root: '/evidence',
    runId: 'unit',
    summaryPath: '/evidence/summary.json',
    webuiDir: '/evidence/webui',
  };
  const profile = {
    baseRoot: '/runtime',
    extensionProfileDir: '/runtime/extension',
    root: '/runtime/root',
  };
  const commands = Object.fromEntries(
    [
      'webuiChromiumProbe',
      'extensionChromiumProbe',
      'authInit',
      'backend',
      'frontend',
      'webuiPlaywright',
      'extensionBuild',
      'extensionPlaywright',
    ].map((key) => [key, { name: key.replaceAll(/([a-z])([A-Z])/g, '$1-$2').toLowerCase() }])
  );
  const defaultResults: Record<string, unknown> = {
    '/evidence/webui/result.json': result('passed'),
    '/evidence/webui/report.json': report(),
    '/evidence/extension/result.json': result('passed'),
    '/evidence/extension/report.json': report(),
  };
  Object.entries(defaultResults).forEach(([key, value]) => files.set(key, value));
  const operations = {
    buildCommands: vi.fn(() => commands),
    buildEnvironments: vi.fn(() => ({})),
    createEvidence: vi.fn(() => evidence),
    createProfile: vi.fn(() => profile),
    createRegistry: vi.fn(() => registry),
    fetch: vi.fn(async (url: string) => {
      calls.push(`fetch:${new URL(url).pathname}`);
      if (url.includes('/trash'))
        return { json: async () => ({ skills: [], total: 0 }), status: 200 };
      if (
        url.includes(`/skills/${encodeURIComponent(webName)}`) ||
        url.includes(encodeURIComponent(extensionName))
      ) {
        return { json: async () => ({}), status: 404 };
      }
      return { json: async () => ({ total: 0 }), status: 200 };
    }),
    finalize: vi.fn(async ({ summaryInput }) => ({
      ...summaryInput,
      status: summaryInput.failures.length ? 'failed' : 'passed',
    })),
    installHandlers: vi.fn(() => vi.fn()),
    isBindConflict: vi.fn((text: string) => /EADDRINUSE/.test(text)),
    readJson: vi.fn((filePath: string) => files.get(filePath)),
    reservePorts: vi.fn(async () => ({ backend: 8100, web: 3100 })),
    runChild: vi.fn(async (activeRegistry: typeof registry, command: { name: string }) => {
      calls.push(command.name);
      const record = activeRegistry.spawn(command, `/evidence/logs/${command.name}.log`);
      return activeRegistry.wait(record);
    }),
    startChild: vi.fn((activeRegistry: typeof registry, command: { name: string }) => {
      calls.push(command.name);
      return activeRegistry.spawn(command, `/evidence/logs/${command.name}.log`);
    }),
    waitForHttpOk: vi.fn(async (url: string) => {
      calls.push(`health:${url}`);
    }),
    ...overrides,
  };
  return { calls, commands, evidence, files, operations, registry };
}

describe('Skills certification runner', () => {
  it('requires both initial Library and Trash totals to be exactly zero', async () => {
    const test = harness({
      fetch: vi.fn(async (url: string) => ({
        json: async () => (url.includes('/trash') ? { total: 1 } : { total: 0 }),
        status: 200,
      })),
    });
    const summary = await runSkillsCertification({ operations: test.operations });
    expect(summary.failures).toEqual(
      expect.arrayContaining([expect.objectContaining({ category: 'postcondition' })])
    );
  });

  it('tracks both package-local Chromium probes and treats a missing browser as preflight', async () => {
    const test = harness({
      runChild: vi.fn(async (registry, command) => {
        const record = registry.spawn(command, '/log');
        await registry.wait(record);
        return { code: command.name === 'extension-chromium-probe' ? 1 : 0, signal: null };
      }),
    });
    const summary = await runSkillsCertification({ operations: test.operations });
    expect(test.registry.spawn.mock.calls.map(([command]) => command.name)).toEqual(
      expect.arrayContaining(['webui-chromium-probe', 'extension-chromium-probe'])
    );
    expect(summary.failures).toEqual(
      expect.arrayContaining([expect.objectContaining({ category: 'preflight' })])
    );
  });

  it('retries only confirmed bind conflicts with fresh ports before browser execution', async () => {
    let attempts = 0;
    const test = harness({
      reservePorts: vi.fn(async () => ({ backend: 8100 + attempts, web: 3100 + attempts++ })),
      startChild: vi.fn(async (registry, command) => {
        const record = registry.spawn(command, '/log');
        return command.name === 'backend' && attempts < 2
          ? Promise.reject(new Error('EADDRINUSE'))
          : record;
      }),
    });
    await runSkillsCertification({ operations: test.operations });
    expect(test.operations.reservePorts).toHaveBeenCalledTimes(2);
    expect(test.calls.indexOf('webui-playwright')).toBeGreaterThan(
      test.calls.lastIndexOf('backend')
    );
  });

  it('does not retry a non-bind startup failure', async () => {
    const test = harness({
      startChild: vi.fn(async (registry, command) => {
        const record = registry.spawn(command, '/log');
        if (command.name === 'backend') throw new Error('import failed');
        return record;
      }),
    });
    const summary = await runSkillsCertification({ operations: test.operations });
    expect(test.operations.reservePorts).toHaveBeenCalledTimes(1);
    expect(summary.failures).toEqual(
      expect.arrayContaining([expect.objectContaining({ category: 'backend_startup' })])
    );
  });

  it('continues to extension after WebUI startup, launch, and workflow failures', async () => {
    for (const mode of ['startup', 'launch', 'workflow']) {
      const test = harness();
      if (mode === 'startup') {
        test.operations.waitForHttpOk = vi.fn(async (url: string) => {
          if (url.includes(':3100')) throw new Error('web unavailable');
        });
      }
      if (mode === 'launch') {
        test.files.delete('/evidence/webui/result.json');
        test.operations.runChild = vi.fn(async (registry, command) => {
          const record = registry.spawn(command, '/log');
          await registry.wait(record);
          return { code: command.name === 'webui-playwright' ? 1 : 0, signal: null };
        });
      }
      if (mode === 'workflow')
        test.files.set('/evidence/webui/result.json', result('failed', ['webui_workflow']));
      const summary = await runSkillsCertification({ operations: test.operations });
      expect(test.registry.spawn.mock.calls.map(([command]) => command.name)).toContain(
        'extension-playwright'
      );
      expect(summary.failures).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            category: `webui_${mode}`
              .replace('startup', 'startup')
              .replace('workflow', 'workflow')
              .replace('launch', 'launch'),
          }),
        ])
      );
    }
  });

  it('allows one same-port backend evidence restart but never turns the run green', async () => {
    let healthCalls = 0;
    const test = harness({
      waitForHttpOk: vi.fn(async () => {
        if (++healthCalls === 3) throw new Error('crashed');
      }),
    });
    const summary = await runSkillsCertification({ operations: test.operations });
    expect(test.calls.filter((call) => call === 'backend')).toHaveLength(2);
    expect(summary.status).toBe('failed');
  });

  it('marks extension infrastructure unavailable after a second crash or failed restart', async () => {
    let healthCalls = 0;
    const test = harness({
      waitForHttpOk: vi.fn(async () => {
        if (++healthCalls >= 3) throw new Error('crashed');
      }),
    });
    const summary = await runSkillsCertification({ operations: test.operations });
    expect(summary.surfaces.extension.state).toBe('not_run_infrastructure');
  });

  it('classifies extension build failure and retains every extension result category', async () => {
    const test = harness();
    test.files.set(
      '/evidence/extension/result.json',
      result('failed', ['extension_worker', 'extension_workflow', 'extension_relay'])
    );
    test.operations.runChild = vi.fn(async (registry, command) => {
      const record = registry.spawn(command, '/log');
      await registry.wait(record);
      return { code: command.name === 'extension-build' ? 1 : 0, signal: null };
    });
    const buildSummary = await runSkillsCertification({ operations: test.operations });
    expect(buildSummary.failures).toEqual(
      expect.arrayContaining([expect.objectContaining({ category: 'extension_build' })])
    );
    const retainedHarness = harness();
    retainedHarness.files.set(
      '/evidence/extension/result.json',
      result('failed', ['extension_worker', 'extension_workflow', 'extension_relay'])
    );
    const retained = await runSkillsCertification({ operations: retainedHarness.operations });
    expect(retained.failures.map((failure: { category: string }) => failure.category)).toEqual(
      expect.arrayContaining(['extension_worker', 'extension_workflow', 'extension_relay'])
    );
  });

  it('runs exact-name detail and Trash exclusion postconditions after every attempted surface', async () => {
    const test = harness();
    await runSkillsCertification({ operations: test.operations });
    expect(test.operations.reservePorts).toHaveBeenCalledTimes(1);
    expect(test.calls.filter((value) => value === 'fetch:/api/v1/skills/trash')).toHaveLength(3);
    expect(
      test.calls.filter(
        (value) =>
          value === `fetch:/api/v1/skills/${webName}` ||
          value === `fetch:/api/v1/skills/${extensionName}`
      )
    ).toHaveLength(2);
    expect(test.registry.spawn.mock.calls.map(([command]) => command.name)).toEqual(
      expect.arrayContaining([
        'webui-chromium-probe',
        'extension-chromium-probe',
        'auth-init',
        'backend',
        'frontend',
        'webui-playwright',
        'extension-build',
        'extension-playwright',
      ])
    );
  });

  it('fails zero, skipped, flaky, unexpected, missing, and malformed Playwright reports', async () => {
    for (const value of [
      report({ expected: 0, flaky: 0, skipped: 0, unexpected: 0 }),
      report({ expected: 1, flaky: 0, skipped: 1, unexpected: 0 }),
      report({ expected: 1, flaky: 1, skipped: 0, unexpected: 0 }),
      report({ expected: 1, flaky: 0, skipped: 0, unexpected: 1 }),
      undefined,
      {},
    ]) {
      const test = harness();
      test.files.set('/evidence/webui/report.json', value);
      const summary = await runSkillsCertification({ operations: test.operations });
      expect(
        summary.failures.some(
          (failure: { category: string }) => failure.category === 'webui_workflow'
        )
      ).toBe(true);
    }
  });

  it('constructs a final summary after preflight, workflow, postcondition, cleanup, and artifact failures', async () => {
    const test = harness({
      finalize: vi.fn(async ({ summaryInput }) => ({
        ...summaryInput,
        failures: [
          ...summaryInput.failures,
          { category: 'artifact_safety' },
          { category: 'cleanup' },
        ],
      })),
      runChild: vi.fn(async (registry, command) => {
        const record = registry.spawn(command, '/log');
        await registry.wait(record);
        return { code: command.name.includes('probe') ? 1 : 0, signal: null };
      }),
    });
    await runSkillsCertification({ operations: test.operations });
    expect(test.operations.finalize).toHaveBeenCalledTimes(1);
    expect(test.registry.spawn).toHaveBeenCalled();
  });
});
