import { describe, expect, it, type Mock, vi } from 'vitest';
import { tmpdir } from 'node:os';

import {
  formatSkillsCertificationDiagnostic,
  runSkillsCertification,
} from '../skills-certification/run.mjs';

const webName = 'skills-cert-web';
const extensionName = 'skills-cert-extension';

type CertificationPorts = { backend: number; web: number };
type CertificationCommand = {
  name: string;
  args?: string[];
  env?: Record<string, string>;
};
type CertificationChildOutcome = { code?: number | null; signal?: string | null };
type CertificationProcessRecord = { command: CertificationCommand };
type CertificationProcessRegistry = {
  spawn: (command: CertificationCommand, logPath?: string) => CertificationProcessRecord;
  stop: (record?: CertificationProcessRecord) => Promise<void>;
  teardown: () => Promise<void>;
  wait: (record?: CertificationProcessRecord) => Promise<CertificationChildOutcome>;
};
type CertificationEvidence = {
  extensionDir: string;
  frontendRoot: string;
  logsDir: string;
  relayLedgerPath: string;
  root: string;
  runId: string;
  summaryPath: string;
  webuiDir: string;
};
type CertificationProfile = {
  baseRoot: string;
  extensionProfileDir: string;
  root: string;
};
type CertificationFailure = { category: string; detail?: string; surface?: string };
type CertificationSummaryInput = {
  failures: CertificationFailure[];
  surfaces: Record<string, { postcondition: boolean; state: string }>;
};
type CertificationFinalizeInput = {
  evidence: CertificationEvidence;
  runtime?: object;
  summaryInput: CertificationSummaryInput;
  teardownOutcome:
    | { status: 'fulfilled'; value: unknown }
    | { status: 'rejected'; reason: unknown };
};
type CertificationHttpResponse = {
  json: () => Promise<unknown>;
  status: number;
};

interface SkillsCertificationOperationMocks {
  buildCommands: Mock<
    (input: {
      frontendRoot: string;
      ports: CertificationPorts;
      profile: CertificationProfile;
      repoRoot: string;
    }) => Record<string, CertificationCommand>
  >;
  createEvidence: Mock<(input: { frontendRoot: string }) => CertificationEvidence>;
  createProfile: Mock<
    (input: { repoRoot: string; temporaryBase: string }) => CertificationProfile
  >;
  createRegistry: Mock<() => CertificationProcessRegistry>;
  fetch: Mock<
    (url: string, init?: RequestInit) => Promise<CertificationHttpResponse>
  >;
  finalize: Mock<
    (input: CertificationFinalizeInput) => Promise<Record<string, unknown>>
  >;
  installHandlers: Mock<
    (input: {
      onSignal: () => void;
      registry: CertificationProcessRegistry;
    }) => () => void
  >;
  isBindConflict: Mock<(text: string) => boolean>;
  readJson: Mock<(filePath: string) => unknown>;
  readText: Mock<(filePath: string) => string>;
  reservePorts: Mock<(names: string[]) => Promise<CertificationPorts>>;
  removeEvidence: Mock<(evidence?: CertificationEvidence) => boolean>;
  removeRuntime: Mock<(runtime?: object) => boolean>;
  runChild: Mock<
    (
      registry: CertificationProcessRegistry,
      command: CertificationCommand,
      logPath: string
    ) => Promise<CertificationChildOutcome>
  >;
  startChild: Mock<
    (
      registry: CertificationProcessRegistry,
      command: CertificationCommand,
      logPath: string
    ) => CertificationProcessRecord | Promise<CertificationProcessRecord>
  >;
  stopChild: Mock<
    (
      registry: CertificationProcessRegistry,
      record: CertificationProcessRecord
    ) => Promise<void>
  >;
  waitForHttpOk: Mock<
    (url: string, options?: { headers?: Record<string, string> }) => Promise<void>
  >;
}

function result(status: 'passed' | 'failed' | 'running', categories: string[] = []) {
  return { categories, status };
}

function report(stats = { expected: 1, flaky: 0, skipped: 0, unexpected: 0 }) {
  return { stats };
}

function harness(overrides: Partial<SkillsCertificationOperationMocks> = {}) {
  const calls: string[] = [];
  const files = new Map<string, unknown>();
  const registry = {
    spawn: vi.fn((command, _logPath?: string) => ({ command })),
    stop: vi.fn(async (_record?: unknown) => undefined),
    teardown: vi.fn(async () => undefined),
    wait: vi.fn(async (_record?: unknown) => ({ code: 0, signal: null })),
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
  const operations: SkillsCertificationOperationMocks = {
    buildCommands: vi.fn(() => commands),
    createEvidence: vi.fn(() => evidence),
    createProfile: vi.fn(() => profile),
    createRegistry: vi.fn(() => registry),
    fetch: vi.fn(async (url: string, _init?: RequestInit) => {
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
    readText: vi.fn(() => ''),
    reservePorts: vi.fn(async () => ({ backend: 8100, web: 3100 })),
    removeEvidence: vi.fn(() => true),
    removeRuntime: vi.fn(() => true),
    runChild: vi.fn(async (activeRegistry: typeof registry, command: { name: string }) => {
      calls.push(command.name);
      const record = activeRegistry.spawn(command, `/evidence/logs/${command.name}.log`);
      return activeRegistry.wait(record);
    }),
    startChild: vi.fn((activeRegistry: typeof registry, command: { name: string }) => {
      calls.push(command.name);
      return activeRegistry.spawn(command, `/evidence/logs/${command.name}.log`);
    }),
    stopChild: vi.fn((activeRegistry: typeof registry, record: { command: { name: string } }) =>
      activeRegistry.stop(record)
    ),
    waitForHttpOk: vi.fn(async (url: string) => {
      calls.push(`health:${url}`);
    }),
    ...overrides,
  };
  return { calls, commands, evidence, files, operations, registry };
}

describe('Skills certification runner', () => {
  it('checks both initial Library and Trash totals independently', async () => {
    for (const totals of [
      { library: 1, trash: 0 },
      { library: 0, trash: 1 },
    ]) {
      const initialLibraryUrl = 'http://127.0.0.1:8100/api/v1/skills/?limit=1&offset=0';
      const initialTrashUrl = 'http://127.0.0.1:8100/api/v1/skills/trash?limit=1&offset=0';
      const test = harness({
        fetch: vi.fn(async (url: string) => ({
          json: async () => {
            if (url === initialLibraryUrl) return { total: totals.library };
            if (url === initialTrashUrl) return { total: totals.trash };
            if (url.includes('/trash?limit=500&offset=0')) return { skills: [] };
            return {};
          },
          status:
            url.includes(`/skills/${webName}`) || url.includes(`/skills/${extensionName}`)
              ? 404
              : 200,
        })),
      });
      const summary = await runSkillsCertification({ operations: test.operations });
      const failedRoute =
        totals.library === 1
          ? '/api/v1/skills/?limit=1&offset=0'
          : '/api/v1/skills/trash?limit=1&offset=0';
      expect(summary.failures).toEqual([
        { category: 'postcondition', detail: `${failedRoute} status/invariant` },
      ]);
      expect(test.operations.fetch.mock.calls.map(([url]) => url)).toEqual(
        expect.arrayContaining([initialLibraryUrl, initialTrashUrl])
      );
    }
  });

  it('bounds every runner-owned Skills API request with an abort signal', async () => {
    const test = harness();

    await runSkillsCertification({ operations: test.operations });

    expect(test.operations.fetch).toHaveBeenCalled();
    expect(
      test.operations.fetch.mock.calls.every(([, init]) => init?.signal instanceof AbortSignal)
    ).toBe(true);
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

  it('uses attempt-local logs so a later non-bind backend failure does not retry', async () => {
    let attempt = 0;
    const test = harness({
      reservePorts: vi.fn(async () => ({ backend: 8100 + attempt, web: 3100 + attempt++ })),
      readText: vi.fn((filePath) =>
        filePath.endsWith('attempt-1.log') ? `${'x'.repeat(600)} EADDRINUSE` : ''
      ),
      startChild: vi.fn(async (registry, command) => {
        if (command.name === 'backend') {
          if (attempt === 1) throw new Error('x'.repeat(201));
          throw new Error('import failed');
        }
        return registry.spawn(command, '/log');
      }),
    });
    const summary = await runSkillsCertification({ operations: test.operations });
    expect(test.operations.reservePorts).toHaveBeenCalledTimes(2);
    expect(summary.failures.map((failure) => failure.category)).toContain('backend_startup');
    expect(summary.failures.map((failure) => failure.category)).not.toContain('preflight');
  });

  it('keeps startup and health failures out of preflight when diagnostics cannot be read', async () => {
    for (const phase of ['startup', 'health']) {
      const test = harness({
        readText: vi.fn(() => {
          throw new Error('unreadable');
        }),
      });
      if (phase === 'startup') {
        test.operations.startChild = vi.fn(async () => {
          throw new Error('startup failed');
        });
      } else {
        test.operations.waitForHttpOk = vi.fn(async (url) => {
          if (url.includes('/api/v1/health')) throw new Error('health failed');
        });
      }
      const summary = await runSkillsCertification({ operations: test.operations });
      const categories = summary.failures.map((failure) => failure.category);
      expect(categories).toContain(phase === 'startup' ? 'backend_startup' : 'backend_health');
      expect(categories).not.toContain('preflight');
    }
  });

  it('stops a health-bind backend before reserving a fresh pair and does not continue when stop fails', async () => {
    const events: string[] = [];
    const test = harness({
      reservePorts: vi.fn(async () => {
        events.push('reserve');
        return { backend: 8100 + events.length, web: 3100 + events.length };
      }),
      startChild: vi.fn((registry, command) => {
        events.push(`start:${command.name}`);
        return registry.spawn(command, '/log');
      }),
      stopChild: vi.fn(async () => {
        events.push('stop:backend');
      }),
      waitForHttpOk: vi.fn(async (_url) => {
        events.push('health');
        if (events.filter((event) => event === 'health').length === 1)
          throw new Error('EADDRINUSE');
      }),
    });
    await runSkillsCertification({ operations: test.operations });
    expect(events.slice(0, 5)).toEqual([
      'reserve',
      'start:backend',
      'health',
      'stop:backend',
      'reserve',
    ]);

    const failingStop = harness({
      reservePorts: vi.fn(async () => ({ backend: 8200, web: 3200 })),
      stopChild: vi.fn(async () => {
        throw new Error('stop failed');
      }),
      waitForHttpOk: vi.fn(async () => {
        throw new Error('EADDRINUSE');
      }),
    });
    await runSkillsCertification({ operations: failingStop.operations });
    expect(failingStop.operations.reservePorts).toHaveBeenCalledTimes(1);
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

  it('fails zero, skipped, flaky, unexpected, missing, and malformed reports for both surfaces', async () => {
    for (const value of [
      report({ expected: 0, flaky: 0, skipped: 0, unexpected: 0 }),
      report({ expected: 1, flaky: 0, skipped: 1, unexpected: 0 }),
      report({ expected: 1, flaky: 1, skipped: 0, unexpected: 0 }),
      report({ expected: 1, flaky: 0, skipped: 0, unexpected: 1 }),
      undefined,
      {},
    ]) {
      for (const surface of ['webui', 'extension']) {
        const test = harness();
        test.files.set(`/evidence/${surface}/report.json`, value);
        const summary = await runSkillsCertification({ operations: test.operations });
        expect(
          summary.failures.some(
            (failure: { category: string }) => failure.category === `${surface}_workflow`
          )
        ).toBe(true);
      }
    }
  });

  it('requires an explicit allowed categories array in every final surface result', async () => {
    for (const value of [
      { status: 'passed' },
      { categories: ['unknown_category'], status: 'passed' },
    ]) {
      for (const surface of ['webui', 'extension']) {
        const test = harness();
        test.files.set(`/evidence/${surface}/result.json`, value);

        const summary = await runSkillsCertification({ operations: test.operations });

        expect(summary.failures.map((failure: { category: string }) => failure.category)).toContain(
          `${surface}_workflow`
        );
      }
    }
  });

  it('classifies malformed direct API JSON as a postcondition failure', async () => {
    const route = '/api/v1/skills/?limit=1&offset=0';
    const test = harness({
      fetch: vi.fn(async (url: string) => {
        if (url.endsWith(route)) {
          return {
            json: async () => {
              throw new SyntaxError('malformed JSON');
            },
            status: 200,
          };
        }
        if (url.includes('/trash'))
          return { json: async () => ({ skills: [], total: 0 }), status: 200 };
        if (url.includes(`/skills/${webName}`) || url.includes(`/skills/${extensionName}`))
          return { json: async () => ({}), status: 404 };
        return { json: async () => ({ total: 0 }), status: 200 };
      }),
    });

    const summary = await runSkillsCertification({ operations: test.operations });

    expect(summary.failures).toContainEqual({
      category: 'postcondition',
      detail: `${route} status/invariant`,
    });
  });

  it('retains workflow, postcondition, cleanup, and artifact safety in the final summary', async () => {
    const test = harness({
      fetch: vi.fn(async (url) => {
        const pathname = new URL(url).pathname;
        if (pathname.endsWith(webName)) return { json: async () => ({}), status: 500 };
        if (pathname.includes('/trash'))
          return { json: async () => ({ skills: [], total: 0 }), status: 200 };
        return { json: async () => ({ total: 0 }), status: 200 };
      }),
      finalize: vi.fn(async ({ summaryInput }) => ({
        ...summaryInput,
        artifact_safety: { passed: false },
        cleanup: { children_closed: false, runtime_deleted: false },
        failures: [
          ...summaryInput.failures,
          { category: 'artifact_safety' },
          { category: 'cleanup' },
        ],
      })),
    });
    test.files.set('/evidence/webui/result.json', result('failed', ['webui_workflow']));
    const summary = await runSkillsCertification({ operations: test.operations });
    expect(test.operations.finalize).toHaveBeenCalledTimes(1);
    expect(summary.failures.map((failure: { category: string }) => failure.category)).toEqual(
      expect.arrayContaining(['webui_workflow', 'postcondition', 'cleanup', 'artifact_safety'])
    );
    expect(summary.artifact_safety).toEqual({ passed: false });
    expect(summary.cleanup).toEqual({ children_closed: false, runtime_deleted: false });
  });

  it('keeps extension evidence after thrown WebUI run and result reads', async () => {
    for (const failure of ['run', 'read']) {
      const test = harness();
      if (failure === 'run') {
        test.files.delete('/evidence/webui/result.json');
        test.operations.runChild = vi.fn(async (registry, command) => {
          const record = registry.spawn(command, '/log');
          if (command.name === 'webui-playwright') throw new Error('web child exploded');
          return registry.wait(record);
        });
      } else {
        test.operations.readJson = vi.fn((filePath) => {
          if (filePath === '/evidence/webui/result.json') throw new Error('web result unreadable');
          return test.files.get(filePath);
        });
      }
      const summary = await runSkillsCertification({ operations: test.operations });
      expect(summary.failures.map((failure: { category: string }) => failure.category)).toContain(
        failure === 'run' ? 'webui_launch' : 'webui_workflow'
      );
      expect(test.registry.spawn.mock.calls.map(([command]) => command.name)).toContain(
        'extension-playwright'
      );
    }
  });

  it('keeps a present WebUI result when only report reading throws', async () => {
    const test = harness();
    test.files.set('/evidence/webui/result.json', result('failed', ['webui_workflow']));
    test.operations.runChild = vi.fn(async (registry, command) => {
      const record = registry.spawn(command, '/log');
      return command.name === 'webui-playwright'
        ? { code: 1, signal: null }
        : registry.wait(record);
    });
    test.operations.readJson = vi.fn((filePath) => {
      if (filePath === '/evidence/webui/report.json') throw new Error('report unreadable');
      return test.files.get(filePath);
    });
    const summary = await runSkillsCertification({ operations: test.operations });
    expect(summary.failures.map((failure: { category: string }) => failure.category)).toContain(
      'webui_workflow'
    );
    expect(summary.failures.map((failure: { category: string }) => failure.category)).not.toContain(
      'webui_launch'
    );
    expect(test.registry.spawn.mock.calls.map(([command]) => command.name)).toContain(
      'extension-playwright'
    );
  });

  it('maps present running WebUI results to workflow for both browser exits and continues extension evidence', async () => {
    for (const code of [0, 1]) {
      const test = harness();
      test.files.set('/evidence/webui/result.json', { status: 'running' });
      test.operations.runChild = vi.fn(async (registry, command) => {
        const record = registry.spawn(command, '/log');
        return command.name === 'webui-playwright' ? { code, signal: null } : registry.wait(record);
      });
      const summary = await runSkillsCertification({ operations: test.operations });
      const categories = summary.failures.map((failure: { category: string }) => failure.category);
      expect(categories).toContain('webui_workflow');
      expect(categories).not.toContain('webui_launch');
      expect(test.registry.spawn.mock.calls.map(([command]) => command.name)).toContain(
        'extension-playwright'
      );
    }
  });

  it('classifies thrown build and extension browser operations without reclassifying backend health', async () => {
    for (const target of ['extension-build', 'extension-playwright', 'extension-read']) {
      const test = harness();
      if (target === 'extension-read') {
        test.operations.readJson = vi.fn((filePath) => {
          if (filePath === '/evidence/extension/result.json')
            throw new Error('extension result unreadable');
          return test.files.get(filePath);
        });
      } else {
        if (target === 'extension-playwright') test.files.delete('/evidence/extension/result.json');
        test.operations.runChild = vi.fn(async (registry, command) => {
          const record = registry.spawn(command, '/log');
          if (command.name === target) throw new Error(`${target} exploded`);
          return registry.wait(record);
        });
      }
      const summary = await runSkillsCertification({ operations: test.operations });
      expect(summary.failures.map((failure: { category: string }) => failure.category)).toContain(
        target === 'extension-build'
          ? 'extension_build'
          : target === 'extension-playwright'
            ? 'extension_launch'
            : 'extension_workflow'
      );
      expect(
        summary.failures.map((failure: { category: string }) => failure.category)
      ).not.toContain('backend_health');
    }
  });

  it('retains every present extension category when only its report read throws', async () => {
    const test = harness();
    const categories = [
      'extension_launch',
      'extension_worker',
      'extension_workflow',
      'extension_relay',
    ];
    test.files.set('/evidence/extension/result.json', result('failed', categories));
    test.operations.readJson = vi.fn((filePath) => {
      if (filePath === '/evidence/extension/report.json') throw new Error('report unreadable');
      return test.files.get(filePath);
    });
    const summary = await runSkillsCertification({ operations: test.operations });
    expect(summary.failures.map((failure: { category: string }) => failure.category)).toEqual(
      expect.arrayContaining(categories)
    );
  });

  it('maps present running extension results to workflow for both browser exits', async () => {
    for (const code of [0, 1]) {
      const test = harness();
      test.files.set('/evidence/extension/result.json', { status: 'running' });
      test.operations.runChild = vi.fn(async (registry, command) => {
        const record = registry.spawn(command, '/log');
        return command.name === 'extension-playwright'
          ? { code, signal: null }
          : registry.wait(record);
      });
      const summary = await runSkillsCertification({ operations: test.operations });
      const categories = summary.failures.map((failure: { category: string }) => failure.category);
      expect(categories).toContain('extension_workflow');
      expect(categories).not.toContain('extension_launch');
    }
  });

  it('retries a frontend-only bind conflict with fresh pairs and stops old attempt children', async () => {
    const pairs = [
      { backend: 8100, web: 3100 },
      { backend: 8101, web: 3101 },
    ];
    const test = harness({
      reservePorts: vi.fn(async () => pairs.shift()),
      waitForHttpOk: vi.fn(async (url) => {
        if (url.includes(':3100')) throw new Error('EADDRINUSE');
      }),
    });
    await runSkillsCertification({ operations: test.operations });
    expect(test.operations.reservePorts).toHaveBeenCalledTimes(2);
    expect(
      test.operations.buildCommands.mock.calls.filter(([value]) => value?.ports?.backend === 8100)
    ).toHaveLength(1);
    expect(test.registry.stop).toHaveBeenCalledTimes(2);
  });

  it('continues on the same backend after a frontend bind cleanup failure without reserving again', async () => {
    let backendHealthCalls = 0;
    const test = harness({
      stopChild: vi.fn(async (_registry, record) => {
        if (record.command.name === 'frontend') throw new Error('frontend stop failed');
      }),
      waitForHttpOk: vi.fn(async (url) => {
        if (url.includes(':3100')) throw new Error('EADDRINUSE');
        backendHealthCalls += 1;
      }),
    });
    test.registry.teardown.mockRejectedValue(new Error('final teardown failed'));
    await runSkillsCertification({ operations: test.operations });
    const categories = test.operations.finalize.mock.calls[0][0].summaryInput.failures.map(
      (failure) => failure.category
    );
    expect(test.operations.reservePorts).toHaveBeenCalledTimes(1);
    expect(test.operations.stopChild.mock.calls.map(([, record]) => record.command.name)).toEqual([
      'frontend',
    ]);
    expect(categories).toEqual(expect.arrayContaining(['cleanup', 'webui_startup']));
    expect(categories).not.toContain('preflight');
    expect(backendHealthCalls).toBeGreaterThanOrEqual(3);
    expect(test.registry.spawn.mock.calls.map(([command]) => command.name)).toContain(
      'extension-playwright'
    );
    expect(test.operations.finalize.mock.calls[0][0].teardownOutcome.status).toBe('rejected');
  });

  it('continues on the same backend after a backend bind cleanup failure without reserving again', async () => {
    const test = harness({
      stopChild: vi.fn(async (_registry, record) => {
        if (record.command.name === 'backend') throw new Error('backend stop failed');
      }),
      waitForHttpOk: vi.fn(async (url) => {
        if (url.includes(':3100')) throw new Error('EADDRINUSE');
      }),
    });
    await runSkillsCertification({ operations: test.operations });
    const categories = test.operations.finalize.mock.calls[0][0].summaryInput.failures.map(
      (failure) => failure.category
    );
    expect(test.operations.reservePorts).toHaveBeenCalledTimes(1);
    expect(test.operations.stopChild.mock.calls.map(([, record]) => record.command.name)).toEqual([
      'frontend',
      'backend',
    ]);
    expect(categories).toEqual(expect.arrayContaining(['cleanup', 'webui_startup']));
    expect(categories).not.toContain('preflight');
    expect(test.registry.spawn.mock.calls.map(([command]) => command.name)).toContain(
      'extension-playwright'
    );
  });

  it('blocks extension after a failed same-URL health decision following frontend cleanup failure', async () => {
    let backendHealthCalls = 0;
    const test = harness({
      stopChild: vi.fn(async () => {
        throw new Error('frontend stop failed');
      }),
      waitForHttpOk: vi.fn(async (url) => {
        if (url.includes(':3100')) throw new Error('EADDRINUSE');
        backendHealthCalls += 1;
        if (backendHealthCalls === 2) throw new Error('backend unavailable');
      }),
    });
    const summary = await runSkillsCertification({ operations: test.operations });
    expect(test.operations.reservePorts).toHaveBeenCalledTimes(1);
    expect(summary.surfaces.extension.state).toBe('not_run_infrastructure');
    expect(test.registry.spawn.mock.calls.map(([command]) => command.name)).not.toContain(
      'extension-playwright'
    );
  });

  it('classifies backend-health bind cleanup failure without reserving again or using preflight', async () => {
    let backendHealthCalls = 0;
    const test = harness({
      stopChild: vi.fn(async () => {
        throw new Error('backend stop failed');
      }),
      waitForHttpOk: vi.fn(async (url) => {
        if (!url.includes('/api/v1/health')) return;
        backendHealthCalls += 1;
        if (backendHealthCalls === 1) throw new Error('EADDRINUSE');
      }),
    });
    await runSkillsCertification({ operations: test.operations });
    const categories = test.operations.finalize.mock.calls[0][0].summaryInput.failures.map(
      (failure) => failure.category
    );
    expect(test.operations.reservePorts).toHaveBeenCalledTimes(1);
    expect(categories).toContain('cleanup');
    expect(categories).not.toContain('preflight');
    expect(test.registry.spawn.mock.calls.map(([command]) => command.name)).toContain(
      'extension-playwright'
    );
  });

  it('caps frontend bind retries at three fresh pairs', async () => {
    let next = 0;
    const pairs: string[] = [];
    const test = harness({
      buildCommands: vi.fn((input) => {
        if (input.ports.backend > 1) pairs.push(`${input.ports.backend}:${input.ports.web}`);
        return test.commands;
      }),
      reservePorts: vi.fn(async () => ({ backend: 8100 + next, web: 3100 + next++ })),
      waitForHttpOk: vi.fn(async (url) => {
        if (url.includes(':31')) throw new Error('EADDRINUSE');
      }),
    });
    await runSkillsCertification({ operations: test.operations });
    expect(test.operations.reservePorts).toHaveBeenCalledTimes(3);
    expect(pairs).toEqual(['8100:3100', '8101:3101', '8102:3102']);
    expect(test.registry.stop).toHaveBeenCalledTimes(4);
  });

  it('restarts the same backend after frontend startup failure to collect extension evidence', async () => {
    let healthCalls = 0;
    const test = harness({
      waitForHttpOk: vi.fn(async (url) => {
        if (url.includes(':3100')) throw new Error('frontend unavailable');
        if (++healthCalls === 2) throw new Error('backend unavailable');
      }),
    });
    const summary = await runSkillsCertification({ operations: test.operations });
    expect(
      test.registry.spawn.mock.calls
        .map(([command]) => command.name)
        .filter((name) => name === 'backend')
    ).toHaveLength(2);
    expect(test.registry.stop).toHaveBeenCalledTimes(1);
    expect(test.registry.spawn.mock.calls.map(([command]) => command.name)).toContain(
      'extension-playwright'
    );
    expect(summary.surfaces.extension.state).toBe('passed');
    expect(summary.status).toBe('failed');
  });

  it('stops the prior backend before its one same-port evidence restart and blocks a second crash', async () => {
    let healthCalls = 0;
    const test = harness({
      waitForHttpOk: vi.fn(async () => {
        healthCalls += 1;
        if (healthCalls === 3 || healthCalls >= 5) throw new Error('backend crashed');
      }),
    });
    const summary = await runSkillsCertification({ operations: test.operations });
    expect(test.registry.stop).toHaveBeenCalledTimes(1);
    expect(summary.surfaces.extension.state).toBe('not_run_infrastructure');
  });

  it('locks all reserve and startup command construction before WebUI execution', async () => {
    const events: string[] = [];
    const test = harness({
      buildCommands: vi.fn((input) => {
        if (input.ports.backend > 1) events.push(`build:${input.ports.backend}:${input.ports.web}`);
        return test.commands;
      }),
      reservePorts: vi.fn(async () => {
        events.push('reserve');
        return { backend: 8100, web: 3100 };
      }),
      runChild: vi.fn(async (registry, command) => {
        if (command.name === 'webui-playwright') events.push('webui');
        const record = registry.spawn(command, '/log');
        return registry.wait(record);
      }),
    });
    await runSkillsCertification({ operations: test.operations });
    const webuiIndex = events.indexOf('webui');
    expect(webuiIndex).toBeGreaterThan(0);
    expect(events.slice(0, webuiIndex)).toEqual(
      expect.arrayContaining(['reserve', 'build:8100:3100'])
    );
    expect(events.slice(webuiIndex + 1)).not.toContain('reserve');
    expect(events.slice(webuiIndex + 1)).not.toContain('build:8100:3100');
  });

  it('restarts the exact original backend after WebUI without a fresh reserve or build', async () => {
    const events: string[] = [];
    let healthCalls = 0;
    const test = harness({
      buildCommands: vi.fn((input) => {
        const commands = Object.fromEntries(
          Object.entries(test.commands).map(([key, command]) => [
            key,
            { ...command, args: [`--port=${input.ports.backend}`] },
          ])
        );
        if (input.ports.backend > 1) events.push(`build:${input.ports.backend}:${input.ports.web}`);
        return commands;
      }),
      reservePorts: vi.fn(async () => {
        events.push('reserve');
        return { backend: 8100, web: 3100 };
      }),
      startChild: vi.fn((registry, command) => {
        events.push(`start:${command.name}:${command.args[0]}`);
        return registry.spawn(command, '/log');
      }),
      stopChild: vi.fn(async (_registry, record) => {
        events.push(`stop:${record.command.name}:${record.command.args[0]}`);
      }),
      waitForHttpOk: vi.fn(async () => {
        if (++healthCalls === 3) throw new Error('backend crashed');
      }),
    });
    await runSkillsCertification({ operations: test.operations });
    const webuiStart = events.findIndex((event) => event.startsWith('start:frontend'));
    const restartStop = events.indexOf('stop:backend:--port=8100');
    const backendStarts = events
      .map((event, index) => [event, index] as const)
      .filter(([event]) => event === 'start:backend:--port=8100');
    expect(webuiStart).toBeGreaterThan(-1);
    expect(backendStarts).toHaveLength(2);
    expect(restartStop).toBeLessThan(backendStarts[1][1]);
    expect(events.filter((event) => event === 'reserve')).toHaveLength(1);
    expect(events.filter((event) => event === 'build:8100:3100')).toHaveLength(1);
  });

  it('uses the original backend URL through restart ceilings without a second restart', async () => {
    for (const failureCall of [3, 4]) {
      let backendHealthCalls = 0;
      const healthUrls: string[] = [];
      const test = harness({
        buildCommands: vi.fn((input) => ({
          ...test.commands,
          backend: { ...test.commands.backend, args: [`--port=${input.ports.backend}`] },
        })),
        reservePorts: vi.fn(async () => ({ backend: 8100, web: 3100 })),
        startChild: vi.fn((registry, command) => registry.spawn(command, '/log')),
        waitForHttpOk: vi.fn(async (url) => {
          if (!url.includes('/api/v1/health')) return;
          healthUrls.push(url);
          if (++backendHealthCalls === 2 || backendHealthCalls === failureCall)
            throw new Error('backend crashed');
        }),
      });
      const summary = await runSkillsCertification({ operations: test.operations });
      expect(healthUrls).toEqual(healthUrls.map(() => 'http://127.0.0.1:8100/api/v1/health'));
      expect(
        test.registry.spawn.mock.calls
          .map(([command]) => command.name)
          .filter((name) => name === 'backend')
      ).toHaveLength(2);
      expect(test.operations.reservePorts).toHaveBeenCalledTimes(1);
      expect(
        test.operations.buildCommands.mock.calls.filter(([input]) => input.ports.backend === 8100)
      ).toHaveLength(1);
      expect(summary.surfaces.extension.state).toBe('not_run_infrastructure');
    }
  });

  it('runs Trash exclusion even when a surface detail request throws', async () => {
    const test = harness({
      fetch: vi.fn(async (url) => {
        const pathname = new URL(url).pathname;
        test.calls.push(`fetch:${pathname}`);
        if (pathname.endsWith(webName) || pathname.endsWith(extensionName))
          throw new Error('detail failed');
        if (pathname.includes('/trash'))
          return { json: async () => ({ skills: [], total: 0 }), status: 200 };
        return { json: async () => ({ total: 0 }), status: 200 };
      }),
    });
    await runSkillsCertification({ operations: test.operations });
    expect(test.calls.filter((value) => value === 'fetch:/api/v1/skills/trash')).toHaveLength(3);
  });

  it('uses a generic artifact safety diagnostic and bounds/redacts other diagnostics', () => {
    expect(
      formatSkillsCertificationDiagnostic({
        artifact_safety: { passed: false },
        primary_category: 'webui_workflow',
      })
    ).toBe('artifact_safety');
    expect(
      formatSkillsCertificationDiagnostic({
        artifact_safety: { passed: true },
        primary_category: `x${'y'.repeat(600)}`,
      })
    ).toHaveLength(500);
  });

  it('creates the disposable profile below the system temporary directory', async () => {
    const test = harness();
    await runSkillsCertification({ operations: test.operations });
    expect(test.operations.createProfile.mock.calls[0][0].temporaryBase).toBe(tmpdir());
  });

  it('finalizes evidence when disposable profile creation fails', async () => {
    const test = harness({
      createProfile: vi.fn(() => {
        throw new Error('profile failed');
      }),
    });
    await runSkillsCertification({ operations: test.operations });
    expect(test.operations.finalize).toHaveBeenCalledWith(
      expect.objectContaining({ evidence: test.evidence, runtime: undefined })
    );
  });

  it('reports truthful cleanup when evidence creation fails before profile setup', async () => {
    const test = harness({
      createEvidence: vi.fn(() => {
        throw new Error('evidence failed');
      }),
    });
    const summary = await runSkillsCertification({ operations: test.operations });
    expect(summary.failures.map((failure) => failure.category)).toContain('preflight');
    expect(summary.failures.map((failure) => failure.category)).not.toContain('cleanup');
    expect(summary.artifact_safety.passed).toBe(false);
    expect(summary.cleanup).toEqual({ children_closed: true, runtime_deleted: true });
  });

  it('passes a rollback-failed runtime descriptor to finalization and reports failed retry cleanup', async () => {
    const runtime = {
      baseRoot: '/runtime',
      markerPath: '/runtime/root/.marker',
      root: '/runtime/root',
    };
    const error = new AggregateError([new Error('setup'), new Error('rollback')], 'profile failed') as
      AggregateError & { runtime?: typeof runtime };
    error.runtime = runtime;
    const test = harness({
      createProfile: vi.fn(() => {
        throw error;
      }),
      finalize: vi.fn(async () => {
        throw new Error('finalizer failed');
      }),
      removeRuntime: vi.fn(() => {
        throw new Error('retry failed');
      }),
    });
    const summary = await runSkillsCertification({ operations: test.operations });
    expect(test.operations.finalize).toHaveBeenCalledWith(expect.objectContaining({ runtime }));
    expect(test.operations.removeRuntime).toHaveBeenCalledWith(runtime);
    expect(summary.cleanup.runtime_deleted).toBe(false);
  });

  it('safely removes remaining roots after a finalizer rejection', async () => {
    const test = harness({
      finalize: vi.fn(async () => {
        throw new Error('finalizer failed');
      }),
      removeEvidence: vi.fn(() => true),
      removeRuntime: vi.fn(() => true),
    });
    const summary = await runSkillsCertification({ operations: test.operations });
    expect(test.operations.removeRuntime).toHaveBeenCalledWith(
      test.operations.createProfile.mock.results[0].value
    );
    expect(test.operations.removeEvidence).toHaveBeenCalledWith(test.evidence);
    expect(summary.artifact_safety).toEqual({ passed: false });
  });

  it('reports artifact safety when marker-safe fallback removal rejects', async () => {
    const test = harness({
      finalize: vi.fn(async () => {
        throw new Error('finalizer failed');
      }),
      removeEvidence: vi.fn(() => {
        throw new Error('evidence marker mismatch');
      }),
      removeRuntime: vi.fn(() => {
        throw new Error('runtime marker mismatch');
      }),
    });
    const summary = await runSkillsCertification({ operations: test.operations });
    expect(test.operations.removeRuntime).toHaveBeenCalledTimes(1);
    expect(test.operations.removeEvidence).toHaveBeenCalledTimes(1);
    expect(summary.artifact_safety).toEqual({ passed: false });
    expect(summary.cleanup.runtime_deleted).toBe(false);
  });

  it('retains SIGINT through deferred teardown before removing handlers', async () => {
    let onSignal: () => void;
    let releaseTeardown: () => void;
    const events: string[] = [];
    const test = harness({
      installHandlers: vi.fn(({ onSignal: captured }) => {
        onSignal = captured;
        return () => events.push('remove');
      }),
    });
    test.operations.finalize = vi.fn(async ({ summaryInput }) => ({
      ...summaryInput,
      primary_category: summaryInput.failures[0]?.category ?? null,
    }));
    test.registry.teardown.mockImplementation(
      () => new Promise<void>((resolve) => (releaseTeardown = resolve))
    );
    const run = runSkillsCertification({ operations: test.operations });
    await vi.waitFor(() => expect(test.registry.teardown).toHaveBeenCalled());
    onSignal();
    expect(events).toEqual([]);
    releaseTeardown();
    const summary = await run;
    expect(summary.primary_category).toBe('interrupted');
    expect(events).toEqual(['remove']);
  });

  it('re-finalizes an interrupted summary when SIGTERM arrives during finalization', async () => {
    let onSignal: () => void;
    let releaseFinalizer: () => void;
    const test = harness({
      installHandlers: vi.fn(({ onSignal: captured }) => {
        onSignal = captured;
        return vi.fn();
      }),
    });
    test.operations.finalize = vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise(
            (resolve) =>
              (releaseFinalizer = () =>
                resolve({
                  artifact_safety: { passed: true },
                  cleanup: { children_closed: true, runtime_deleted: true },
                  status: 'passed',
                }))
          )
      )
      .mockImplementation(async ({ summaryInput }) => ({
        ...summaryInput,
        primary_category: 'interrupted',
      }));
    const run = runSkillsCertification({ operations: test.operations });
    await vi.waitFor(() => expect(test.operations.finalize).toHaveBeenCalled());
    onSignal();
    releaseFinalizer();
    const summary = await run;
    expect(test.operations.finalize).toHaveBeenCalledTimes(2);
    expect(summary.primary_category).toBe('interrupted');
  });

  it('does not re-finalize removed artifact-failing evidence after a signal', async () => {
    let onSignal: () => void;
    let release: () => void;
    const test = harness({
      installHandlers: vi.fn(({ onSignal: captured }) => {
        onSignal = captured;
        return vi.fn();
      }),
    });
    test.operations.finalize = vi.fn(
      () =>
        new Promise(
          (resolve) =>
            (release = () =>
              resolve({
                artifact_safety: { passed: false },
                cleanup: { children_closed: true, runtime_deleted: true },
              }))
        )
    );
    const run = runSkillsCertification({ operations: test.operations });
    await vi.waitFor(() => expect(test.operations.finalize).toHaveBeenCalled());
    onSignal();
    release();
    const summary = await run;
    expect(test.operations.finalize).toHaveBeenCalledTimes(1);
    expect(summary.primary_category).toBe('interrupted');
  });

  it('uses marker-safe cleanup when an interrupted refresh finalizer rejects', async () => {
    let onSignal: () => void;
    let release: () => void;
    const test = harness({
      installHandlers: vi.fn(({ onSignal: captured }) => {
        onSignal = captured;
        return vi.fn();
      }),
    });
    test.operations.removeRuntime = vi.fn(() => true);
    test.operations.removeEvidence = vi.fn(() => true);
    test.operations.finalize = vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise(
            (resolve) =>
              (release = () =>
                resolve({
                  artifact_safety: { passed: true },
                  cleanup: { children_closed: true, runtime_deleted: true },
                }))
          )
      )
      .mockRejectedValueOnce(new Error('refresh failed'));
    const summaryPromise = runSkillsCertification({ operations: test.operations });
    await vi.waitFor(() => expect(test.operations.finalize).toHaveBeenCalled());
    onSignal();
    release();
    const summary = await summaryPromise;
    expect(test.operations.removeRuntime).toHaveBeenCalled();
    expect(test.operations.removeEvidence).toHaveBeenCalled();
    expect(summary.artifact_safety.passed).toBe(false);
    expect(summary.failures.map((failure) => failure.category)).toContain('interrupted');
  });

  it('uses safe fallback when handler-removal refresh finalization rejects', async () => {
    let removals = 0;
    const test = harness({
      installHandlers: vi.fn(() => () => {
        removals += 1;
        if (removals === 1) throw new Error('partial removal');
      }),
    });
    test.operations.removeRuntime = vi.fn(() => true);
    test.operations.removeEvidence = vi.fn(() => true);
    test.operations.finalize = vi
      .fn()
      .mockResolvedValueOnce({
        artifact_safety: { passed: true },
        cleanup: { children_closed: true, runtime_deleted: true },
      })
      .mockRejectedValueOnce(new Error('refresh failed'));
    const summary = await runSkillsCertification({ operations: test.operations });
    expect(removals).toBe(2);
    expect(test.operations.removeRuntime).toHaveBeenCalled();
    expect(test.operations.removeEvidence).toHaveBeenCalled();
    expect(summary.artifact_safety.passed).toBe(false);
    expect(summary.failures.map((failure) => failure.category)).toContain('cleanup');
  });
});
