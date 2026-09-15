import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it, vi } from 'vitest';

import { assertNoSecretLeaks } from '../onboarding-uat/artifacts.mjs';
import { SKILLS_CERT_API_KEY } from '../skills-certification/profile.mjs';
import * as evidenceModule from '../skills-certification/evidence.mjs';

const {
  buildCertificationSummary,
  createSkillsCertificationEvidence,
  finalizeSkillsCertificationEvidence,
  removeSkillsCertificationEvidence,
  removeSkillsCertificationRuntime,
  writeSanitizedJson,
} = evidenceModule;

const maxLogBytes = 1024 * 1024;
type EvidenceFinalizationOptions = {
  evidence: ReturnType<typeof createSkillsCertificationEvidence>;
  runtime?: { baseRoot: string; markerPath: string; root: string };
  scanArtifacts?: (root: string, options?: { additionalSecrets?: string[] }) => void;
  summaryInput?: ReturnType<typeof passingSummaryInput>;
  syntheticKey?: string;
  teardownOutcome?: PromiseSettledResult<unknown>;
};
const finalizeEvidence = finalizeSkillsCertificationEvidence as unknown as (
  options: EvidenceFinalizationOptions
) => Promise<ReturnType<typeof buildCertificationSummary>>;
const temporaryRoots: string[] = [];
const primaryCategoryOrder = [
  'preflight',
  'backend_startup',
  'backend_health',
  'webui_startup',
  'webui_launch',
  'webui_workflow',
  'extension_build',
  'extension_launch',
  'extension_worker',
  'extension_workflow',
  'extension_relay',
  'postcondition',
  'artifact_safety',
  'cleanup',
];

function createFixture(runId = 'unit-run') {
  const root = mkdtempSync(path.join(tmpdir(), 'skills-certification-evidence-test-'));
  temporaryRoots.push(root);

  const frontendRoot = path.join(root, 'repo/apps/tldw-frontend');
  const runtimeBaseRoot = path.join(root, 'runtime');
  mkdirSync(frontendRoot, { recursive: true });
  mkdirSync(runtimeBaseRoot, { recursive: true });

  const evidence = createSkillsCertificationEvidence({ frontendRoot, runId });
  const runtimeRoot = path.join(runtimeBaseRoot, 'tldw-skills-certification-runtime-unit');
  const runtimeMarkerPath = path.join(runtimeRoot, '.skills-certification-runtime');
  mkdirSync(runtimeRoot);
  writeFileSync(runtimeMarkerPath, 'tldw-skills-certification-runtime\n', 'utf8');

  return {
    evidence,
    frontendRoot,
    runtime: {
      baseRoot: runtimeBaseRoot,
      markerPath: runtimeMarkerPath,
      root: runtimeRoot,
    },
  };
}

function surface(
  state: 'passed' | 'failed' | 'not_run_infrastructure' = 'passed',
  postcondition = true
) {
  return { postcondition, state };
}

function passingSummaryInput(runId = 'unit-run') {
  return {
    artifact_safety: { passed: true },
    cleanup: { children_closed: true, runtime_deleted: true },
    failures: [],
    run_id: runId,
    surfaces: {
      extension: surface(),
      webui: surface(),
    },
  };
}

function finalizationOptions(fixture: ReturnType<typeof createFixture>) {
  return {
    evidence: fixture.evidence,
    runtime: fixture.runtime,
    summaryInput: passingSummaryInput(fixture.evidence.runId),
    syntheticKey: SKILLS_CERT_API_KEY,
    teardownOutcome: { status: 'fulfilled' as const, value: undefined },
  };
}

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) {
    rmSync(root, { force: true, recursive: true });
  }
});

describe('Skills certification evidence paths and writes', () => {
  it('exports only the fixed evidence entry points', () => {
    expect(Object.keys(evidenceModule).sort()).toEqual([
      'buildCertificationSummary',
      'createSkillsCertificationEvidence',
      'finalizeSkillsCertificationEvidence',
      'removeSkillsCertificationEvidence',
      'removeSkillsCertificationRuntime',
      'writeSanitizedJson',
    ]);
  });

  it('creates the fixed evidence layout with a distinct exact marker', () => {
    const { evidence, frontendRoot, runtime } = createFixture('fixed-layout');
    const expectedRoot = path.join(frontendRoot, 'test-results/skills-certification/fixed-layout');

    expect(evidence).toEqual({
      baseRoot: path.dirname(expectedRoot),
      extensionDir: path.join(expectedRoot, 'extension'),
      frontendRoot: path.resolve(frontendRoot),
      logsDir: path.join(expectedRoot, 'logs'),
      markerPath: path.join(expectedRoot, '.skills-certification-evidence'),
      relayLedgerPath: path.join(expectedRoot, 'extension/relay-ledger.json'),
      root: expectedRoot,
      runId: 'fixed-layout',
      summaryPath: path.join(expectedRoot, 'summary.json'),
      webuiDir: path.join(expectedRoot, 'webui'),
    });
    expect(readFileSync(evidence.markerPath, 'utf8')).toBe('tldw-skills-certification-evidence\n');
    expect(path.basename(evidence.markerPath)).not.toBe(path.basename(runtime.markerPath));
    expect(existsSync(evidence.logsDir)).toBe(true);
    expect(existsSync(evidence.webuiDir)).toBe(true);
    expect(existsSync(evidence.extensionDir)).toBe(true);
    expect(existsSync(evidence.summaryPath)).toBe(false);
    expect(existsSync(evidence.relayLedgerPath)).toBe(false);
  });

  it('safely removes marked roots idempotently and rejects a mismatched marker path', () => {
    const fixture = createFixture('safe-removal');

    expect(removeSkillsCertificationRuntime(fixture.runtime)).toBe(true);
    expect(removeSkillsCertificationRuntime(fixture.runtime)).toBe(true);
    expect(removeSkillsCertificationEvidence(fixture.evidence)).toBe(true);
    expect(removeSkillsCertificationEvidence(fixture.evidence)).toBe(true);

    expect(() =>
      removeSkillsCertificationRuntime({
        ...fixture.runtime,
        markerPath: path.join(fixture.runtime.baseRoot, '.skills-certification-runtime'),
      })
    ).toThrow(/exact marker path/);
  });

  it('redacts retained JSON while keeping it complete and parseable', () => {
    const { evidence } = createFixture('sanitized-json');
    const payload = 'x'.repeat(maxLogBytes + 1024);

    const sanitized = writeSanitizedJson(evidence.relayLedgerPath, {
      api_key: SKILLS_CERT_API_KEY,
      payload,
    });
    const retained = readFileSync(evidence.relayLedgerPath, 'utf8');
    const parsed = JSON.parse(retained);

    expect(sanitized).toEqual(parsed);
    expect(parsed.api_key).toBe('[REDACTED]');
    expect(parsed.payload).toBe(payload);
    expect(retained).not.toContain(SKILLS_CERT_API_KEY);
    expect(Buffer.byteLength(retained)).toBeGreaterThan(maxLogBytes);
  });
});

describe('Skills certification summary aggregation', () => {
  it('returns exactly the documented passing summary shape', () => {
    const summary = buildCertificationSummary(passingSummaryInput('summary-shape'));

    expect(summary).toEqual({
      artifact_safety: { passed: true },
      cleanup: { children_closed: true, runtime_deleted: true },
      failures: [],
      primary_category: null,
      run_id: 'summary-shape',
      status: 'passed',
      surfaces: {
        extension: { postcondition: true, state: 'passed' },
        webui: { postcondition: true, state: 'passed' },
      },
    });
    expect(Object.keys(summary).sort()).toEqual([
      'artifact_safety',
      'cleanup',
      'failures',
      'primary_category',
      'run_id',
      'status',
      'surfaces',
    ]);
  });

  it('uses interrupted as the primary category regardless of phase order', () => {
    const summary = buildCertificationSummary({
      ...passingSummaryInput(),
      failures: [{ category: 'preflight' }, { category: 'interrupted', detail: 'SIGINT' }],
    });

    expect(summary.status).toBe('failed');
    expect(summary.primary_category).toBe('interrupted');
  });

  it('chooses the first present category in the fixed phase order', () => {
    for (let index = 0; index < primaryCategoryOrder.length; index += 1) {
      const failures = primaryCategoryOrder
        .slice(index)
        .reverse()
        .map((category) => ({ category }));

      expect(
        buildCertificationSummary({
          ...passingSummaryInput(),
          failures,
        }).primary_category
      ).toBe(primaryCategoryOrder[index]);
    }
  });

  it('retains workflow, artifact safety, and cleanup failures together', () => {
    const summary = buildCertificationSummary({
      ...passingSummaryInput(),
      artifact_safety: { passed: false },
      cleanup: { children_closed: false, runtime_deleted: false },
      failures: [
        {
          category: 'webui_workflow',
          detail: 'workflow failed',
          ignored: 'not retained',
          surface: 'webui',
        },
      ],
      surfaces: {
        extension: surface('not_run_infrastructure', false),
        webui: surface('failed', false),
      },
    });

    expect(summary.failures).toEqual([
      {
        category: 'webui_workflow',
        detail: 'workflow failed',
        surface: 'webui',
      },
      { category: 'artifact_safety' },
      { category: 'cleanup' },
    ]);
    expect(summary.primary_category).toBe('webui_workflow');
    expect(summary.cleanup).toEqual({
      children_closed: false,
      runtime_deleted: false,
    });
    expect(summary.artifact_safety).toEqual({ passed: false });
    expect(summary.status).toBe('failed');
  });

  it('redacts failure detail before capping it at 500 characters', () => {
    const detail = `${'p'.repeat(480)} ${SKILLS_CERT_API_KEY} ${'tail'.repeat(100)}`;

    const summary = buildCertificationSummary({
      ...passingSummaryInput(),
      failures: [{ category: 'webui_workflow', detail, surface: 'webui' }],
    });
    const retainedDetail = summary.failures[0].detail;

    expect(retainedDetail).toContain('[REDACTED]');
    expect(retainedDetail).not.toContain(SKILLS_CERT_API_KEY);
    expect(retainedDetail).toHaveLength(500);
  });
});

describe('Skills certification evidence finalization', () => {
  it('deletes runtime, compacts logs, writes summary, then scans with the exact key', async () => {
    const fixture = createFixture('ordered-finalization');
    const logPath = path.join(fixture.evidence.logsDir, 'backend.log');
    const oversizedLog = `HEAD\n${'x'.repeat(maxLogBytes + 4096)}\nTAIL`;
    writeFileSync(logPath, oversizedLog, 'utf8');
    const scanArtifacts = vi.fn((root, options) => {
      expect(existsSync(fixture.runtime.root)).toBe(false);
      expect(statSync(logPath).size).toBeLessThanOrEqual(maxLogBytes);
      expect(existsSync(fixture.evidence.summaryPath)).toBe(true);
      assertNoSecretLeaks(root, options);
    });

    const summary = await finalizeSkillsCertificationEvidence({
      ...finalizationOptions(fixture),
      scanArtifacts,
    });

    expect(scanArtifacts).toHaveBeenCalledWith(fixture.evidence.root, {
      additionalSecrets: [SKILLS_CERT_API_KEY],
    });
    expect(summary.status).toBe('passed');
  });

  it('finalizes and scans evidence when profile creation retained no runtime', async () => {
    const fixture = createFixture('no-runtime');
    const scanArtifacts = vi.fn();

    const summary = await finalizeEvidence({
      evidence: fixture.evidence,
      runtime: undefined,
      scanArtifacts,
      summaryInput: passingSummaryInput(fixture.evidence.runId),
      teardownOutcome: { status: 'fulfilled', value: undefined },
    });

    expect(scanArtifacts).toHaveBeenCalledWith(fixture.evidence.root, {
      additionalSecrets: [SKILLS_CERT_API_KEY],
    });
    expect(summary.cleanup.runtime_deleted).toBe(true);
    expect(existsSync(fixture.evidence.summaryPath)).toBe(true);
  });

  it('deterministically compacts every retained log to bounded head and tail', async () => {
    const fixture = createFixture('bounded-logs');
    const firstLog = path.join(fixture.evidence.logsDir, 'first.log');
    const secondLog = path.join(fixture.evidence.webuiDir, 'second.log');
    const source = `HEAD-MARKER\n${SKILLS_CERT_API_KEY}\n${'z'.repeat(
      maxLogBytes + 4096
    )}\nTAIL-MARKER`;
    writeFileSync(firstLog, source, 'utf8');
    writeFileSync(secondLog, source, 'utf8');

    await finalizeSkillsCertificationEvidence(finalizationOptions(fixture));

    const first = readFileSync(firstLog, 'utf8');
    const second = readFileSync(secondLog, 'utf8');
    expect(Buffer.byteLength(first)).toBeLessThanOrEqual(maxLogBytes);
    expect(first).toBe(second);
    expect(first).toMatch(/^HEAD-MARKER/);
    expect(first).toMatch(/skills certification log truncated/i);
    expect(first).toMatch(/TAIL-MARKER$/);
    expect(first).toContain('[REDACTED]');
    expect(first).not.toContain(SKILLS_CERT_API_KEY);
  });

  it('never truncates retained JSON reports', async () => {
    const fixture = createFixture('json-not-truncated');
    const reportPath = path.join(fixture.evidence.webuiDir, 'report.json');
    const payload = 'r'.repeat(maxLogBytes + 2048);
    writeSanitizedJson(reportPath, { payload });

    await finalizeSkillsCertificationEvidence(finalizationOptions(fixture));

    expect(JSON.parse(readFileSync(reportPath, 'utf8')).payload).toBe(payload);
    expect(statSync(reportPath).size).toBeGreaterThan(maxLogBytes);
  });

  it('retains teardown failure in the sanitized summary and final status', async () => {
    const fixture = createFixture('teardown-failure');

    const summary = await finalizeEvidence({
      ...finalizationOptions(fixture),
      teardownOutcome: {
        reason: new Error(`teardown leaked ${SKILLS_CERT_API_KEY}`),
        status: 'rejected',
      },
    });
    const retained = JSON.parse(readFileSync(fixture.evidence.summaryPath, 'utf8'));

    expect(summary).toEqual(retained);
    expect(summary.status).toBe('failed');
    expect(summary.failures).toContainEqual({ category: 'cleanup' });
    expect(summary.cleanup).toEqual({
      children_closed: false,
      runtime_deleted: true,
    });
    expect(JSON.stringify(retained)).not.toContain(SKILLS_CERT_API_KEY);
  });

  it('refuses runtime deletion outside its expected base', async () => {
    const fixture = createFixture('unsafe-runtime-root');
    const nestedRoot = path.join(fixture.runtime.baseRoot, 'nested', 'runtime');
    const markerPath = path.join(nestedRoot, '.skills-certification-runtime');
    mkdirSync(nestedRoot, { recursive: true });
    writeFileSync(markerPath, 'tldw-skills-certification-runtime\n', 'utf8');

    const summary = await finalizeEvidence({
      ...finalizationOptions(fixture),
      runtime: {
        ...fixture.runtime,
        markerPath,
        root: nestedRoot,
      },
    });

    expect(existsSync(nestedRoot)).toBe(true);
    expect(summary.status).toBe('failed');
    expect(summary.cleanup.runtime_deleted).toBe(false);
    expect(summary.failures).toContainEqual({ category: 'cleanup' });
  });

  it('refuses runtime deletion without the exact marker', async () => {
    const fixture = createFixture('wrong-runtime-marker');
    writeFileSync(fixture.runtime.markerPath, 'wrong marker\n', 'utf8');

    const summary = await finalizeSkillsCertificationEvidence(finalizationOptions(fixture));

    expect(existsSync(fixture.runtime.root)).toBe(true);
    expect(summary.cleanup.runtime_deleted).toBe(false);
    expect(summary.failures).toContainEqual({ category: 'cleanup' });
  });

  it('detects the exact synthetic key and removes all contaminated evidence', async () => {
    const fixture = createFixture('contaminated-evidence');
    const contamination = path.join(fixture.evidence.webuiDir, 'diagnostics.txt');
    writeFileSync(contamination, `leaked ${SKILLS_CERT_API_KEY}`, 'utf8');

    const summary = await finalizeSkillsCertificationEvidence(finalizationOptions(fixture));

    expect(summary.status).toBe('failed');
    expect(summary.artifact_safety.passed).toBe(false);
    expect(summary.failures).toContainEqual({ category: 'artifact_safety' });
    expect(existsSync(fixture.evidence.root)).toBe(false);
    expect(existsSync(fixture.evidence.summaryPath)).toBe(false);
  });

  it('returns artifact failure when contaminated evidence cannot be safely removed', async () => {
    const fixture = createFixture('failed-contamination-removal');
    writeFileSync(fixture.evidence.markerPath, 'wrong marker\n', 'utf8');
    writeFileSync(
      path.join(fixture.evidence.webuiDir, 'diagnostics.txt'),
      `leaked ${SKILLS_CERT_API_KEY}`,
      'utf8'
    );

    const summary = await finalizeSkillsCertificationEvidence(finalizationOptions(fixture));
    const retained = JSON.parse(readFileSync(fixture.evidence.summaryPath, 'utf8'));

    expect(summary.status).toBe('failed');
    expect(summary.artifact_safety.passed).toBe(false);
    expect(summary.failures).toContainEqual({ category: 'artifact_safety' });
    expect(retained).toEqual(summary);
    expect(retained.status).toBe('failed');
    expect(retained.artifact_safety.passed).toBe(false);
    expect(existsSync(fixture.evidence.root)).toBe(true);
  });
});
