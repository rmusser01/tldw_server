import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  assertNoSecretLeaks as onboardingAssertNoSecretLeaks,
  redactText,
} from '../onboarding-uat/artifacts.mjs';
import { SKILLS_CERT_API_KEY } from './profile.mjs';

const moduleDir = path.dirname(fileURLToPath(import.meta.url));
const defaultFrontendRoot = path.resolve(moduleDir, '../..');
const evidenceMarkerName = '.skills-certification-evidence';
const evidenceMarkerContent = 'tldw-skills-certification-evidence\n';
const runtimeMarkerName = '.skills-certification-runtime';
const runtimeMarkerContent = 'tldw-skills-certification-runtime\n';
const maxLogBytes = 1024 * 1024;
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
const surfaceStates = new Set(['passed', 'failed', 'not_run_infrastructure']);

function createRunId() {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  return `${stamp}-${Math.random().toString(36).slice(2, 8)}`;
}

function assertRunId(runId) {
  if (
    typeof runId !== 'string' ||
    !runId ||
    runId === '.' ||
    runId === '..' ||
    path.basename(runId) !== runId
  ) {
    throw new Error(`Invalid Skills certification run id: ${runId}`);
  }
}

function sanitizedJsonValue(value) {
  const serialized = JSON.stringify(value, null, 2);
  if (serialized === undefined) {
    throw new Error('Cannot write an undefined JSON value');
  }
  const text = redactText(serialized);
  return {
    text,
    value: JSON.parse(text),
  };
}

/** Create the fixed retained-evidence layout for one certification run. */
export function createSkillsCertificationEvidence({
  frontendRoot = defaultFrontendRoot,
  runId = createRunId(),
} = {}) {
  assertRunId(runId);
  const resolvedFrontendRoot = path.resolve(frontendRoot);
  const baseRoot = path.join(resolvedFrontendRoot, 'test-results/skills-certification');
  const root = path.join(baseRoot, runId);
  const logsDir = path.join(root, 'logs');
  const webuiDir = path.join(root, 'webui');
  const extensionDir = path.join(root, 'extension');
  const markerPath = path.join(root, evidenceMarkerName);

  mkdirSync(baseRoot, { recursive: true });
  mkdirSync(root);
  mkdirSync(logsDir);
  mkdirSync(webuiDir);
  mkdirSync(extensionDir);
  writeFileSync(markerPath, evidenceMarkerContent, { encoding: 'utf8', flag: 'wx' });

  return {
    baseRoot,
    extensionDir,
    frontendRoot: resolvedFrontendRoot,
    logsDir,
    markerPath,
    relayLedgerPath: path.join(extensionDir, 'relay-ledger.json'),
    root,
    runId,
    summaryPath: path.join(root, 'summary.json'),
    webuiDir,
  };
}

/** Write complete JSON after applying the shared onboarding redactor. */
export function writeSanitizedJson(filePath, value) {
  const sanitized = sanitizedJsonValue(value);
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, `${sanitized.text}\n`, 'utf8');
  return sanitized.value;
}

function normalizeFailure(failure) {
  const normalized = { category: String(failure?.category ?? '') };
  if (failure?.surface === 'webui' || failure?.surface === 'extension') {
    normalized.surface = failure.surface;
  }
  if (typeof failure?.detail === 'string') {
    normalized.detail = failure.detail;
  }
  return normalized;
}

function normalizeSurface(value) {
  return {
    state: surfaceStates.has(value?.state) ? value.state : 'not_run_infrastructure',
    postcondition: value?.postcondition === true,
  };
}

/** Aggregate all phase, cleanup, and artifact outcomes into the fixed summary. */
export function buildCertificationSummary(input = {}) {
  const cleanup = {
    children_closed: input.cleanup?.children_closed === true,
    runtime_deleted: input.cleanup?.runtime_deleted === true,
  };
  const artifactSafety = {
    passed: input.artifact_safety?.passed === true,
  };
  const failures = (input.failures ?? []).map(normalizeFailure);

  if (!artifactSafety.passed && !failures.some(({ category }) => category === 'artifact_safety')) {
    failures.push({ category: 'artifact_safety' });
  }
  if (
    (!cleanup.children_closed || !cleanup.runtime_deleted) &&
    !failures.some(({ category }) => category === 'cleanup')
  ) {
    failures.push({ category: 'cleanup' });
  }

  let primaryCategory = null;
  if (failures.some(({ category }) => category === 'interrupted')) {
    primaryCategory = 'interrupted';
  } else {
    primaryCategory =
      primaryCategoryOrder.find((category) =>
        failures.some((failure) => failure.category === category)
      ) ??
      failures[0]?.category ??
      null;
  }

  return {
    run_id: String(input.run_id ?? ''),
    status: failures.length === 0 ? 'passed' : 'failed',
    primary_category: primaryCategory,
    failures,
    surfaces: {
      webui: normalizeSurface(input.surfaces?.webui),
      extension: normalizeSurface(input.surfaces?.extension),
    },
    cleanup,
    artifact_safety: artifactSafety,
  };
}

function compareNames(left, right) {
  if (left.name < right.name) {
    return -1;
  }
  return left.name > right.name ? 1 : 0;
}

function collectLogFiles(root) {
  const files = [];
  for (const entry of readdirSync(root, { withFileTypes: true }).sort(compareNames)) {
    const fullPath = path.join(root, entry.name);
    if (entry.isDirectory()) {
      files.push(...collectLogFiles(fullPath));
    } else if (entry.isFile() && path.extname(entry.name) === '.log') {
      files.push(fullPath);
    }
  }
  return files;
}

function utf8Head(buffer, byteLimit) {
  let end = Math.min(buffer.length, byteLimit);
  while (end > 0 && end < buffer.length && (buffer[end] & 0xc0) === 0x80) {
    end -= 1;
  }
  return buffer.subarray(0, end);
}

function utf8Tail(buffer, byteLimit) {
  let start = Math.max(0, buffer.length - byteLimit);
  while (start < buffer.length && (buffer[start] & 0xc0) === 0x80) {
    start += 1;
  }
  return buffer.subarray(start);
}

function compactLog(filePath) {
  const sanitized = Buffer.from(redactText(readFileSync(filePath, 'utf8')), 'utf8');
  if (sanitized.length <= maxLogBytes) {
    writeFileSync(filePath, sanitized);
    return;
  }

  const marker = Buffer.from(
    `\n[skills certification log truncated; original_bytes=${sanitized.length}]\n`,
    'utf8'
  );
  const retainedBytes = maxLogBytes - marker.length;
  const head = utf8Head(sanitized, Math.floor(retainedBytes / 2));
  const tail = utf8Tail(sanitized, retainedBytes - Math.floor(retainedBytes / 2));
  writeFileSync(filePath, Buffer.concat([head, marker, tail]));
}

function compactRetainedLogs(root) {
  for (const filePath of collectLogFiles(root)) {
    compactLog(filePath);
  }
}

function assertMarker(markerPath, root, markerName, markerContent, label) {
  if (path.resolve(markerPath) !== path.join(root, markerName)) {
    throw new Error(`Refusing to remove ${label} root without exact marker path: ${root}`);
  }
  if (!existsSync(markerPath) || readFileSync(markerPath, 'utf8') !== markerContent) {
    throw new Error(`Refusing to remove ${label} root without exact marker: ${root}`);
  }
}

function removeRuntimeRoot(runtime) {
  const root = path.resolve(runtime.root);
  const baseRoot = path.resolve(runtime.baseRoot);
  if (path.dirname(root) !== baseRoot) {
    throw new Error(`Refusing to remove runtime root outside expected base: ${root}`);
  }
  assertMarker(runtime.markerPath, root, runtimeMarkerName, runtimeMarkerContent, 'runtime');
  rmSync(root, { recursive: true });
  if (existsSync(root)) {
    throw new Error(`Runtime root still exists after removal: ${root}`);
  }
}

function removeEvidenceRoot(evidence) {
  const root = path.resolve(evidence.root);
  const expectedBaseRoot = path.join(
    path.resolve(evidence.frontendRoot),
    'test-results/skills-certification'
  );
  if (
    path.resolve(evidence.baseRoot) !== expectedBaseRoot ||
    path.dirname(root) !== expectedBaseRoot ||
    path.basename(root) !== evidence.runId
  ) {
    throw new Error(`Refusing to remove evidence root outside expected base: ${root}`);
  }
  assertMarker(evidence.markerPath, root, evidenceMarkerName, evidenceMarkerContent, 'evidence');
  rmSync(root, { recursive: true });
  if (existsSync(root)) {
    throw new Error(`Evidence root still exists after removal: ${root}`);
  }
}

function finalSummaryInput(summaryInput, evidence, childrenClosed, runtimeDeleted, artifactPassed) {
  return {
    ...summaryInput,
    run_id: evidence.runId,
    cleanup: {
      children_closed: childrenClosed,
      runtime_deleted: runtimeDeleted,
    },
    artifact_safety: { passed: artifactPassed },
  };
}

/** Finalize cleanup and retained evidence in the required fixed order. */
export async function finalizeSkillsCertificationEvidence({
  evidence,
  runtime,
  scanArtifacts = onboardingAssertNoSecretLeaks,
  summaryInput = {},
  syntheticKey = SKILLS_CERT_API_KEY,
  teardownOutcome,
} = {}) {
  const childrenClosed = teardownOutcome?.status === 'fulfilled';
  let runtimeDeleted = false;
  let artifactPassed = summaryInput.artifact_safety?.passed !== false;

  try {
    removeRuntimeRoot(runtime);
    runtimeDeleted = true;
  } catch {
    runtimeDeleted = false;
  }

  try {
    compactRetainedLogs(evidence.root);
  } catch {
    artifactPassed = false;
  }

  let summary = buildCertificationSummary(
    finalSummaryInput(summaryInput, evidence, childrenClosed, runtimeDeleted, artifactPassed)
  );
  try {
    summary = writeSanitizedJson(evidence.summaryPath, summary);
  } catch {
    artifactPassed = false;
  }

  try {
    await scanArtifacts(evidence.root, { additionalSecrets: [syntheticKey] });
  } catch {
    artifactPassed = false;
  }

  if (!artifactPassed) {
    summary = sanitizedJsonValue(
      buildCertificationSummary(
        finalSummaryInput(summaryInput, evidence, childrenClosed, runtimeDeleted, false)
      )
    ).value;
    try {
      removeEvidenceRoot(evidence);
    } catch {
      // The failing in-memory summary is returned without bypassing marker safety.
    }
  }

  return summary;
}
