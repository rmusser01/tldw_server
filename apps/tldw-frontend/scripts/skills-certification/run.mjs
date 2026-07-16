import { existsSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { redactText } from '../onboarding-uat/artifacts.mjs';
import { waitForHttpOk } from '../onboarding-uat/processes.mjs';
import { reservePorts } from '../onboarding-uat/ports.mjs';
import {
  buildCertificationSummary,
  createSkillsCertificationEvidence,
  finalizeSkillsCertificationEvidence,
  removeSkillsCertificationEvidence,
  removeSkillsCertificationRuntime,
} from './evidence.mjs';
import { createProcessRegistry, installCertificationSignalHandlers } from './lifecycle.mjs';
import {
  SKILLS_CERT_API_KEY,
  SKILLS_CERT_NAMES,
  buildSkillsCertificationCommands,
  createSkillsCertificationProfile,
  isConfirmedBindConflict,
} from './profile.mjs';

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const repoRoot = path.resolve(frontendRoot, '../..');
const failureDetailLimit = 500;
const reportStats = ['expected', 'skipped', 'flaky', 'unexpected'];

function boundedDetail(value) {
  return redactText(String(value ?? 'unknown failure')).slice(0, failureDetailLimit);
}

/** Format the only diagnostic emitted by the standalone certification command. */
export function formatSkillsCertificationDiagnostic(summary) {
  if (summary?.artifact_safety?.passed === false) return 'artifact_safety';
  return boundedDetail(summary?.primary_category ?? 'certification_failed');
}

function defaultReadJson(filePath) {
  if (!existsSync(filePath)) return undefined;
  try {
    return JSON.parse(readFileSync(filePath, 'utf8'));
  } catch {
    return undefined;
  }
}

function defaultReadText(filePath) {
  return existsSync(filePath) ? readFileSync(filePath, 'utf8') : '';
}

function defaultRunChild(registry, command, logPath) {
  const record = registry.spawn(command, logPath);
  return registry.wait(record);
}

function commandWithEvidence(command, extraEnv) {
  return { ...command, env: { ...command.env, ...extraEnv } };
}

function exactReport(report) {
  const stats = report?.stats;
  if (
    !stats ||
    typeof stats !== 'object' ||
    reportStats.some((key) => !Number.isInteger(stats[key]))
  ) {
    return false;
  }
  return stats.expected === 1 && stats.skipped === 0 && stats.flaky === 0 && stats.unexpected === 0;
}

function resultCategories(result, allowed) {
  if (
    !result ||
    typeof result !== 'object' ||
    !['passed', 'failed', 'running'].includes(result.status)
  ) {
    return null;
  }
  if (
    !Array.isArray(result.categories) ||
    result.categories.some((category) => !allowed.has(category))
  ) {
    return null;
  }
  return result.categories;
}

function apiUrl(backendUrl, pathname) {
  return new URL(pathname, backendUrl).toString();
}

async function responseJson(response) {
  try {
    return await response.json();
  } catch {
    return undefined;
  }
}

/** Execute the fixed local Skills certification gate. */
export async function runSkillsCertification({ operations: suppliedOperations = {} } = {}) {
  const operations = {
    buildCommands: buildSkillsCertificationCommands,
    createEvidence: createSkillsCertificationEvidence,
    createProfile: createSkillsCertificationProfile,
    createRegistry: createProcessRegistry,
    fetch: globalThis.fetch,
    finalize: finalizeSkillsCertificationEvidence,
    installHandlers: installCertificationSignalHandlers,
    isBindConflict: isConfirmedBindConflict,
    readJson: defaultReadJson,
    readText: defaultReadText,
    reservePorts,
    removeEvidence: removeSkillsCertificationEvidence,
    removeRuntime: removeSkillsCertificationRuntime,
    runChild: defaultRunChild,
    startChild: (registry, command, logPath) => registry.spawn(command, logPath),
    stopChild: (registry, record) => registry.stop(record),
    waitForHttpOk,
    ...suppliedOperations,
  };
  const failures = [];
  const surfaces = {
    extension: { postcondition: false, state: 'not_run_infrastructure' },
    webui: { postcondition: false, state: 'not_run_infrastructure' },
  };
  let evidence;
  let profile;
  let registry;
  let removeSignalHandlers = () => undefined;
  let interrupted = false;
  let urlLocked = false;
  let backendUrl;
  let commands;
  let backendUsable = false;
  let backendRecord;
  let webuiAttempted = false;
  let webReady = false;
  let finalSummary;

  const fail = (category, detail, surface) => {
    const failure = { category };
    if (surface) failure.surface = surface;
    if (detail) failure.detail = boundedDetail(detail);
    failures.push(failure);
  };
  const logPath = (name) => path.join(evidence.logsDir, `${name}.log`);
  const diagnostic = (error, filePath) => {
    let logText = '';
    try {
      logText = operations.readText(filePath);
    } catch {
      // Logging diagnostics must not change the failing phase classification.
    }
    const message = boundedDetail(error?.message ?? '').slice(0, 200);
    const tail = redactText(String(logText ?? '')).slice(-300);
    return boundedDetail(`${message} ${tail}`);
  };
  const runFinite = async (command) =>
    operations.runChild(registry, command, logPath(command.name));
  const health = async () =>
    operations.waitForHttpOk(apiUrl(backendUrl, '/api/v1/health'), {
      headers: { 'X-API-KEY': SKILLS_CERT_API_KEY },
    });
  const directInitial = async () => {
    for (const [route, invariant] of [
      ['/api/v1/skills/?limit=1&offset=0', (body) => body?.total === 0],
      ['/api/v1/skills/trash?limit=1&offset=0', (body) => body?.total === 0],
    ]) {
      try {
        const response = await operations.fetch(apiUrl(backendUrl, route), {
          headers: { 'X-API-KEY': SKILLS_CERT_API_KEY },
        });
        if (response.status !== 200 || !invariant(await responseJson(response))) {
          fail('postcondition', `${route} status/invariant`);
        }
      } catch {
        fail('postcondition', `${route} status/invariant`);
      }
    }
  };
  const directPostcondition = async (name, surface) => {
    if (!backendUsable) return;
    let passed = true;
    const detailRoute = `/api/v1/skills/${encodeURIComponent(name)}`;
    try {
      const detail = await operations.fetch(apiUrl(backendUrl, detailRoute), {
        headers: { 'X-API-KEY': SKILLS_CERT_API_KEY },
      });
      if (detail.status !== 404) {
        passed = false;
        fail('postcondition', `${detailRoute} status ${detail.status}`, surface);
      }
    } catch {
      passed = false;
      fail('postcondition', `${detailRoute} status/invariant`, surface);
    }
    const trashRoute = '/api/v1/skills/trash?limit=500&offset=0';
    try {
      const trash = await operations.fetch(apiUrl(backendUrl, trashRoute), {
        headers: { 'X-API-KEY': SKILLS_CERT_API_KEY },
      });
      const body = await responseJson(trash);
      if (
        trash.status !== 200 ||
        !Array.isArray(body?.skills) ||
        body.skills.some((skill) => skill?.name === name)
      ) {
        passed = false;
        fail('postcondition', `${trashRoute} status/invariant`, surface);
      }
    } catch {
      passed = false;
      fail('postcondition', `${trashRoute} status/invariant`, surface);
    }
    surfaces[surface].postcondition = passed;
  };

  try {
    registry = operations.createRegistry();
    removeSignalHandlers = operations.installHandlers({
      registry,
      onSignal: () => {
        interrupted = true;
      },
    });
    evidence = operations.createEvidence({ frontendRoot });
    try {
      profile = operations.createProfile({ repoRoot, temporaryBase: tmpdir() });
    } catch (error) {
      profile = error?.runtime;
      throw error;
    }

    for (const key of ['webuiChromiumProbe', 'extensionChromiumProbe']) {
      const probeCommands = operations.buildCommands({
        repoRoot,
        frontendRoot,
        profile,
        ports: { backend: 1, web: 1 },
      });
      const outcome = await runFinite(probeCommands[key]);
      if (outcome?.code !== 0 || outcome?.signal) fail('preflight', `${key} exited`);
    }
    if (!failures.some((failure) => failure.category === 'preflight')) {
      const authCommands = operations.buildCommands({
        repoRoot,
        frontendRoot,
        profile,
        ports: { backend: 1, web: 1 },
      });
      const outcome = await runFinite(authCommands.authInit);
      if (outcome?.code !== 0 || outcome?.signal) fail('preflight', 'auth-init exited');
    }

    if (!failures.some((failure) => failure.category === 'preflight')) {
      for (let attempt = 1; attempt <= 3 && !webReady; attempt += 1) {
        if (urlLocked)
          throw new Error('Certification URLs are immutable after browser execution starts');
        const ports = await operations.reservePorts(['backend', 'web']);
        backendUrl = `http://127.0.0.1:${ports.backend}`;
        commands = operations.buildCommands({ repoRoot, frontendRoot, profile, ports });
        let frontendRecord;
        const backendLogPath = logPath(`backend-attempt-${attempt}`);
        const frontendLogPath = logPath(`frontend-attempt-${attempt}`);
        try {
          backendRecord = await operations.startChild(registry, commands.backend, backendLogPath);
        } catch (error) {
          const detail = diagnostic(error, backendLogPath);
          if (operations.isBindConflict(detail) && attempt < 3 && !urlLocked) continue;
          fail('backend_startup', 'backend startup failed');
          break;
        }
        try {
          await health();
          backendUsable = true;
          try {
            frontendRecord = await operations.startChild(
              registry,
              commands.frontend,
              frontendLogPath
            );
            await operations.waitForHttpOk(`http://127.0.0.1:${ports.web}`);
            webReady = true;
          } catch (error) {
            const detail = diagnostic(error, frontendLogPath);
            if (operations.isBindConflict(detail) && attempt < 3 && !urlLocked) {
              let retryCleanupFailed = false;
              if (frontendRecord) {
                try {
                  await operations.stopChild(registry, frontendRecord);
                } catch {
                  retryCleanupFailed = true;
                }
              }
              if (!retryCleanupFailed && backendRecord) {
                try {
                  await operations.stopChild(registry, backendRecord);
                } catch {
                  retryCleanupFailed = true;
                }
              }
              if (!retryCleanupFailed) {
                backendRecord = undefined;
                backendUsable = false;
                continue;
              }
              fail('cleanup', 'frontend retry cleanup failed');
              fail('webui_startup', 'frontend startup failed', 'webui');
              try {
                await health();
                backendUsable = true;
              } catch {
                backendUsable = false;
                fail('backend_health', 'backend health failed');
              }
              break;
            }
            fail('webui_startup', 'frontend startup failed', 'webui');
            break;
          }
        } catch (error) {
          const detail = diagnostic(error, backendLogPath);
          if (operations.isBindConflict(detail) && attempt < 3 && !urlLocked) {
            try {
              await operations.stopChild(registry, backendRecord);
            } catch {
              fail('cleanup', 'backend health retry cleanup failed');
              try {
                await health();
                backendUsable = true;
              } catch {
                backendUsable = false;
                fail('backend_health', 'backend health failed');
              }
              break;
            }
            backendRecord = undefined;
            backendUsable = false;
            continue;
          }
          fail('backend_health', 'backend health failed');
          break;
        }
      }
      if (!backendUsable && !failures.some((failure) => failure.category === 'backend_health')) {
        fail('backend_startup', 'backend did not start');
      }
    }

    if (backendUsable) await directInitial();
    if (backendUsable && webReady) {
      urlLocked = true;
      webuiAttempted = true;
      const webResultPath = path.join(evidence.webuiDir, 'result.json');
      const webReportPath = path.join(evidence.webuiDir, 'report.json');
      let outcome;
      let surfaceResult;
      let surfaceReport;
      let browserThrew = false;
      try {
        outcome = await runFinite(
          commandWithEvidence(commands.webuiPlaywright, {
            TLDW_SKILLS_CERT_WEB_OUTPUT: path.join(evidence.webuiDir, 'output'),
            TLDW_SKILLS_CERT_WEB_REPORT: webReportPath,
            TLDW_SKILLS_CERT_WEB_RESULT: webResultPath,
          })
        );
      } catch {
        browserThrew = true;
      }
      try {
        surfaceResult = operations.readJson(webResultPath);
      } catch {
        surfaceResult = undefined;
      }
      try {
        surfaceReport = operations.readJson(webReportPath);
      } catch {
        surfaceReport = undefined;
      }
      const categories = resultCategories(surfaceResult, new Set(['webui_workflow']));
      const passed =
        outcome?.code === 0 &&
        exactReport(surfaceReport) &&
        surfaceResult?.status === 'passed' &&
        categories?.length === 0;
      if (passed) surfaces.webui.state = 'passed';
      else {
        surfaces.webui.state = 'failed';
        fail(
          (browserThrew || outcome?.code !== 0) && surfaceResult === undefined
            ? 'webui_launch'
            : 'webui_workflow',
          'WebUI report/result failed',
          'webui'
        );
      }
      await directPostcondition(SKILLS_CERT_NAMES.webui, 'webui');
    }

    if (backendUsable && webuiAttempted) {
      try {
        await health();
      } catch {
        fail('backend_health', 'backend crashed after WebUI');
        try {
          await operations.stopChild(registry, backendRecord);
          backendRecord = await operations.startChild(
            registry,
            commands.backend,
            logPath('backend-restart')
          );
          await health();
        } catch {
          backendUsable = false;
        }
      }
    }

    if (backendUsable) {
      let build;
      try {
        build = await runFinite(commands.extensionBuild);
      } catch {
        build = { code: 1 };
      }
      if (build?.code !== 0 || build?.signal) {
        surfaces.extension.state = 'failed';
        fail('extension_build', 'extension build failed', 'extension');
      } else {
        try {
          await health();
        } catch {
          backendUsable = false;
          surfaces.extension.state = 'not_run_infrastructure';
          fail('backend_health', 'backend unavailable before extension');
        }
        if (backendUsable) {
          urlLocked = true;
          const resultPath = path.join(evidence.extensionDir, 'result.json');
          const reportPath = path.join(evidence.extensionDir, 'report.json');
          let outcome;
          let browserThrew = false;
          try {
            outcome = await runFinite(
              commandWithEvidence(commands.extensionPlaywright, {
                TLDW_SKILLS_CERT_EXTENSION_LEDGER: evidence.relayLedgerPath,
                TLDW_SKILLS_CERT_EXTENSION_OUTPUT: path.join(evidence.extensionDir, 'output'),
                TLDW_SKILLS_CERT_EXTENSION_REPORT: reportPath,
                TLDW_SKILLS_CERT_EXTENSION_RESULT: resultPath,
              })
            );
          } catch {
            browserThrew = true;
          }
          let surfaceResult;
          let surfaceReport;
          try {
            surfaceResult = operations.readJson(resultPath);
          } catch {
            surfaceResult = undefined;
          }
          try {
            surfaceReport = operations.readJson(reportPath);
          } catch {
            surfaceReport = undefined;
          }
          const categories = resultCategories(
            surfaceResult,
            new Set([
              'extension_launch',
              'extension_worker',
              'extension_workflow',
              'extension_relay',
            ])
          );
          const passed =
            outcome?.code === 0 &&
            exactReport(surfaceReport) &&
            surfaceResult?.status === 'passed' &&
            categories?.length === 0;
          if (passed) surfaces.extension.state = 'passed';
          else {
            surfaces.extension.state = 'failed';
            if (categories?.length) {
              categories.forEach((category) =>
                fail(category, 'Extension result failed', 'extension')
              );
            } else {
              fail(
                (browserThrew || outcome?.code !== 0) && surfaceResult === undefined
                  ? 'extension_launch'
                  : 'extension_workflow',
                'Extension report/result failed',
                'extension'
              );
            }
          }
          await directPostcondition(SKILLS_CERT_NAMES.extension, 'extension');
        }
      }
    } else {
      fail('backend_health', 'backend unavailable before extension');
    }
  } catch (error) {
    fail('preflight', error?.message ?? error);
  } finally {
    const teardownOutcome = await Promise.resolve(registry?.teardown?.()).then(
      (value) => ({ status: 'fulfilled', value }),
      (reason) => ({ status: 'rejected', reason })
    );
    if (interrupted) fail('interrupted', 'signal received');
    if (teardownOutcome.status === 'rejected') fail('cleanup', 'process teardown failed');
    const summaryInput = { failures, surfaces };
    const finalizerFailure = () => {
      let runtimeDeleted = !profile;
      try {
        runtimeDeleted = operations.removeRuntime(profile) === true;
      } catch {
        runtimeDeleted = false;
      }
      try {
        operations.removeEvidence(evidence);
      } catch {
        // Marker-safe removal failure remains artifact unsafe.
      }
      return buildCertificationSummary({
        failures,
        surfaces,
        artifact_safety: { passed: false },
        cleanup: {
          children_closed: teardownOutcome.status === 'fulfilled',
          runtime_deleted: runtimeDeleted,
        },
      });
    };
    if (evidence) {
      try {
        finalSummary = await operations.finalize({
          evidence,
          runtime: profile,
          summaryInput,
          teardownOutcome,
        });
      } catch {
        finalSummary = finalizerFailure();
      }
    } else {
      finalSummary = buildCertificationSummary({
        ...summaryInput,
        artifact_safety: { passed: false },
        cleanup: { children_closed: false, runtime_deleted: false },
      });
    }
    if (
      interrupted &&
      !finalSummary.failures?.some((failure) => failure.category === 'interrupted')
    ) {
      fail('interrupted', 'signal received');
      if (evidence && finalSummary?.artifact_safety?.passed === true) {
        try {
          finalSummary = await operations.finalize({
            evidence,
            runtime: profile,
            summaryInput: { failures, surfaces },
            teardownOutcome,
          });
        } catch {
          finalSummary = finalizerFailure();
        }
      } else {
        finalSummary = buildCertificationSummary({
          failures,
          surfaces,
          artifact_safety: finalSummary?.artifact_safety,
          cleanup: finalSummary?.cleanup,
        });
      }
    }
    try {
      removeSignalHandlers();
    } catch {
      fail('cleanup', 'signal handler cleanup failed');
      if (evidence && finalSummary?.artifact_safety?.passed === true) {
        try {
          finalSummary = await operations.finalize({
            evidence,
            runtime: profile,
            summaryInput: { failures, surfaces },
            teardownOutcome,
          });
        } catch {
          finalSummary = finalizerFailure();
        }
      } else
        finalSummary = buildCertificationSummary({
          failures,
          surfaces,
          artifact_safety: finalSummary?.artifact_safety,
          cleanup: finalSummary?.cleanup,
        });
      try {
        removeSignalHandlers();
      } catch {
        // A single retry handles a partially removed listener pair without looping.
      }
    }
  }
  return finalSummary;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  runSkillsCertification().then(
    (summary) => {
      process.exitCode = summary.status === 'passed' ? 0 : 1;
      if (summary.status !== 'passed') {
        process.stderr.write(`${formatSkillsCertificationDiagnostic(summary)}\n`);
      }
    },
    (error) => {
      process.stderr.write(`${boundedDetail(error)}\n`);
      process.exitCode = 1;
    }
  );
}
