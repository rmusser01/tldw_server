import {
  spawnLoggedProcess as onboardingSpawnLoggedProcess,
  stopProcessTree as onboardingStopProcessTree,
} from '../onboarding-uat/processes.mjs';

function defaultProbeProcessTree(target) {
  try {
    process.kill(target, 0);
    return true;
  } catch (error) {
    if (error?.code === 'ESRCH') {
      return false;
    }
    throw error;
  }
}

function withTimeout(value, timeoutMs, message) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(message)), timeoutMs);
    Promise.resolve(value).then(
      (result) => {
        clearTimeout(timer);
        resolve(result);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      }
    );
  });
}

function childFromRecord(record) {
  return record?.child ?? record;
}

/** Track every spawned command until its child emits close. */
export function createProcessRegistry({
  closeTimeoutMs = 10_000,
  platform = process.platform,
  probeProcessTree = defaultProbeProcessTree,
  probeTimeoutMs = 1_000,
  spawnLoggedProcess = onboardingSpawnLoggedProcess,
  stopProcessTree = onboardingStopProcessTree,
  stopTimeoutMs = 5_000,
} = {}) {
  const registered = [];
  const states = new WeakMap();
  let teardownPromise;

  function spawn(command, logPath) {
    if (teardownPromise) {
      throw new Error('Cannot spawn after process teardown has started');
    }

    const record = spawnLoggedProcess({ ...command, logPath });
    const child = childFromRecord(record);
    if (!child || typeof child.once !== 'function') {
      throw new Error('spawnLoggedProcess must return a child process record');
    }

    let resolveClose;
    const closePromise = new Promise((resolve) => {
      resolveClose = resolve;
    });
    states.set(record, { closePromise });
    registered.push(record);
    child.once('close', (code, signal) => {
      resolveClose({ code, signal });
    });

    return record;
  }

  function wait(record) {
    const state = states.get(record);
    if (!state) {
      throw new Error('Cannot wait for an unregistered process record');
    }
    return state.closePromise;
  }

  async function teardownRegisteredProcesses() {
    const errors = [];
    const records = [...registered].reverse();

    for (const record of records) {
      const child = childFromRecord(record);
      const state = states.get(record);

      try {
        await stopProcessTree(record, { timeoutMs: stopTimeoutMs });
      } catch (error) {
        errors.push(error);
      }

      try {
        await withTimeout(
          state.closePromise,
          closeTimeoutMs,
          `Timed out waiting for process ${child?.pid ?? 'unknown'} to close`
        );
      } catch (error) {
        errors.push(error);
      }

      if (!Number.isInteger(child?.pid) || child.pid <= 0) {
        errors.push(new Error('Cannot verify a process without a positive PID'));
        continue;
      }

      const target = platform === 'win32' ? child.pid : -child.pid;
      try {
        const alive = await withTimeout(
          probeProcessTree(target),
          probeTimeoutMs,
          `Timed out probing process ${target}`
        );
        if (alive) {
          const label = platform === 'win32' ? 'process' : 'process group';
          errors.push(new Error(`${label} ${target} is still running`));
        }
      } catch (error) {
        errors.push(error);
      }
    }

    registered.length = 0;

    if (errors.length > 0) {
      throw new AggregateError(
        errors,
        `Skills certification process teardown failed: ${errors
          .map((error) => error?.message ?? String(error))
          .join('; ')}`
      );
    }
  }

  function teardown() {
    if (!teardownPromise) {
      teardownPromise = teardownRegisteredProcesses();
    }
    return teardownPromise;
  }

  return { spawn, teardown, wait };
}

/** Install removable SIGINT and SIGTERM handlers that share registry teardown. */
export function installCertificationSignalHandlers({
  onSignal,
  processObject = process,
  registry,
} = {}) {
  if (!registry || typeof registry.teardown !== 'function') {
    throw new Error('installCertificationSignalHandlers requires a process registry');
  }

  const handleSignal = (signal) => {
    const teardownPromise = registry.teardown();
    teardownPromise.catch(() => undefined);
    try {
      onSignal?.(signal, teardownPromise);
    } catch {
      // Cleanup takes precedence over optional signal notification callbacks.
    }
  };
  const handleSigint = () => handleSignal('SIGINT');
  const handleSigterm = () => handleSignal('SIGTERM');

  processObject.on('SIGINT', handleSigint);
  processObject.on('SIGTERM', handleSigterm);

  let removed = false;
  return () => {
    if (removed) {
      return;
    }
    removed = true;
    const remove =
      processObject.off?.bind(processObject) ?? processObject.removeListener.bind(processObject);
    remove('SIGINT', handleSigint);
    remove('SIGTERM', handleSigterm);
  };
}
