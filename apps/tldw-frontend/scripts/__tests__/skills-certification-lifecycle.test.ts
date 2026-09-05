import { EventEmitter } from 'node:events';

import { describe, expect, it, vi } from 'vitest';

import * as lifecycleModule from '../skills-certification/lifecycle.mjs';

const { createProcessRegistry, installCertificationSignalHandlers } = lifecycleModule;

class FakeChild extends EventEmitter {
  exitCode: number | null = null;
  signalCode: NodeJS.Signals | null = null;

  constructor(public readonly pid: number) {
    super();
  }
}

type ProcessCommand = {
  args: string[];
  child: FakeChild;
  command: string;
  cwd: string;
  env: Record<string, string>;
  name: string;
};

function command(name: string, child: FakeChild): ProcessCommand {
  return {
    args: [],
    child,
    command: 'fake-command',
    cwd: '/tmp',
    env: {},
    name,
  };
}

type ProcessRecord = ProcessCommand & {
  loggingErrors?: Error[];
  pid: number;
};

type ProcessRegistry = {
  spawn: (command: ProcessCommand, logPath: string) => ProcessRecord;
  stop: (record: ProcessRecord) => Promise<void>;
  teardown: () => Promise<void>;
  wait: (record: ProcessRecord) => Promise<{ code: number | null; signal: NodeJS.Signals | null }>;
};

type ProcessRegistryOptions = {
  closeTimeoutMs?: number;
  platform?: NodeJS.Platform;
  probeProcessTree?: (target: number) => boolean | Promise<boolean>;
  probeTimeoutMs?: number;
  spawnLoggedProcess?: (command: ProcessCommand & { logPath: string }) => ProcessRecord;
  stopProcessTree?: (record: ProcessRecord, options?: { timeoutMs?: number }) => void | Promise<void>;
  stopTimeoutMs?: number;
};

const createTypedProcessRegistry = createProcessRegistry as unknown as (
  options?: ProcessRegistryOptions
) => ProcessRegistry;
const installTypedSignalHandlers = installCertificationSignalHandlers as unknown as (options: {
  onSignal?: (signal: string, teardown: Promise<void>) => void;
  processObject?: EventEmitter;
  registry: ProcessRegistry;
}) => () => void;

type HarnessOptions = Pick<
  ProcessRegistryOptions,
  "closeTimeoutMs" | "platform" | "probeProcessTree" | "probeTimeoutMs" | "stopProcessTree"
>;

function createHarness({
  closeTimeoutMs = 50,
  platform = 'linux' as NodeJS.Platform,
  probeProcessTree = vi.fn(async (_target: number) => false),
  probeTimeoutMs = 50,
  stopProcessTree = vi.fn(async (_record: ProcessRecord) => undefined),
}: HarnessOptions = {}) {
  const spawnLoggedProcess = vi.fn((specification) => ({
    ...specification,
    pid: specification.child.pid,
  }));
  const registry = createTypedProcessRegistry({
    closeTimeoutMs,
    platform,
    probeProcessTree,
    probeTimeoutMs,
    spawnLoggedProcess,
    stopProcessTree,
    stopTimeoutMs: 25,
  });

  return {
    probeProcessTree,
    registry,
    spawnLoggedProcess,
    stopProcessTree,
  };
}

async function flushPromises() {
  await Promise.resolve();
  await Promise.resolve();
}

describe('Skills certification process registry', () => {
  it('exports only the concrete lifecycle entry points', () => {
    expect(Object.keys(lifecycleModule).sort()).toEqual([
      'createProcessRegistry',
      'installCertificationSignalHandlers',
    ]);
  });

  it('attaches the close listener during registration', () => {
    const child = new FakeChild(4101);
    const { registry, spawnLoggedProcess } = createHarness();

    expect(child.listenerCount('close')).toBe(0);
    const record = registry.spawn(command('finite', child), '/tmp/finite.log');

    expect(spawnLoggedProcess).toHaveBeenCalledWith({
      ...command('finite', child),
      logPath: '/tmp/finite.log',
    });
    expect(record.child).toBe(child);
    expect(child.listenerCount('close')).toBe(1);
  });

  it('keeps finite commands registered until close and returns code and signal', async () => {
    const child = new FakeChild(4102);
    const { registry, stopProcessTree } = createHarness();
    const record = registry.spawn(command('finite', child), '/tmp/finite.log');
    const waitOutcome = registry.wait(record);

    child.emit('exit', 7, 'SIGTERM');
    const teardown = registry.teardown();
    await flushPromises();

    expect(stopProcessTree).toHaveBeenCalledTimes(1);
    await expect(Promise.race([waitOutcome, Promise.resolve('still-open')])).resolves.toBe(
      'still-open'
    );

    child.emit('close', 7, 'SIGTERM');

    await expect(waitOutcome).resolves.toEqual({ code: 7, signal: 'SIGTERM' });
    await expect(teardown).resolves.toBeUndefined();
  });

  it('fails finite wait and teardown after a registered child reports logging errors', async () => {
    const child = new FakeChild(4120);
    const record = {
      ...command('logging-error', child),
      loggingErrors: [new Error('log sink failed')],
      pid: child.pid,
    };
    const loggingRegistry = createTypedProcessRegistry({
      closeTimeoutMs: 50,
      probeProcessTree: vi.fn(async () => false),
      spawnLoggedProcess: vi.fn(() => record),
      stopProcessTree: vi.fn(async (value) => value.child.emit('close', 0, null)),
    });
    const registered = loggingRegistry.spawn(command('logging-error', child), '/tmp/bad.log');
    child.emit('close', 0, null);
    await expect(loggingRegistry.wait(registered)).rejects.toThrow(/logging failed/i);
    await expect(loggingRegistry.teardown()).rejects.toThrow(/logging failed/i);
  });

  it('treats a logging error arriving before close as a retryable stop failure', async () => {
    const child = new FakeChild(4121);
    const record = { ...command('late-log', child), loggingErrors: [], pid: child.pid };
    let resolveFirstStop!: () => void;
    const stopProcessTree = vi
      .fn()
      .mockImplementationOnce(() => new Promise<void>((resolve) => (resolveFirstStop = resolve)))
      .mockResolvedValueOnce(undefined);
    const registry = createTypedProcessRegistry({
      probeProcessTree: vi.fn(async () => false),
      spawnLoggedProcess: vi.fn(() => record),
      stopProcessTree,
    });
    const registered = registry.spawn(command('late-log', child), '/tmp/late.log');
    const stop = registry.stop(registered);
    resolveFirstStop();
    await Promise.resolve();
    record.loggingErrors.push(new Error('late log failure'));
    child.emit('close', 0, null);
    await expect(stop).rejects.toThrow(/logging failed/i);
    await expect(registry.teardown()).rejects.toThrow(/logging failed/i);
    expect(stopProcessTree).toHaveBeenCalledTimes(2);
  });

  it('verifies a surviving process group after its parent has already closed', async () => {
    const child = new FakeChild(4112);
    const events: string[] = [];
    const stopProcessTree = vi.fn(async () => {
      events.push('stop');
    });
    const probeProcessTree = vi.fn(async () => {
      events.push('probe');
      return true;
    });
    const { registry } = createHarness({ probeProcessTree, stopProcessTree });
    const record = registry.spawn(command('closed-parent', child), '/tmp/closed-parent.log');

    child.emit('close', 0, null);
    await expect(registry.wait(record)).resolves.toEqual({ code: 0, signal: null });

    await expect(registry.teardown()).rejects.toThrow(/process group -4112 is still running/i);
    expect(stopProcessTree).toHaveBeenCalledWith(record, { timeoutMs: 25 });
    expect(probeProcessTree).toHaveBeenCalledWith(-4112);
    expect(events).toEqual(['stop', 'probe']);
  });

  it('stops, waits for close, then probes the detached POSIX process group', async () => {
    const child = new FakeChild(4103);
    const stopProcessTree = vi.fn(
      () => new Promise<void>((resolve) => child.once('exit', () => resolve()))
    );
    const probeProcessTree = vi.fn(async () => false);
    const { registry } = createHarness({ probeProcessTree, stopProcessTree });
    const record = registry.spawn(command('backend', child), '/tmp/backend.log');

    const teardown = registry.teardown();
    expect(stopProcessTree).toHaveBeenCalledWith(record, { timeoutMs: 25 });
    expect(probeProcessTree).not.toHaveBeenCalled();

    child.emit('exit', null, 'SIGTERM');
    await flushPromises();
    expect(probeProcessTree).not.toHaveBeenCalled();

    child.emit('close', null, 'SIGTERM');
    await teardown;

    expect(probeProcessTree).toHaveBeenCalledWith(-4103);
  });

  it('probes the child PID on Windows', async () => {
    const child = new FakeChild(4104);
    const stopProcessTree = vi.fn(async (record) => {
      record.child.emit('close', 0, null);
    });
    const probeProcessTree = vi.fn(async () => false);
    const { registry } = createHarness({
      platform: 'win32',
      probeProcessTree,
      stopProcessTree,
    });

    registry.spawn(command('frontend', child), '/tmp/frontend.log');
    await registry.teardown();

    expect(probeProcessTree).toHaveBeenCalledWith(4104);
  });

  it('rejects teardown when the process tree survives', async () => {
    const child = new FakeChild(4105);
    const stopProcessTree = vi.fn(async (record) => {
      record.child.emit('close', null, 'SIGKILL');
    });
    const { registry } = createHarness({
      probeProcessTree: vi.fn(async () => true),
      stopProcessTree,
    });

    registry.spawn(command('stubborn', child), '/tmp/stubborn.log');

    await expect(registry.teardown()).rejects.toThrow(/process group -4105 is still running/i);
  });

  it('treats probe errors as teardown failures', async () => {
    const child = new FakeChild(4106);
    const stopProcessTree = vi.fn(async (record) => {
      record.child.emit('close', 0, null);
    });
    const { registry } = createHarness({
      probeProcessTree: vi.fn(async () => {
        throw new Error('probe exploded');
      }),
      stopProcessTree,
    });

    registry.spawn(command('probe-error', child), '/tmp/probe-error.log');

    await expect(registry.teardown()).rejects.toThrow(/probe exploded/i);
  });

  it('treats probe timeouts as teardown failures', async () => {
    const child = new FakeChild(4107);
    const stopProcessTree = vi.fn(async (record) => {
      record.child.emit('close', 0, null);
    });
    const { registry } = createHarness({
      probeProcessTree: vi.fn(() => new Promise<boolean>(() => undefined)),
      probeTimeoutMs: 5,
      stopProcessTree,
    });

    registry.spawn(command('probe-timeout', child), '/tmp/probe-timeout.log');

    await expect(registry.teardown()).rejects.toThrow(/timed out probing/i);
  });

  it('tears registered processes down in reverse order', async () => {
    const first = new FakeChild(4108);
    const second = new FakeChild(4109);
    const order: string[] = [];
    const stopProcessTree = vi.fn(async (record) => {
      order.push(record.name);
      record.child.emit('close', 0, null);
    });
    const { registry } = createHarness({ stopProcessTree });

    registry.spawn(command('first', first), '/tmp/first.log');
    registry.spawn(command('second', second), '/tmp/second.log');
    await registry.teardown();

    expect(order).toEqual(['second', 'first']);
  });

  it('shares one teardown promise and signals each child only once', async () => {
    const child = new FakeChild(4110);
    let releaseStop!: () => void;
    const stopProcessTree = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          releaseStop = resolve;
        })
    );
    const { registry } = createHarness({ stopProcessTree });

    registry.spawn(command('backend', child), '/tmp/backend.log');
    const firstTeardown = registry.teardown();
    const secondTeardown = registry.teardown();

    expect(firstTeardown).toBe(secondTeardown);
    expect(stopProcessTree).toHaveBeenCalledTimes(1);

    releaseStop();
    child.emit('close', null, 'SIGTERM');
    await firstTeardown;

    expect(stopProcessTree).toHaveBeenCalledTimes(1);
  });

  it('stops one registered child once and leaves final teardown compatible', async () => {
    const first = new FakeChild(4113);
    const second = new FakeChild(4114);
    const stopProcessTree = vi.fn(async (record) => {
      record.child.emit('close', 0, null);
    });
    const { registry } = createHarness({ stopProcessTree });
    const firstRecord = registry.spawn(command('first', first), '/tmp/first.log');
    registry.spawn(command('second', second), '/tmp/second.log');

    await registry.stop(firstRecord);
    await registry.stop(firstRecord);
    await registry.teardown();

    expect(stopProcessTree).toHaveBeenCalledTimes(2);
    expect(stopProcessTree).toHaveBeenNthCalledWith(1, firstRecord, { timeoutMs: 25 });
  });

  it('shares one in-progress stop promise for concurrent callers', async () => {
    const child = new FakeChild(4118);
    const { registry, stopProcessTree } = createHarness();
    const record = registry.spawn(command('shared-stop', child), '/tmp/shared-stop.log');

    const first = registry.stop(record);
    const second = registry.stop(record);
    expect(first).toBe(second);
    expect(stopProcessTree).toHaveBeenCalledTimes(1);

    child.emit('close', 0, null);
    await expect(Promise.all([first, second])).resolves.toEqual([undefined, undefined]);
  });

  it('retries a failed alive-process stop during teardown instead of silently skipping it', async () => {
    const child = new FakeChild(4115);
    const stopProcessTree = vi.fn(async () => undefined);
    const probeProcessTree = vi.fn(async () => true);
    const { registry } = createHarness({ probeProcessTree, stopProcessTree });
    const record = registry.spawn(command('alive', child), '/tmp/alive.log');
    child.emit('close', 0, null);

    await expect(registry.stop(record)).rejects.toBeInstanceOf(AggregateError);
    await expect(registry.teardown()).rejects.toBeInstanceOf(AggregateError);
    expect(stopProcessTree).toHaveBeenCalledTimes(2);
    expect(probeProcessTree).toHaveBeenCalledTimes(2);
  });

  it('retries a stopProcessTree rejection during teardown', async () => {
    const child = new FakeChild(4116);
    const stopProcessTree = vi.fn(async () => {
      throw new Error('stop exploded');
    });
    const { registry } = createHarness({ stopProcessTree });
    const record = registry.spawn(command('stop-error', child), '/tmp/stop-error.log');
    child.emit('close', 0, null);

    await expect(registry.stop(record)).rejects.toBeInstanceOf(AggregateError);
    await expect(registry.teardown()).rejects.toBeInstanceOf(AggregateError);
    expect(stopProcessTree).toHaveBeenCalledTimes(2);
  });

  it('retries a close timeout during teardown', async () => {
    const child = new FakeChild(4117);
    const { registry, stopProcessTree } = createHarness({ closeTimeoutMs: 5 });
    const record = registry.spawn(command('close-timeout', child), '/tmp/close-timeout.log');

    await expect(registry.stop(record)).rejects.toBeInstanceOf(AggregateError);
    await expect(registry.teardown()).rejects.toBeInstanceOf(AggregateError);
    expect(stopProcessTree).toHaveBeenCalledTimes(2);
  });
});

describe('Skills certification signal handlers', () => {
  it('installs before spawning, routes SIGINT and SIGTERM through teardown, and removes cleanly', () => {
    const child = new FakeChild(4111);
    const processObject = new EventEmitter();
    const teardownPromise = Promise.resolve();
    const onSignal = vi.fn();
    const { registry } = createHarness();
    const teardown = vi.spyOn(registry, 'teardown').mockReturnValue(teardownPromise);

    const removeHandlers = installTypedSignalHandlers({
      onSignal,
      processObject,
      registry,
    });
    expect(processObject.listenerCount('SIGINT')).toBe(1);
    expect(processObject.listenerCount('SIGTERM')).toBe(1);

    registry.spawn(command('first-child', child), '/tmp/first-child.log');
    processObject.emit('SIGINT');
    processObject.emit('SIGTERM');

    expect(teardown).toHaveBeenCalledTimes(2);
    expect(onSignal).toHaveBeenNthCalledWith(1, 'SIGINT', teardownPromise);
    expect(onSignal).toHaveBeenNthCalledWith(2, 'SIGTERM', teardownPromise);

    removeHandlers();
    expect(processObject.listenerCount('SIGINT')).toBe(0);
    expect(processObject.listenerCount('SIGTERM')).toBe(0);
    processObject.emit('SIGINT');
    expect(teardown).toHaveBeenCalledTimes(2);
  });

  it('isolates throwing signal callbacks after attaching teardown rejection handling', () => {
    const events: string[] = [];
    const processObject = new EventEmitter();
    const { registry } = createHarness();
    const teardownPromise = {
      catch: vi.fn(() => {
        events.push('catch');
        return teardownPromise;
      }),
    } as unknown as Promise<void>;
    const teardown = vi.spyOn(registry, 'teardown').mockImplementation(() => {
      events.push('teardown');
      return teardownPromise;
    });
    const onSignal = vi.fn(() => {
      events.push('callback');
      throw new Error('callback exploded');
    });

    const removeHandlers = installTypedSignalHandlers({
      onSignal,
      processObject,
      registry,
    });

    expect(() => processObject.emit('SIGINT')).not.toThrow();
    expect(() => processObject.emit('SIGTERM')).not.toThrow();
    expect(teardown).toHaveBeenCalledTimes(2);
    expect(onSignal).toHaveBeenCalledTimes(2);
    expect(events).toEqual(['teardown', 'catch', 'callback', 'teardown', 'catch', 'callback']);

    removeHandlers();
  });
});
