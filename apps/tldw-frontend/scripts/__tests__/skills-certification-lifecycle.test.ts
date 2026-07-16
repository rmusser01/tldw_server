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

function command(name: string, child: FakeChild) {
  return {
    args: [],
    child,
    command: 'fake-command',
    cwd: '/tmp',
    env: {},
    name,
  };
}

function createHarness({
  closeTimeoutMs = 50,
  platform = 'linux',
  probeProcessTree = vi.fn(async () => false),
  probeTimeoutMs = 50,
  stopProcessTree = vi.fn(async () => undefined),
} = {}) {
  const spawnLoggedProcess = vi.fn((specification) => ({
    ...specification,
    pid: specification.child.pid,
  }));
  const registry = createProcessRegistry({
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
});

describe('Skills certification signal handlers', () => {
  it('installs before spawning, routes SIGINT and SIGTERM through teardown, and removes cleanly', () => {
    const child = new FakeChild(4111);
    const processObject = new EventEmitter();
    const teardownPromise = Promise.resolve();
    const onSignal = vi.fn();
    const { registry } = createHarness();
    const teardown = vi.spyOn(registry, 'teardown').mockReturnValue(teardownPromise);

    const removeHandlers = installCertificationSignalHandlers({
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

    const removeHandlers = installCertificationSignalHandlers({
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
