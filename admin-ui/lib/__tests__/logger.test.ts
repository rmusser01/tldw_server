import { describe, it, expect, vi, afterEach } from 'vitest';

describe('logger', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('exports info, warn, error, debug methods', async () => {
    vi.resetModules();
    const { logger } = await import('../logger');
    expect(typeof logger.info).toBe('function');
    expect(typeof logger.warn).toBe('function');
    expect(typeof logger.error).toBe('function');
    expect(typeof logger.debug).toBe('function');
  });

  it('outputs JSON in production', async () => {
    vi.resetModules();
    const origNodeEnv = process.env.NODE_ENV;
    process.env.NODE_ENV = 'production';
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

    const { logger } = await import('../logger');
    logger.info('test message', { component: 'test' });

    expect(consoleSpy).toHaveBeenCalledTimes(1);
    const output = consoleSpy.mock.calls[0][0];
    const parsed = JSON.parse(output);
    expect(parsed).toMatchObject({
      level: 'info',
      message: 'test message',
      component: 'test',
    });
    expect(parsed).toHaveProperty('timestamp');

    process.env.NODE_ENV = origNodeEnv;
  });
});
