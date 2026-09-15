// @vitest-environment node
import { createRequire } from 'node:module';
import { once } from 'node:events';
import type { AddressInfo } from 'node:net';
import { describe, expect, it } from 'vitest';

// Run the same consumer-boundary checks against the independent Admin install
// by setting this to its absolute package.json path. CI defaults to the WebUI.
const require = createRequire(
  process.env.TLDW_OTEL_TEST_PACKAGE ?? new URL('../package.json', import.meta.url)
);
const nextRequire = createRequire(require.resolve('@sentry/nextjs'));
const sentryRequire = createRequire(nextRequire.resolve('@sentry/node'));
const httpRequire = createRequire(sentryRequire.resolve('@opentelemetry/instrumentation-http'));
const core = httpRequire('@opentelemetry/core');
const api = httpRequire('@opentelemetry/api');
const sentryCore = sentryRequire('@opentelemetry/core');
const propagator = new core.W3CBaggagePropagator();

function extract(baggage: string | string[]) {
  return api.propagation.getBaggage(
    propagator.extract(api.ROOT_CONTEXT, { baggage }, api.defaultTextMapGetter)
  );
}

describe('HTTP instrumentation dependency boundary (CVE-2026-54285)', () => {
  it('resolves the same patched core as the Sentry SDK', () => {
    expect(httpRequire.resolve('@opentelemetry/core')).toBe(
      sentryRequire.resolve('@opentelemetry/core')
    );
  });

  it.each(['single', 'array'])('caps entry count across %s header values', (shape) => {
    const entries = Array.from({ length: 181 }, (_, i) => `k${i}=v`);
    const baggage = extract(shape === 'single' ? entries.join(',') : entries);
    expect(baggage.getAllEntries()).toHaveLength(180);
    expect(baggage.getEntry('k179')).toEqual({ value: 'v' });
    expect(baggage.getEntry('k180')).toBeUndefined();
  });

  it.each(['single', 'array'])('caps aggregate size across %s header values', (shape) => {
    // 4096 + comma + 4095 = exactly 8192 ASCII bytes.
    const entries = [`a=${'x'.repeat(4094)}`, `b=${'y'.repeat(4093)}`, 'overflow=z'];
    const baggage = extract(shape === 'single' ? entries.join(',') : entries);
    expect(baggage.getAllEntries().map(([key]: [string, unknown]) => key)).toEqual(['a', 'b']);
  });

  it.each([4097, 65546])(
    'skips a %i-byte entry while retaining the valid entry at the limit',
    (size) => {
      const baggage = extract(`oversized=${'x'.repeat(size - 10)},ok=${'y'.repeat(4093)}`);
      expect(baggage.getEntry('oversized') === undefined).toBe(true);
      expect(baggage.getEntry('ok')).toEqual({ value: 'y'.repeat(4093) });
    }
  );

  it('round-trips ordinary percent-encoded values and metadata', () => {
    const context = propagator.extract(
      api.ROOT_CONTEXT,
      { baggage: 'topic=caf%C3%A9%20notes;source=ui' },
      api.defaultTextMapGetter
    );
    expect(api.propagation.getBaggage(context).getEntry('topic').value).toBe('café notes');
    const carrier = {};
    propagator.inject(context, carrier, api.defaultTextMapSetter);
    expect(carrier).toEqual({ baggage: 'topic=caf%C3%A9%20notes;source=ui' });
  });

  it('ignores malformed entries without discarding a valid sibling', () => {
    expect(extract('broken=%ZZ,no-equals,ok=yes').getAllEntries()).toEqual([
      ['ok', { value: 'yes' }],
    ]);
  });

  it('leaves an absent baggage context unchanged', () => {
    expect(propagator.extract(api.ROOT_CONTEXT, {}, api.defaultTextMapGetter)).toBe(
      api.ROOT_CONTEXT
    );
  });

  it('preserves shared RPC metadata and suppression across the SDK boundary', () => {
    const metadata = { type: core.RPCType.HTTP, route: '/compat' };
    const context = core.setRPCMetadata(api.ROOT_CONTEXT, metadata);
    expect(sentryCore.getRPCMetadata(context)).toBe(metadata);
    expect(sentryCore.isTracingSuppressed(core.suppressTracing(context))).toBe(true);
  });

  it('preserves Sentry remote-parent extraction and outbound context', () => {
    // Sentry overrides extract; this is a compatibility control, not a claim
    // that the core update limits Sentry's separate inbound baggage parser.
    const { SentryPropagator } = sentryRequire('@sentry/opentelemetry');
    const sentry = new SentryPropagator();
    const context = sentry.extract(
      api.ROOT_CONTEXT,
      {
        'sentry-trace': '12345678901234567890123456789012-1234567890123456-1',
        baggage: 'sentry-environment=development',
      },
      api.defaultTextMapGetter
    );
    expect(api.trace.getSpanContext(context)).toMatchObject({
      traceId: '12345678901234567890123456789012',
      spanId: '1234567890123456',
      isRemote: true,
    });
    const carrier: Record<string, string> = { baggage: 'topic=research' };
    sentry.inject(context, carrier, api.defaultTextMapSetter);
    expect(carrier['sentry-trace']).toMatch(/^12345678901234567890123456789012-/);
    expect(carrier.baggage).toContain('topic=research');
  });

  it('preserves HTTP responses, parent traces, route metadata and ignored requests', async () => {
    const { BasicTracerProvider, InMemorySpanExporter, SimpleSpanProcessor } = sentryRequire(
      '@opentelemetry/sdk-trace-base'
    );
    const { AsyncLocalStorageContextManager } = sentryRequire('@opentelemetry/context-async-hooks');
    const { HttpInstrumentation } = sentryRequire('@opentelemetry/instrumentation-http');
    const exporter = new InMemorySpanExporter();
    const provider = new BasicTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
    const manager = new AsyncLocalStorageContextManager().enable();
    api.context.setGlobalContextManager(manager);
    api.propagation.setGlobalPropagator(new sentryCore.W3CTraceContextPropagator());
    const instrumentation = new HttpInstrumentation({
      enabled: false,
      ignoreIncomingRequestHook: (request: { url: string }) => request.url === '/ignored',
      ignoreOutgoingRequestHook: () => true,
    });
    instrumentation.setTracerProvider(provider);
    instrumentation.enable();
    const http = require('node:http') as typeof import('node:http');
    const server = http.createServer((request, response) => {
      const metadata = sentryCore.getRPCMetadata(api.context.active());
      if (metadata) metadata.route = '/compat';
      response.end(request.url === '/ignored' ? 'ignored ok' : 'traced ok');
    });
    try {
      server.listen(0, '127.0.0.1');
      await once(server, 'listening');
      const port = (server.address() as AddressInfo).port;
      for (const [path, expected] of [
        ['/compat', 'traced ok'],
        ['/ignored', 'ignored ok'],
      ]) {
        const body = await new Promise<string>((resolve, reject) => {
          http
            .get(
              {
                host: '127.0.0.1',
                port,
                path,
                headers: { traceparent: '00-12345678901234567890123456789012-1234567890123456-01' },
              },
              (response) => {
                let body = '';
                response.setEncoding('utf8');
                response.on('data', (chunk: string) => {
                  body += chunk;
                });
                response.on('end', () => resolve(body));
                response.on('error', reject);
              }
            )
            .on('error', reject);
        });
        expect(body).toBe(expected);
      }
      await provider.forceFlush();
      const spans = exporter.getFinishedSpans();
      expect(spans).toHaveLength(1);
      expect(spans[0].spanContext().traceId).toBe('12345678901234567890123456789012');
      expect(spans[0].parentSpanContext.spanId).toBe('1234567890123456');
      expect(spans[0].attributes['http.route']).toBe('/compat');
    } finally {
      if (server.listening) await new Promise<void>((resolve) => server.close(() => resolve()));
      instrumentation.disable();
      await provider.shutdown();
      manager.disable();
      api.context.disable();
      api.propagation.disable();
    }
  });
});
