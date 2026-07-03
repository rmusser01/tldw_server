import { defineConfig } from 'vitest/config';
import path from 'path';
import react from '@vitejs/plugin-react';

type VitestConfig = Extract<Parameters<typeof defineConfig>[0], { plugins?: unknown }>;
const reactPlugins = react() as unknown as VitestConfig['plugins'];

export default defineConfig({
  plugins: reactPlugins,
  resolve: {
    alias: {
      '@tldw/ui': path.resolve(__dirname, '../packages/ui/src'),
      '@': path.resolve(__dirname, '../packages/ui/src'),
      '~': path.resolve(__dirname, '../packages/ui/src'),
      '@web': path.resolve(__dirname, '.'),
      '@plasmohq/storage/hook': path.resolve(
        __dirname,
        './extension/shims/plasmo-storage-hook.tsx'
      ),
      '@plasmohq/storage': path.resolve(__dirname, './extension/shims/plasmo-storage.ts'),
      'wxt/browser': path.resolve(__dirname, './extension/shims/wxt-browser.ts'),
      'react-router-dom': path.resolve(
        __dirname,
        '../packages/ui/node_modules/react-router-dom'
      ),
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./vitest.setup.ts'],
    include: [
      '**/__tests__/**/*.test.{ts,tsx}',
      '**/__tests__/**/*.spec.{ts,tsx}',
      '../packages/ui/src/**/__tests__/**/*.test.{ts,tsx}',
      '../packages/ui/src/**/__tests__/**/*.spec.{ts,tsx}',
    ],
    exclude: ['node_modules/**', 'dist/**', 'build/**', 'pages/**'],
    coverage: {
      provider: 'v8',
      reporter: ['text-summary', 'json-summary'],
      include: ['components/**/*.{ts,tsx}', 'hooks/**/*.{ts,tsx}', 'lib/**/*.{ts,tsx}'],
      exclude: ['**/__tests__/**', '**/*.d.ts'],
      // Report-only (no thresholds): pre-existing test failures must not
      // suppress the summary, since vitest v8's default is to skip the
      // coverage report when any test fails (audit F4).
      reportOnFailure: true,
    },
  },
});
