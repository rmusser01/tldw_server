import { configDefaults, defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  esbuild: {
    jsx: 'automatic',
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '.'),
    },
  },
  test: {
    environment: 'jsdom',
    exclude: [...configDefaults.exclude, 'tests/e2e/**'],
    // Node 25+ Web Storage shadows JSDOM unless a process-wide storage file is configured.
    execArgv: process.allowedNodeEnvironmentFlags.has('--no-experimental-webstorage')
      ? ['--no-experimental-webstorage']
      : [],
    setupFiles: ['./vitest.setup.ts'],
  },
});
