import { defineConfig } from 'vitest/config';
import path from 'path';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@tldw/ui': path.resolve(__dirname, '../packages/ui/src'),
      '@': path.resolve(__dirname, '../packages/ui/src'),
      '~': path.resolve(__dirname, '../packages/ui/src'),
      '@web': path.resolve(__dirname, '.'),
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
  },
});
