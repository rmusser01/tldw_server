import { defineConfig } from 'vitest/config'
import path from 'path'
import react from '@vitejs/plugin-react'

type VitestConfig = Extract<Parameters<typeof defineConfig>[0], { plugins?: unknown }>
const reactPlugins = react() as unknown as VitestConfig['plugins']

export default defineConfig({
  plugins: reactPlugins,
  resolve: {
    alias: {
      '@tldw/ui': path.resolve(__dirname, '../packages/ui/src'),
      '@': path.resolve(__dirname, '../packages/ui/src'),
      '~': path.resolve(__dirname, '../packages/ui/src'),
      // Ensure testing libraries resolve from tldw-frontend for packages/ui tests
      '@testing-library/react': path.resolve(__dirname, 'node_modules/@testing-library/react'),
      '@testing-library/jest-dom': path.resolve(__dirname, 'node_modules/@testing-library/jest-dom'),
      '@testing-library/user-event': path.resolve(__dirname, 'node_modules/@testing-library/user-event')
    }
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./vitest.extension.setup.ts'],
    include: ['../packages/ui/src/**/__tests__/**/*.{test,spec}.{ts,tsx}'],
    exclude: ['node_modules/**', 'dist/**', 'build/**']
  }
})
