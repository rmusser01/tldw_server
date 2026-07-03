import path from "path"
import { defineConfig } from "vitest/config"

export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "~": path.resolve(__dirname, "./src")
    }
  },
  test: {
    environment: "jsdom",
    maxWorkers: 2,
    restoreMocks: true,
    setupFiles: ["./vitest.setup.ts"],
    include: [
      "src/**/__tests__/**/*.test.{ts,tsx}",
      "src/**/__tests__/**/*.spec.{ts,tsx}"
    ],
    exclude: ["node_modules/**", "dist/**", "build/**"],
    coverage: {
      provider: "v8",
      reporter: ["text-summary", "json-summary"],
      include: ["src/**/*.{ts,tsx}"],
      exclude: ["src/**/__tests__/**", "src/**/*.d.ts"],
      // Report-only (no thresholds): pre-existing test failures must not
      // suppress the summary, since vitest v8's default is to skip the
      // coverage report when any test fails (audit F4).
      reportOnFailure: true,
    },
  }
})
