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
    exclude: ["node_modules/**", "dist/**", "build/**"]
  }
})
