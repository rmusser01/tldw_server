export default {
  resolve: {
    alias: {
      "@": `${__dirname}/apps/packages/ui/src`,
      "~": `${__dirname}/apps/packages/ui/src`,
      "@web": `${__dirname}/apps/tldw-frontend`
    }
  },
  test: {
    environment: "node",
    setupFiles: ["./apps/packages/ui/src/components/Option/Calendar/__tests__/root-dom-setup.ts"],
    // Root bunx Vitest is intentionally scoped to Calendar tests for this PRD slice.
    include: [
      "apps/packages/ui/src/services/__tests__/calendar.test.ts",
      "apps/packages/ui/src/components/Option/Calendar/__tests__/**/*.test.tsx"
    ],
    exclude: [
      "node_modules/**",
      "dist/**",
      "build/**",
      ".worktrees/**",
      ".claude/worktrees/**",
      "apps/node_modules/**",
      "apps/tldw-frontend/.next/**"
    ],
    maxWorkers: 2,
    restoreMocks: true
  }
}
