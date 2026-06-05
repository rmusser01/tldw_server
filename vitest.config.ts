import path from "path"

export default {
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "apps/packages/ui/src"),
      "~": path.resolve(__dirname, "apps/packages/ui/src"),
      "@web": path.resolve(__dirname, "apps/tldw-frontend")
    }
  },
  test: {
    environment: "node",
    maxWorkers: 2,
    restoreMocks: true,
    exclude: [
      "node_modules/**",
      "dist/**",
      "build/**",
      ".worktrees/**",
      ".claude/worktrees/**",
      "apps/node_modules/**",
      "apps/tldw-frontend/.next/**"
    ]
  }
}
