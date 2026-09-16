const ui = '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui'
export default {
  root: ui,
  resolve: { alias: { '@': ui+'/src', '~': ui+'/src', vitest: ui+'/node_modules/vitest/dist/index.js' } },
  test: { environment: 'jsdom', setupFiles: [ui+'/vitest.setup.ts'], include: ['/private/tmp/source013-diagnosis-20260916/*.test.ts'], maxWorkers: 1 },
}
