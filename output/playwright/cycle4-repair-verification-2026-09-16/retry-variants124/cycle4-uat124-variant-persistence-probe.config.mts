import base from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config'
export default { ...base, test: { ...base.test, include: ['/private/tmp/cycle4-uat124-variant-persistence-probe.test.ts'], setupFiles: ['/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.setup.ts'], maxWorkers: 1 } }
