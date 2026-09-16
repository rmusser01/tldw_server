import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config.ts";
import fs from 'node:fs';
const target = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Layouts/SettingsOptionLayout.tsx";
export default {
  ...base,
  plugins: [...(base.plugins || []), {
    name: 'uat155-review-baseline-source',
    enforce: 'pre',
    load(id) {
      if (id.split('?')[0] !== target) return;
      process.stderr.write('UAT155_BASELINE_LOADED sha256=7202f11585da5d61a17f864139eb9adb27c051b97924aea33838b26935855f8f source=2d5ad06c86cf279fe0fcc6445009813d97ab1c1b\n');
      return fs.readFileSync('/private/tmp/uat155-independent-baseline-source.tsx', 'utf8');
    }
  }]
};
