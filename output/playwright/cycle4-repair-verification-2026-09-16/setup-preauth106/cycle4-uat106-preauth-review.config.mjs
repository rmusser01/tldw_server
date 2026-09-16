import fs from 'node:fs';
import base from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/vitest.config.ts';
const target = '/apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.model-handoff.test.tsx';
export default {
  ...base,
  root: '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend',
  plugins: [
    {name:'uat106-read-only-review-controls',enforce:'pre',transform(code,id) {
      if (!id.endsWith(target)) return;
      const at = code.lastIndexOf('\n});');
      if (at < 0) throw Error('Missing suite end');
      return code.slice(0, at) + '\n' + fs.readFileSync('/private/tmp/cycle4-uat106-preauth-review-probes.txt','utf8') + code.slice(at);
    }},
    ...base.plugins
  ]
};
