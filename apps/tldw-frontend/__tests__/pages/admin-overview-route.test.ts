import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const pagePath = path.resolve(__dirname, '../../pages/admin/index.tsx');

describe('/admin page wiring', () => {
  it('mounts the admin operations overview instead of redirecting to server admin', () => {
    const source = readFileSync(pagePath, 'utf8');

    expect(source).toContain('AdminOperationsOverviewPage');
    expect(source).toContain('@/components/Option/Admin/AdminOperationsOverviewPage');
    expect(source).not.toContain('RouteRedirect');
    expect(source).not.toContain('/admin/server');
  });
});
