import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const pagePath = path.resolve(__dirname, '../../pages/admin/index.tsx');

describe('/admin page wiring', () => {
  it('mounts the admin operations overview instead of redirecting to server admin', () => {
    const source = readFileSync(pagePath, 'utf8');
    const sharedRoute = readFileSync(
      path.resolve(__dirname, '../../../packages/ui/src/routes/option-admin.tsx'),
      'utf8'
    );

    expect(source).toContain('dynamic(() => import("@/routes/option-admin"), { ssr: false })');
    expect(sharedRoute).toContain('@/components/Option/Admin/AdminOperationsOverviewPage');
    expect(sharedRoute).toContain('<AdminOperationsOverviewPage />');
    expect(sharedRoute).toContain('<AdminRouteShell path="/admin">');
    expect(source).not.toContain('RouteRedirect');
    expect(source).not.toContain('/admin/server');
    expect(sharedRoute).not.toContain('RouteRedirect');
    expect(sharedRoute).not.toContain('/admin/server');
  });
});
