import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const pageRoot = path.resolve(__dirname, '../../pages/connectors');

const readConnectorPage = (fileName: string): string =>
  readFileSync(path.join(pageRoot, fileName), 'utf8');

describe('connector placeholder page wiring', () => {
  it.each([
    ['index.tsx', '/connectors'],
    ['browse.tsx', '/connectors/browse'],
    ['jobs.tsx', '/connectors/jobs'],
    ['sources.tsx', '/connectors/sources'],
  ])('%s renders the connector-specific placeholder for %s', (fileName, route) => {
    const source = readConnectorPage(fileName);

    expect(source).toContain('ConnectorRoutePlaceholder');
    expect(source).toContain(`route="${route}"`);
    expect(source).not.toContain("components/navigation/RoutePlaceholder");
    expect(source).not.toContain("<RoutePlaceholder");
    expect(source).not.toContain('Coming Soon');
  });
});
