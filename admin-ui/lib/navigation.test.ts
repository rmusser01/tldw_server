import { describe, expect, it } from 'vitest';
import {
  buildBreadcrumbs,
  getPageTitleForPath,
  matchesNavigationQuery,
  navigation,
  navigationSections,
  type NavigationItem,
} from './navigation';

describe('navigation information architecture', () => {
  it('keeps sections in the intended order', () => {
    expect(navigationSections.map((section) => section.title)).toEqual([
      'Overview',
      'Identity & Access',
      'AI & Models',
      'Operations',
      'Cost & Usage',
      'Security & Compliance',
      'Integrations',
      'Advanced',
    ]);
  });

  it('includes key admin destinations in operations, cost/usage, and security groups', () => {
    const operations = navigationSections.find((section) => section.title === 'Operations');
    const costUsage = navigationSections.find((section) => section.title === 'Cost & Usage');
    const securityCompliance = navigationSections.find((section) => section.title === 'Security & Compliance');
    const advanced = navigationSections.find((section) => section.title === 'Advanced');

    expect(operations?.items.map((item) => item.href)).toContain('/monitoring');
    expect(operations?.items.map((item) => item.href)).toContain('/dependencies');
    expect(operations?.items.map((item) => item.href)).toContain('/incidents');
    expect(costUsage?.items.map((item) => item.href)).toContain('/resource-governor');
    expect(securityCompliance?.items.map((item) => item.href)).toContain('/security');
    expect(advanced?.items.map((item) => item.href)).toContain('/config');

    const integrations = navigationSections.find((section) => section.title === 'Integrations');
    expect(integrations?.items.map((item) => item.href)).toContain('/webhooks');
  });
});

describe('matchesNavigationQuery', () => {
  const usersItem = navigation.find((item) => item.href === '/users') as NavigationItem;
  const providersItem = navigation.find((item) => item.href === '/providers') as NavigationItem;

  it('matches by item name', () => {
    expect(matchesNavigationQuery(usersItem, 'Identity & Access', 'users')).toBe(true);
  });

  it('matches by section name', () => {
    expect(matchesNavigationQuery(usersItem, 'Identity & Access', 'identity')).toBe(true);
  });

  it('matches by keyword metadata', () => {
    expect(matchesNavigationQuery(providersItem, 'AI & Models', 'models')).toBe(true);
  });

  it('returns false when query does not match', () => {
    expect(matchesNavigationQuery(usersItem, 'Identity & Access', 'billing')).toBe(false);
  });
});

describe('buildBreadcrumbs', () => {
  it('builds nested breadcrumbs for static nested routes', () => {
    expect(buildBreadcrumbs('/roles/matrix')).toEqual([
      { label: 'Dashboard', href: '/', current: false },
      { label: 'Roles & Permissions', href: '/roles', current: false },
      { label: 'Permission Matrix', current: true },
    ]);
  });

  it('builds breadcrumbs for dynamic user detail routes', () => {
    expect(buildBreadcrumbs('/users/123')).toEqual([
      { label: 'Dashboard', href: '/', current: false },
      { label: 'Users', href: '/users', current: false },
      { label: 'User 123', current: true },
    ]);
  });
});

describe('getPageTitleForPath', () => {
  it('returns admin dashboard title format for route', () => {
    expect(getPageTitleForPath('/users')).toBe('Users | Admin Dashboard');
    expect(getPageTitleForPath('/users/123')).toBe('User 123 | Admin Dashboard');
  });
});
