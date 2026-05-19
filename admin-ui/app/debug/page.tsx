'use client';

import { useState } from 'react';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Bug, Gauge, Key, Wallet, Search, Shield, FileKey } from 'lucide-react';
import { Select } from '@/components/ui/select';
import { api } from '@/lib/api-client';
import { isSingleUserMode } from '@/lib/auth';
import { formatDateTime } from '@/lib/format';

type ApiKeyInfo = {
  user_id?: number;
  username?: string;
  email?: string;
  api_key_id?: number;
  name?: string;
  scopes?: string[];
  created_at?: string;
  expires_at?: string;
};

type BudgetSummary = {
  user_id?: number;
  total_budget?: number;
  used_budget?: number;
  remaining_budget?: number;
  budget_period?: string;
  reset_at?: string;
};

const parsePositiveUserId = (rawValue: string): number | null => {
  const trimmed = rawValue.trim();
  if (!/^\d+$/.test(trimmed)) {
    return null;
  }
  const parsed = Number(trimmed);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null;
};

export default function DebugPage() {
  const allowedRoles = isSingleUserMode()
    ? ['admin', 'super_admin', 'owner']
    : ['super_admin', 'owner'];
  const [apiKeyLookupMode, setApiKeyLookupMode] = useState<'raw_key' | 'key_id' | 'user_id'>('raw_key');
  const [apiKeyInput, setApiKeyInput] = useState('');
  const [apiKeyResult, setApiKeyResult] = useState<ApiKeyInfo | null>(null);
  const [apiKeyLoading, setApiKeyLoading] = useState(false);
  const [apiKeyError, setApiKeyError] = useState('');

  const [budgetKeyInput, setBudgetKeyInput] = useState('');
  const [budgetResult, setBudgetResult] = useState<BudgetSummary | null>(null);
  const [budgetLoading, setBudgetLoading] = useState(false);
  const [budgetError, setBudgetError] = useState('');

  // Permission resolver state (8.3)
  const [permUserIdInput, setPermUserIdInput] = useState('');
  const [permResult, setPermResult] = useState<Record<string, unknown> | null>(null);
  const [permLoading, setPermLoading] = useState(false);
  const [permError, setPermError] = useState('');

  // Token validator state (8.3)
  const [tokenInput, setTokenInput] = useState('');
  const [tokenResult, setTokenResult] = useState<Record<string, unknown> | null>(null);
  const [tokenLoading, setTokenLoading] = useState(false);
  const [tokenError, setTokenError] = useState('');

  const getApiKeyPlaceholder = () => {
    if (apiKeyLookupMode === 'key_id') return 'e.g., 42';
    if (apiKeyLookupMode === 'user_id') return 'e.g., 7';
    return 'tldw_...';
  };

  const getApiKeyLabel = () => {
    if (apiKeyLookupMode === 'key_id') return 'Key ID';
    if (apiKeyLookupMode === 'user_id') return 'User ID';
    return 'API Key';
  };

  const handleResolveApiKey = async () => {
    if (!apiKeyInput.trim()) {
      setApiKeyError(`Please enter a ${getApiKeyLabel().toLowerCase()}`);
      return;
    }

    try {
      setApiKeyLoading(true);
      setApiKeyError('');
      setApiKeyResult(null);
      let result: unknown;
      if (apiKeyLookupMode === 'key_id') {
        result = await api.debugResolveApiKey(apiKeyInput.trim(), { mode: 'key_id' });
      } else if (apiKeyLookupMode === 'user_id') {
        result = await api.debugResolveApiKey(apiKeyInput.trim(), { mode: 'user_id' });
      } else {
        result = await api.debugResolveApiKey(apiKeyInput.trim());
      }
      setApiKeyResult(result as ApiKeyInfo);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to resolve API key';
      setApiKeyError(message);
    } finally {
      setApiKeyLoading(false);
    }
  };

  const handleResolvePermissions = async () => {
    if (!permUserIdInput.trim()) {
      setPermError('Please enter a user ID');
      return;
    }
    try {
      setPermLoading(true);
      setPermError('');
      setPermResult(null);
      const result = await api.debugResolvePermissions(permUserIdInput.trim());
      setPermResult(result as Record<string, unknown>);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to resolve permissions';
      setPermError(message);
    } finally {
      setPermLoading(false);
    }
  };

  const handleValidateToken = async () => {
    if (!tokenInput.trim()) {
      setTokenError('Please enter a JWT token');
      return;
    }
    try {
      setTokenLoading(true);
      setTokenError('');
      setTokenResult(null);
      const result = await api.debugValidateToken(tokenInput.trim());
      setTokenResult(result as Record<string, unknown>);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to validate token';
      setTokenError(message);
    } finally {
      setTokenLoading(false);
    }
  };

  const handleGetBudgetSummary = async () => {
    if (!budgetKeyInput.trim()) {
      setBudgetError('Please enter an API key');
      return;
    }

    try {
      setBudgetLoading(true);
      setBudgetError('');
      setBudgetResult(null);
      const result = await api.debugGetBudgetSummary(budgetKeyInput.trim());
      setBudgetResult(result as BudgetSummary);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to get budget summary';
      setBudgetError(message);
    } finally {
      setBudgetLoading(false);
    }
  };

  const formatDate = (dateStr?: string) => formatDateTime(dateStr, { fallback: '—' });

  const formatCurrency = (value?: number) => {
    if (value === undefined || value === null) return '—';
    return `$${value.toFixed(2)}`;
  };

  const budgetUsageRatio =
    budgetResult && budgetResult.total_budget !== undefined && budgetResult.used_budget !== undefined
      ? budgetResult.total_budget > 0
        ? budgetResult.used_budget / budgetResult.total_budget
        : 0
      : 0;

  return (
    <PermissionGuard variant="route" requireAuth role={allowedRoles}>
      <ResponsiveLayout>
        <div className="p-4 lg:p-8">
          <div className="mb-8">
            <h1 className="text-3xl font-bold flex items-center gap-2">
              <Bug className="h-8 w-8" />
              Debug Tools
            </h1>
            <p className="text-muted-foreground">
              Admin-only diagnostic tools for troubleshooting authentication and billing issues
            </p>
          </div>

          <Alert className="mb-6 border-yellow-200 bg-yellow-50">
            <AlertDescription className="text-yellow-800">
              These tools are for debugging purposes only. Be careful with sensitive information.
            </AlertDescription>
          </Alert>

          <div className="grid gap-6 lg:grid-cols-2">
            {/* API Key Resolver */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Key className="h-5 w-5" />
                  API Key Resolver
                </CardTitle>
                <CardDescription>
                  Look up which user owns an API key and view key metadata
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Label htmlFor="api-key-input">{getApiKeyLabel()}</Label>
                    <Select
                      value={apiKeyLookupMode}
                      onChange={(e) => {
                        setApiKeyLookupMode(e.target.value as 'raw_key' | 'key_id' | 'user_id');
                        setApiKeyResult(null);
                        setApiKeyError('');
                      }}
                      className="w-32 h-7 text-xs"
                      data-testid="api-key-lookup-mode"
                    >
                      <option value="raw_key">Raw key</option>
                      <option value="key_id">By Key ID</option>
                      <option value="user_id">By User ID</option>
                    </Select>
                  </div>
                  <div className="flex gap-2">
                    <Input
                      id="api-key-input"
                      type={apiKeyLookupMode === 'raw_key' ? 'password' : 'text'}
                      placeholder={getApiKeyPlaceholder()}
                      value={apiKeyInput}
                      onChange={(e) => setApiKeyInput(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleResolveApiKey()}
                    />
                    <Button onClick={handleResolveApiKey} disabled={apiKeyLoading} loading={apiKeyLoading} loadingText="Looking up...">
                      <Search className="mr-2 h-4 w-4" />
                      Resolve
                    </Button>
                  </div>
                </div>

                {apiKeyError && (
                  <Alert variant="destructive">
                    <AlertDescription>{apiKeyError}</AlertDescription>
                  </Alert>
                )}

                {apiKeyResult && (
                  <div className="rounded-lg border p-4 space-y-2">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="text-muted-foreground">User ID:</div>
                      <div className="font-medium">{apiKeyResult.user_id ?? '—'}</div>

                      <div className="text-muted-foreground">Username:</div>
                      <div className="font-medium">{apiKeyResult.username ?? '—'}</div>

                      <div className="text-muted-foreground">Email:</div>
                      <div className="font-medium">{apiKeyResult.email ?? '—'}</div>

                      <div className="text-muted-foreground">Key ID:</div>
                      <div className="font-medium">{apiKeyResult.api_key_id ?? '—'}</div>

                      <div className="text-muted-foreground">Key Name:</div>
                      <div className="font-medium">{apiKeyResult.name ?? '—'}</div>

                      <div className="text-muted-foreground">Scopes:</div>
                      <div className="font-medium">
                        {apiKeyResult.scopes?.join(', ') || '—'}
                      </div>

                      <div className="text-muted-foreground">Created:</div>
                      <div className="font-medium">{formatDate(apiKeyResult.created_at)}</div>

                      <div className="text-muted-foreground">Expires:</div>
                      <div className="font-medium">{formatDate(apiKeyResult.expires_at)}</div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Budget Summary */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Wallet className="h-5 w-5" />
                  Budget Summary
                </CardTitle>
                <CardDescription>
                  View budget usage and limits for an API key
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="budget-key-input">API Key</Label>
                  <div className="flex gap-2">
                    <Input
                      id="budget-key-input"
                      type="password"
                      placeholder="tldw_..."
                      value={budgetKeyInput}
                      onChange={(e) => setBudgetKeyInput(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleGetBudgetSummary()}
                    />
                    <Button onClick={handleGetBudgetSummary} disabled={budgetLoading} loading={budgetLoading} loadingText="Loading...">
                      <Search className="mr-2 h-4 w-4" />
                      Check
                    </Button>
                  </div>
                </div>

                {budgetError && (
                  <Alert variant="destructive">
                    <AlertDescription>{budgetError}</AlertDescription>
                  </Alert>
                )}

                {budgetResult && (
                  <div className="rounded-lg border p-4 space-y-2">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="text-muted-foreground">User ID:</div>
                      <div className="font-medium">{budgetResult.user_id ?? '—'}</div>

                      <div className="text-muted-foreground">Total Budget:</div>
                      <div className="font-medium">{formatCurrency(budgetResult.total_budget)}</div>

                      <div className="text-muted-foreground">Used:</div>
                      <div className="font-medium">{formatCurrency(budgetResult.used_budget)}</div>

                      <div className="text-muted-foreground">Remaining:</div>
                      <div className="font-medium">{formatCurrency(budgetResult.remaining_budget)}</div>

                      <div className="text-muted-foreground">Period:</div>
                      <div className="font-medium">{budgetResult.budget_period ?? '—'}</div>

                      <div className="text-muted-foreground">Resets At:</div>
                      <div className="font-medium">{formatDate(budgetResult.reset_at)}</div>
                    </div>

                    {budgetResult.total_budget !== undefined && budgetResult.used_budget !== undefined && (
                      <div className="mt-4">
                        <div className="flex justify-between text-xs text-muted-foreground mb-1">
                          <span>Usage</span>
                          <span>
                            {(budgetUsageRatio * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-2 bg-muted rounded-full overflow-hidden">
                          <div
                            className={`h-full transition-all ${
                              budgetUsageRatio > 0.9
                                ? 'bg-red-500'
                                : budgetUsageRatio > 0.7
                                  ? 'bg-yellow-500'
                                  : 'bg-green-500'
                            }`}
                            style={{
                              width: `${Math.min(100, budgetUsageRatio * 100)}%`,
                            }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Permission Resolver */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="h-5 w-5" />
                  Permission Resolver
                </CardTitle>
                <CardDescription>
                  Look up effective permissions for a user ID
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="perm-user-id-input">User ID</Label>
                  <div className="flex gap-2">
                    <Input
                      id="perm-user-id-input"
                      placeholder="e.g., 42"
                      value={permUserIdInput}
                      onChange={(e) => setPermUserIdInput(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleResolvePermissions()}
                    />
                    <Button onClick={handleResolvePermissions} disabled={permLoading} loading={permLoading} loadingText="Resolving...">
                      <Search className="mr-2 h-4 w-4" />
                      Resolve
                    </Button>
                  </div>
                </div>

                {permError && (
                  <Alert variant="destructive">
                    <AlertDescription>{permError}</AlertDescription>
                  </Alert>
                )}

                {permResult && (
                  <div className="rounded-lg border p-4 space-y-2">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="text-muted-foreground">User ID:</div>
                      <div className="font-medium">{String(permResult.user_id ?? permUserIdInput)}</div>

                      <div className="text-muted-foreground">Role:</div>
                      <div className="font-medium">{String(permResult.role ?? permResult.roles ?? '—')}</div>

                      <div className="text-muted-foreground">Permissions:</div>
                      <div className="font-medium text-xs">
                        {Array.isArray(permResult.permissions)
                          ? permResult.permissions.join(', ')
                          : String(permResult.permissions ?? '—')}
                      </div>

                      {Boolean(permResult.scopes) && (
                        <>
                          <div className="text-muted-foreground">Scopes:</div>
                          <div className="font-medium text-xs">
                            {Array.isArray(permResult.scopes)
                              ? permResult.scopes.join(', ')
                              : String(permResult.scopes)}
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Token Validator */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileKey className="h-5 w-5" />
                  Token Validator
                </CardTitle>
                <CardDescription>
                  Decode and validate a JWT token to inspect its claims
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="token-input">JWT Token</Label>
                  <div className="flex gap-2">
                    <Input
                      id="token-input"
                      type="password"
                      placeholder="eyJhbG..."
                      value={tokenInput}
                      onChange={(e) => setTokenInput(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleValidateToken()}
                    />
                    <Button onClick={handleValidateToken} disabled={tokenLoading} loading={tokenLoading} loadingText="Validating...">
                      <Search className="mr-2 h-4 w-4" />
                      Validate
                    </Button>
                  </div>
                </div>

                {tokenError && (
                  <Alert variant="destructive">
                    <AlertDescription>{tokenError}</AlertDescription>
                  </Alert>
                )}

                {tokenResult && (
                  <div className="rounded-lg border p-4 space-y-2">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="text-muted-foreground">Valid:</div>
                      <div className="font-medium">
                        {tokenResult.valid === true ? 'Yes' : tokenResult.valid === false ? 'No' : '—'}
                      </div>

                      {Boolean(tokenResult.sub) && (
                        <>
                          <div className="text-muted-foreground">Subject:</div>
                          <div className="font-medium">{String(tokenResult.sub)}</div>
                        </>
                      )}

                      {Boolean(tokenResult.exp) && (
                        <>
                          <div className="text-muted-foreground">Expires:</div>
                          <div className="font-medium">{formatDate(String(tokenResult.exp))}</div>
                        </>
                      )}

                      {Boolean(tokenResult.iat) && (
                        <>
                          <div className="text-muted-foreground">Issued at:</div>
                          <div className="font-medium">{formatDate(String(tokenResult.iat))}</div>
                        </>
                      )}

                      {Boolean(tokenResult.scopes) && (
                        <>
                          <div className="text-muted-foreground">Scopes:</div>
                          <div className="font-medium text-xs">
                            {Array.isArray(tokenResult.scopes)
                              ? tokenResult.scopes.join(', ')
                              : String(tokenResult.scopes)}
                          </div>
                        </>
                      )}

                      {Boolean(tokenResult.error) && (
                        <>
                          <div className="text-muted-foreground">Error:</div>
                          <div className="font-medium text-destructive">{String(tokenResult.error)}</div>
                        </>
                      )}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            {/* User Lookup Tool */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Search className="h-5 w-5" />
                  Lookup by User ID
                </CardTitle>
                <CardDescription>Retrieve user details by numeric ID instead of raw API key</CardDescription>
              </CardHeader>
              <CardContent>
                <UserLookupTool />
              </CardContent>
            </Card>
            {/* Permission Resolver */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Key className="h-5 w-5" />
                  Permission Resolver
                </CardTitle>
                <CardDescription>Resolve effective permissions for a user by ID</CardDescription>
              </CardHeader>
              <CardContent>
                <PermissionResolverTool />
              </CardContent>
            </Card>

            {/* Token Validator */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Search className="h-5 w-5" />
                  Token Validator
                </CardTitle>
                <CardDescription>Decode and validate a JWT token</CardDescription>
              </CardHeader>
              <CardContent>
                <TokenValidatorTool />
              </CardContent>
            </Card>

            {/* Rate Limit Simulator */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Gauge className="h-5 w-5" />
                  Rate Limit Simulator
                </CardTitle>
                <CardDescription>Check what rate limits apply for a given user and endpoint</CardDescription>
              </CardHeader>
              <CardContent>
                <RateLimitSimTool />
              </CardContent>
            </Card>
          </div>
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}

function PermissionResolverTool() {
  const [userId, setUserId] = useState('');
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleResolve = async () => {
    const parsedUserId = parsePositiveUserId(userId);
    if (parsedUserId === null) { setError('Enter a valid positive user ID'); return; }
    setLoading(true); setError(''); setResult(null);
    try {
      const data = await api.debugResolvePermissions(String(parsedUserId));
      setResult(data as unknown as Record<string, unknown>);
    } catch (err) { setError(err instanceof Error ? err.message : 'Failed'); }
    finally { setLoading(false); }
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <div className="flex-1">
          <Label htmlFor="perm-user-id">User ID</Label>
          <Input id="perm-user-id" placeholder="e.g., 42" value={userId}
            onChange={(e) => setUserId(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleResolve()} />
        </div>
        <Button onClick={handleResolve} disabled={loading} className="self-end" loading={loading}>Resolve</Button>
      </div>
      {error && <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>}
      {result && (
        <pre className="bg-muted p-3 rounded text-xs overflow-auto max-h-64">{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  );
}

function TokenValidatorTool() {
  const [token, setToken] = useState('');
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleValidate = async () => {
    if (!token.trim()) { setError('Paste a JWT token'); return; }
    setLoading(true); setError(''); setResult(null);
    try {
      const data = await api.debugValidateToken(token.trim());
      setResult(data as unknown as Record<string, unknown>);
    } catch (err) { setError(err instanceof Error ? err.message : 'Failed'); }
    finally { setLoading(false); }
  };

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="debug-token">JWT Token</Label>
        <Input id="debug-token" type="password" placeholder="eyJhbG..." value={token}
          onChange={(e) => setToken(e.target.value)} />
      </div>
      <Button onClick={handleValidate} disabled={loading} loading={loading}>Validate</Button>
      {error && <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>}
      {result && (
        <pre className="bg-muted p-3 rounded text-xs overflow-auto max-h-64">{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  );
}

function UserLookupTool() {
  const [userId, setUserId] = useState('');
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleLookup = async () => {
    const parsedUserId = parsePositiveUserId(userId);
    if (parsedUserId === null) { setError('Enter a valid positive user ID'); return; }
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const user = await api.getUser(String(parsedUserId));
      setResult(user as unknown as Record<string, unknown>);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'User not found');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <div className="flex-1">
          <Label htmlFor="debug-user-id">User ID</Label>
          <Input id="debug-user-id" placeholder="e.g., 42" value={userId}
            onChange={(e) => setUserId(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleLookup()} />
        </div>
        <Button onClick={handleLookup} disabled={loading} className="self-end" loading={loading}>
          Lookup
        </Button>
      </div>
      {error && <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>}
      {result && (
        <pre className="bg-muted p-3 rounded text-xs overflow-auto max-h-64">
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
}

type RateLimitSimResult = {
  would_allow?: boolean;
  limit_source?: string;
  effective_limit_per_min?: number | null;
  effective_burst?: number | null;
};

function RateLimitSimTool() {
  const [userId, setUserId] = useState('');
  const [endpoint, setEndpoint] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<Record<string, unknown> | null>(null);

  const handleSimulate = async () => {
    const uid = parsePositiveUserId(userId);
    if (uid === null) { setError('Enter a valid positive user ID'); return; }
    if (!endpoint.trim()) { setError('Enter an endpoint path'); return; }
    setLoading(true); setError(''); setResult(null);
    try {
      const res = await api.debugSimulateRateLimit({ user_id: uid, endpoint: endpoint.trim() });
      setResult(res as unknown as Record<string, unknown>);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Simulation failed');
    } finally {
      setLoading(false);
    }
  };
  const typedResult = result as RateLimitSimResult | null;

  return (
    <div className="space-y-3">
      <div className="flex gap-3 items-end">
        <div className="flex-1">
          <Label htmlFor="rl-user-id">User ID</Label>
          <Input id="rl-user-id" placeholder="e.g., 42" value={userId}
            onChange={(e) => setUserId(e.target.value)} />
        </div>
        <div className="flex-[2]">
          <Label htmlFor="rl-endpoint">Endpoint</Label>
          <Input id="rl-endpoint" placeholder="e.g., /api/v1/chat/completions" value={endpoint}
            onChange={(e) => setEndpoint(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSimulate()} />
        </div>
        <Button onClick={handleSimulate} disabled={loading} className="self-end" loading={loading}>
          Simulate
        </Button>
      </div>
      {error && <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>}
      {typedResult && (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Badge variant={typedResult.would_allow ? 'default' : 'destructive'}>
              {typedResult.would_allow ? 'Allowed' : 'Blocked'}
            </Badge>
            <span className="text-sm text-muted-foreground">
              Source: {typedResult.limit_source || 'none'}
            </span>
            {typedResult.effective_limit_per_min != null && (
              <span className="text-sm">
                {typedResult.effective_limit_per_min}/min
                {typedResult.effective_burst != null && (
                  <> (burst: {typedResult.effective_burst})</>
                )}
              </span>
            )}
          </div>
          <pre className="bg-muted p-3 rounded text-xs overflow-auto max-h-48">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
