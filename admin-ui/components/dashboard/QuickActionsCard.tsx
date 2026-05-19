'use client';

import Link from 'next/link';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Activity, Building2, Cpu, FileText, Key, Shield, Users } from 'lucide-react';

export const QuickActionsCard = () => (
  <Card>
    <CardHeader>
      <CardTitle>Quick Actions</CardTitle>
      <CardDescription>Common administrative tasks</CardDescription>
    </CardHeader>
    <CardContent>
      <div className="grid gap-3 sm:grid-cols-2">
        <Link
          href="/users"
          className="flex items-center gap-3 p-4 rounded-lg border hover:bg-muted transition-colors"
        >
          <Users className="h-5 w-5 text-primary" />
          <div>
            <p className="font-medium">Manage Users</p>
            <p className="text-xs text-muted-foreground">Add or edit users</p>
          </div>
        </Link>
        <Link
          href="/organizations"
          className="flex items-center gap-3 p-4 rounded-lg border hover:bg-muted transition-colors"
        >
          <Building2 className="h-5 w-5 text-primary" />
          <div>
            <p className="font-medium">Organizations</p>
            <p className="text-xs text-muted-foreground">Create or manage orgs</p>
          </div>
        </Link>
        <Link
          href="/api-keys"
          className="flex items-center gap-3 p-4 rounded-lg border hover:bg-muted transition-colors"
        >
          <Key className="h-5 w-5 text-primary" />
          <div>
            <p className="font-medium">API Keys</p>
            <p className="text-xs text-muted-foreground">Create or revoke</p>
          </div>
        </Link>
        <Link
          href="/audit"
          className="flex items-center gap-3 p-4 rounded-lg border hover:bg-muted transition-colors"
        >
          <FileText className="h-5 w-5 text-primary" />
          <div>
            <p className="font-medium">Audit Logs</p>
            <p className="text-xs text-muted-foreground">Review system activity</p>
          </div>
        </Link>
        <Link
          href="/roles"
          className="flex items-center gap-3 p-4 rounded-lg border hover:bg-muted transition-colors"
        >
          <Shield className="h-5 w-5 text-primary" />
          <div>
            <p className="font-medium">Roles & Permissions</p>
            <p className="text-xs text-muted-foreground">Manage access</p>
          </div>
        </Link>
        <Link
          href="/config"
          className="flex items-center gap-3 p-4 rounded-lg border hover:bg-muted transition-colors"
        >
          <Cpu className="h-5 w-5 text-primary" />
          <div>
            <p className="font-medium">Configuration</p>
            <p className="text-xs text-muted-foreground">System settings</p>
          </div>
        </Link>
        <Link
          href="/monitoring"
          className="flex items-center gap-3 p-4 rounded-lg border hover:bg-muted transition-colors"
        >
          <Activity className="h-5 w-5 text-primary" />
          <div>
            <p className="font-medium">Monitoring</p>
            <p className="text-xs text-muted-foreground">System health &amp; alerts</p>
          </div>
        </Link>
      </div>
    </CardContent>
  </Card>
);
