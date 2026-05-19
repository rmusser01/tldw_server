'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Users, RefreshCw, XCircle } from 'lucide-react';
import { api } from '@/lib/api-client';
import { useToast } from '@/components/ui/toast';
import { useConfirm } from '@/components/ui/confirm-dialog';
import type { VoiceSession } from '@/types';
import { formatDistanceToNow } from 'date-fns';
import { logger } from '@/lib/logger';

interface ActiveSessionsPanelProps {
  onSessionEnded?: () => void;
}

export function ActiveSessionsPanel({ onSessionEnded }: ActiveSessionsPanelProps) {
  const [sessions, setSessions] = useState<VoiceSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [endingSessionId, setEndingSessionId] = useState<string | null>(null);
  const { success, error: showError } = useToast();
  const confirm = useConfirm();

  const loadSessions = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getVoiceSessions();
      const items = Array.isArray(data) ? data : (data?.sessions || data?.items || []);
      setSessions(items);
    } catch (err) {
      logger.warn('Failed to load voice sessions', { component: 'ActiveSessionsPanel', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error ? err.message : 'Failed to load sessions');
      setSessions([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSessions();
    // Refresh every 30 seconds
    const interval = setInterval(loadSessions, 30000);
    return () => clearInterval(interval);
  }, [loadSessions]);

  const handleEndSession = async (session: VoiceSession) => {
    if (endingSessionId === session.session_id) return;
    const confirmed = await confirm({
      title: 'End Voice Session',
      message: `End session for user ${session.user_id}? This will disconnect their voice assistant.`,
      confirmText: 'End Session',
      variant: 'danger',
      icon: 'warning',
    });
    if (!confirmed) return;

    try {
      setEndingSessionId(session.session_id);
      await api.deleteVoiceSession(session.session_id);
      success('Session ended', 'Voice session has been terminated.');
      void loadSessions();
      onSessionEnded?.();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to end session';
      showError('End session failed', message);
    } finally {
      setEndingSessionId((prev) => (prev === session.session_id ? null : prev));
    }
  };

  const getStateBadgeVariant = (state: string) => {
    switch (state) {
      case 'listening':
        return 'default';
      case 'processing':
        return 'secondary';
      case 'speaking':
        return 'outline';
      case 'awaiting_confirmation':
        return 'destructive';
      case 'error':
        return 'destructive';
      default:
        return 'secondary';
    }
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Active Sessions
          </CardTitle>
          <CardDescription>Currently connected voice assistant sessions</CardDescription>
        </div>
        <Button variant="outline" size="sm" onClick={loadSessions} disabled={loading}>
          <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </CardHeader>
      <CardContent>
        {error ? (
          <div className="text-sm text-destructive py-4 text-center">{error}</div>
        ) : loading && sessions.length === 0 ? (
          <div className="text-sm text-muted-foreground py-4 text-center" role="status" aria-live="polite">Loading sessions...</div>
        ) : sessions.length === 0 ? (
          <div className="text-sm text-muted-foreground py-4 text-center">No active sessions</div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>User ID</TableHead>
                <TableHead>State</TableHead>
                <TableHead>Turn Count</TableHead>
                <TableHead>Started</TableHead>
                <TableHead>Last Activity</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {sessions.map((session) => (
                <TableRow key={session.session_id}>
                  <TableCell className="font-medium">{session.user_id}</TableCell>
                  <TableCell>
                    <Badge variant={getStateBadgeVariant(session.state)}>{session.state}</Badge>
                  </TableCell>
                  <TableCell>{session.turn_count}</TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {formatDistanceToNow(new Date(session.created_at), { addSuffix: true })}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {formatDistanceToNow(new Date(session.last_activity), { addSuffix: true })}
                  </TableCell>
                  <TableCell className="text-right">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleEndSession(session)}
                      className="text-destructive hover:text-destructive"
                      disabled={endingSessionId === session.session_id}
                      loading={endingSessionId === session.session_id}
                      loadingText="Ending..."
                    >
                      <XCircle className="h-4 w-4 mr-1" />
                      End
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}
