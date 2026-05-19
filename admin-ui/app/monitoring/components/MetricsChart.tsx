import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Activity } from 'lucide-react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { MetricsHistoryPoint } from '../types';
import type {
  MonitoringMetricSeriesKey,
  MonitoringMetricsSeriesVisibility,
} from '@/lib/monitoring-metrics';

type MetricsChartProps = {
  metricsHistory: MetricsHistoryPoint[];
  rangeLabel: string;
  seriesVisibility: MonitoringMetricsSeriesVisibility;
  onToggleSeries: (series: MonitoringMetricSeriesKey) => void;
};

type SeriesDefinition = {
  key: MonitoringMetricSeriesKey;
  label: string;
  color: string;
  axis: 'percent' | 'count';
};

const SERIES_DEFINITIONS: SeriesDefinition[] = [
  { key: 'cpu', label: 'CPU %', color: 'hsl(var(--chart-1))', axis: 'percent' },
  { key: 'memory', label: 'Memory %', color: 'hsl(var(--chart-2))', axis: 'percent' },
  { key: 'diskUsage', label: 'Disk %', color: 'hsl(var(--chart-3))', axis: 'percent' },
  { key: 'throughput', label: 'Throughput', color: 'hsl(var(--chart-4))', axis: 'count' },
  { key: 'activeConnections', label: 'Connections', color: 'hsl(var(--chart-5))', axis: 'count' },
  { key: 'queueDepth', label: 'Queue Depth', color: '#ef4444', axis: 'count' },
];

const formatCount = (value: unknown): string => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '0';
  if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
  if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
  return Math.round(value).toString();
};

export default function MetricsChart({
  metricsHistory,
  rangeLabel,
  seriesVisibility,
  onToggleSeries,
}: MetricsChartProps) {
  const visibleSeries = SERIES_DEFINITIONS.filter((series) => seriesVisibility[series.key]);

  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5" />
          System Metrics ({rangeLabel})
        </CardTitle>
        <CardDescription>
          CPU, memory, disk usage, throughput, active connections, and queue depth.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="mb-3 flex flex-wrap gap-2" role="group" aria-label="Metric series toggles">
          {SERIES_DEFINITIONS.map((series) => {
            const isVisible = seriesVisibility[series.key];
            return (
              <Button
                key={series.key}
                type="button"
                size="sm"
                variant={isVisible ? 'secondary' : 'outline'}
                aria-pressed={isVisible}
                data-testid={`metric-series-toggle-${series.key}`}
                onClick={() => onToggleSeries(series.key)}
                className="h-8 gap-2"
              >
                <span
                  className="h-2.5 w-2.5 rounded-full"
                  style={{ backgroundColor: series.color }}
                  aria-hidden="true"
                />
                {series.label}
              </Button>
            );
          })}
        </div>

        {metricsHistory.length === 0 ? (
          <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
            Collecting metrics data...
          </div>
        ) : visibleSeries.length === 0 ? (
          <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
            Enable at least one metric series to render the chart.
          </div>
        ) : (
          <>
          <div role="img" aria-label={`System metrics chart showing ${metricsHistory.length} data points for ${visibleSeries.map((s) => s.label).join(', ')} over ${rangeLabel}`}>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%" minHeight={1} minWidth={1}>
                <LineChart data={metricsHistory}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="label" className="text-xs" minTickGap={24} />
                  <YAxis
                    yAxisId="percent"
                    className="text-xs"
                    domain={[0, 100]}
                    tickFormatter={(value) => `${value}%`}
                  />
                  <YAxis
                    yAxisId="count"
                    orientation="right"
                    className="text-xs"
                    tickFormatter={(value) => formatCount(value)}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--background))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                    }}
                    formatter={(value, name, item) => {
                      const key = String(item?.dataKey ?? name ?? '');
                      if (['cpu', 'memory', 'diskUsage'].includes(key)) {
                        const percent = typeof value === 'number' && Number.isFinite(value) ? value.toFixed(1) : '—';
                        return [`${percent}%`, SERIES_DEFINITIONS.find((series) => series.key === key)?.label ?? key];
                      }
                      return [formatCount(value), SERIES_DEFINITIONS.find((series) => series.key === key)?.label ?? key];
                    }}
                  />
                  {visibleSeries.map((series) => (
                    <Line
                      key={series.key}
                      type="monotone"
                      dataKey={series.key}
                      yAxisId={series.axis}
                      stroke={series.color}
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                      name={series.label}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <details className="mt-2">
            <summary className="text-xs text-muted-foreground cursor-pointer">
              View chart data as table
            </summary>
            <div className="mt-1 max-h-48 overflow-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr>
                    <th className="text-left px-2 py-1">Time</th>
                    {visibleSeries.map((series) => (
                      <th key={series.key} className="text-right px-2 py-1">{series.label}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {metricsHistory.map((point, i) => (
                    <tr key={i} className="border-t border-border">
                      <td className="px-2 py-1">{point.label}</td>
                      {visibleSeries.map((series) => (
                        <td key={series.key} className="text-right px-2 py-1">
                          {series.axis === 'percent'
                            ? `${typeof point[series.key] === 'number' ? point[series.key].toFixed(1) : point[series.key]}%`
                            : formatCount(point[series.key])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
          </>
        )}
      </CardContent>
    </Card>
  );
}
