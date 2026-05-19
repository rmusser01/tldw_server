'use client';

import { Suspense } from 'react';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import MonitoringFeedbackBanners from './components/MonitoringFeedbackBanners';
import MonitoringManagementPanels from './components/MonitoringManagementPanels';
import MonitoringMetricsSection from './components/MonitoringMetricsSection';
import MonitoringPageHeader from './components/MonitoringPageHeader';
import { useMonitoringPageController } from './use-monitoring-page-controller';

function MonitoringPageContent() {
  const {
    headerProps,
    feedbackBannersProps,
    metricsSectionProps,
    managementPanelsProps,
  } = useMonitoringPageController();

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        <div className="p-4 lg:p-8">
          <MonitoringPageHeader {...headerProps} />
          <MonitoringFeedbackBanners {...feedbackBannersProps} />
          <MonitoringMetricsSection {...metricsSectionProps} />
          <MonitoringManagementPanels {...managementPanelsProps} />
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}

// Suspense boundary required: useMonitoringPageController uses useSearchParams(),
// which Next.js requires to be wrapped in Suspense for static generation.
export default function MonitoringPage() {
  return (
    <Suspense
      fallback={
        <PermissionGuard variant="route" requireAuth role="admin">
          <ResponsiveLayout>
            <div className="p-4 lg:p-8">
              <div className="mb-8">
                <div className="mb-2 h-8 w-48 animate-pulse rounded bg-muted" />
                <div className="h-4 w-64 animate-pulse rounded bg-muted" />
              </div>
              <div className="h-96 animate-pulse rounded bg-muted" />
            </div>
          </ResponsiveLayout>
        </PermissionGuard>
      }
    >
      <MonitoringPageContent />
    </Suspense>
  );
}
