'use client';

import { Fragment, useCallback, useEffect, useMemo, useState } from 'react';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { TableSkeleton } from '@/components/ui/skeleton';
import { useToast } from '@/components/ui/toast';
import { Check, X, LayoutGrid } from 'lucide-react';
import { api } from '@/lib/api-client';
import { isBillingEnabled } from '@/lib/billing';
import type { FeatureRegistryEntry, Plan } from '@/types';
import { logger } from '@/lib/logger';

export default function FeatureRegistryPage() {
  const [features, setFeatures] = useState<FeatureRegistryEntry[]>([]);
  const [plans, setPlans] = useState<Plan[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);
  const { success, error: showError } = useToast();

  const billingEnabled = isBillingEnabled();

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      const [featuresData, plansData] = await Promise.all([
        api.getFeatureRegistry(),
        api.getPlans(),
      ]);
      setFeatures(Array.isArray(featuresData) ? featuresData : []);
      setPlans(Array.isArray(plansData) ? plansData : []);
      setDirty(false);
    } catch (err: unknown) {
      logger.error('Failed to load feature registry', { component: 'FeatureRegistryPage', error: err instanceof Error ? err.message : String(err) });
      showError('Failed to load data', err instanceof Error ? err.message : 'Please try again.');
    } finally {
      setLoading(false);
    }
  }, [showError]);

  useEffect(() => {
    if (billingEnabled) {
      loadData();
    } else {
      setLoading(false);
    }
  }, [billingEnabled, loadData]);

  const categories = useMemo(() => {
    const cats = new Map<string, FeatureRegistryEntry[]>();
    for (const feature of features) {
      const cat = feature.category || 'Uncategorized';
      if (!cats.has(cat)) {
        cats.set(cat, []);
      }
      cats.get(cat)!.push(feature);
    }
    return cats;
  }, [features]);

  const toggleFeaturePlan = (featureKey: string, planId: string) => {
    setFeatures((prev) =>
      prev.map((f) => {
        if (f.feature_key !== featureKey) return f;
        const hasPlan = f.plans.includes(planId);
        return {
          ...f,
          plans: hasPlan
            ? f.plans.filter((p) => p !== planId)
            : [...f.plans, planId],
        };
      })
    );
    setDirty(true);
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      await api.updateFeatureRegistry(features);
      success('Changes Saved', 'Feature registry has been updated.');
      setDirty(false);
    } catch (err: unknown) {
      logger.error('Failed to save feature registry', { component: 'FeatureRegistryPage', error: err instanceof Error ? err.message : String(err) });
      showError('Failed to save', err instanceof Error ? err.message : 'Please try again.');
    } finally {
      setSaving(false);
    }
  };

  if (!billingEnabled) {
    return (
      <PermissionGuard variant="route" requireAuth role="admin">
        <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            <h1 className="text-3xl font-bold mb-4">Feature Registry</h1>
            <Card>
              <CardContent className="pt-6">
                <p className="text-muted-foreground">Billing is not enabled</p>
              </CardContent>
            </Card>
          </div>
        </ResponsiveLayout>
      </PermissionGuard>
    );
  }

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        <div className="p-4 lg:p-8">
          <div className="mb-8 flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">Feature Registry</h1>
              <p className="text-muted-foreground">
                Manage which features are included in each plan
              </p>
            </div>
            {dirty && (
              <Button onClick={handleSave} loading={saving} loadingText="Saving...">
                Save Changes
              </Button>
            )}
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <LayoutGrid className="h-5 w-5" />
                Feature-Plan Matrix
              </CardTitle>
            </CardHeader>
            <CardContent>
              {loading ? (
                <TableSkeleton rows={6} columns={4} />
              ) : features.length === 0 ? (
                <div className="text-center text-muted-foreground py-8">
                  No features registered yet.
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="min-w-[200px]">Feature</TableHead>
                        {plans.map((plan) => (
                          <TableHead key={plan.id} className="text-center min-w-[100px]">
                            {plan.name}
                          </TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {Array.from(categories.entries()).map(([category, categoryFeatures]) => (
                        <Fragment key={category}>
                          <TableRow data-testid={`category-${category}`}>
                            <TableCell
                              colSpan={plans.length + 1}
                              className="bg-muted/50 font-semibold text-sm"
                            >
                              {category}
                            </TableCell>
                          </TableRow>
                          {categoryFeatures.map((feature) => (
                            <TableRow key={feature.feature_key}>
                              <TableCell>
                                <div>
                                  <div className="font-medium">{feature.display_name}</div>
                                  {feature.description && (
                                    <div className="text-xs text-muted-foreground">
                                      {feature.description}
                                    </div>
                                  )}
                                </div>
                              </TableCell>
                              {plans.map((plan) => {
                                const included = feature.plans.includes(plan.id);
                                return (
                                  <TableCell key={plan.id} className="text-center">
                                    <button
                                      type="button"
                                      onClick={() => toggleFeaturePlan(feature.feature_key, plan.id)}
                                      className="inline-flex items-center justify-center p-1 rounded hover:bg-muted"
                                      aria-label={`Toggle ${feature.display_name} for ${plan.name}`}
                                    >
                                      {included ? (
                                        <Check className="h-5 w-5 text-green-600" />
                                      ) : (
                                        <X className="h-5 w-5 text-muted-foreground" />
                                      )}
                                    </button>
                                  </TableCell>
                                );
                              })}
                            </TableRow>
                          ))}
                        </Fragment>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
