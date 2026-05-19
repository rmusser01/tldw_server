'use client';

import { useCallback, useEffect, useState } from 'react';
import { useForm, FormProvider } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { PlanBadge } from '@/components/PlanBadge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { useToast } from '@/components/ui/toast';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { usePrivilegedActionDialog } from '@/components/ui/privileged-action-dialog';
import { CardSkeleton } from '@/components/ui/skeleton';
import { Form, FormInput, FormSelect } from '@/components/ui/form';
import { CreditCard, Plus, Pencil, Trash2 } from 'lucide-react';
import { api } from '@/lib/api-client';
import { isBillingEnabled } from '@/lib/billing';
import { Plan } from '@/types';
import { logger } from '@/lib/logger';

const PLAN_TIERS = ['free', 'pro', 'enterprise'] as const;

const planSchema = z.object({
  name: z.string().min(1, 'Plan name is required'),
  tier: z.enum(PLAN_TIERS),
  monthly_price_cents: z.coerce.number().int().min(0, 'Price must be non-negative'),
  included_token_credits: z.coerce.number().int().min(0, 'Must be non-negative'),
  overage_rate_per_1k_tokens_cents: z.coerce.number().int().min(0, 'Must be non-negative'),
  stripe_product_id: z.string().optional(),
  stripe_price_id: z.string().optional(),
});

type PlanFormInput = z.input<typeof planSchema>;
type PlanFormData = z.output<typeof planSchema>;

const tierOptions = [
  { value: 'free', label: 'Free' },
  { value: 'pro', label: 'Pro' },
  { value: 'enterprise', label: 'Enterprise' },
];

function formatCents(cents: number): string {
  return `$${(cents / 100).toFixed(2)}`;
}

export default function PlansPage() {
  const [plans, setPlans] = useState<Plan[]>([]);
  const [compareMode, setCompareMode] = useState(false);
  const [loading, setLoading] = useState(true);
  const { success, error: showError } = useToast();
  const confirm = useConfirm();
  const promptPrivilegedAction = usePrivilegedActionDialog();

  const [showDialog, setShowDialog] = useState(false);
  const [editingPlan, setEditingPlan] = useState<Plan | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const form = useForm<PlanFormInput, unknown, PlanFormData>({
    resolver: zodResolver(planSchema),
    defaultValues: {
      name: '',
      tier: 'free',
      monthly_price_cents: 0,
      included_token_credits: 0,
      overage_rate_per_1k_tokens_cents: 0,
      stripe_product_id: '',
      stripe_price_id: '',
    },
  });

  const billingEnabled = isBillingEnabled();

  const loadPlans = useCallback(async () => {
    try {
      setLoading(true);
      const data = await api.getPlans();
      setPlans(Array.isArray(data) ? data : []);
    } catch (err: unknown) {
      logger.error('Failed to load plans', { component: 'PlansPage', error: err instanceof Error ? err.message : String(err) });
      showError('Failed to load plans', err instanceof Error ? err.message : 'Please try again.');
    } finally {
      setLoading(false);
    }
  }, [showError]);

  useEffect(() => {
    if (billingEnabled) {
      loadPlans();
    } else {
      setLoading(false);
    }
  }, [billingEnabled, loadPlans]);

  useEffect(() => {
    if (!showDialog) {
      form.reset();
      setEditingPlan(null);
    }
  }, [form, showDialog]);

  const openCreate = () => {
    setEditingPlan(null);
    form.reset({
      name: '',
      tier: 'free',
      monthly_price_cents: 0,
      included_token_credits: 0,
      overage_rate_per_1k_tokens_cents: 0,
      stripe_product_id: '',
      stripe_price_id: '',
    });
    setShowDialog(true);
  };

  const openEdit = (plan: Plan) => {
    setEditingPlan(plan);
    form.reset({
      name: plan.name,
      tier: plan.tier,
      monthly_price_cents: plan.monthly_price_cents,
      included_token_credits: plan.included_token_credits,
      overage_rate_per_1k_tokens_cents: plan.overage_rate_per_1k_tokens_cents,
      stripe_product_id: plan.stripe_product_id || '',
      stripe_price_id: plan.stripe_price_id || '',
    });
    setShowDialog(true);
  };

  const handleSubmit = form.handleSubmit(async (data) => {
    try {
      setSubmitting(true);
      if (editingPlan) {
        await api.updatePlan(editingPlan.id, data);
        success('Plan Updated', `Plan "${data.name}" has been updated`);
      } else {
        await api.createPlan(data);
        success('Plan Created', `Plan "${data.name}" has been created`);
      }
      setShowDialog(false);
      form.reset();
      loadPlans();
    } catch (err: unknown) {
      logger.error('Failed to save plan', { component: 'PlansPage', error: err instanceof Error ? err.message : String(err) });
      showError('Failed to save plan', err instanceof Error ? err.message : 'Please try again.');
    } finally {
      setSubmitting(false);
    }
  });

  const handleDelete = async (plan: Plan) => {
    let subscriberCount = 0;
    let subscriberCheckFailed = false;

    if (billingEnabled) {
      try {
        const subs = await api.getSubscriptions({ plan_id: String(plan.id) });
        const allSubs = Array.isArray(subs) ? subs : [];
        subscriberCount = allSubs.filter(
          (s) => s.status === 'active' || s.status === 'trialing'
        ).length;
      } catch (err: unknown) {
        logger.error('Failed to check plan subscribers', { component: 'PlansPage', error: err instanceof Error ? err.message : String(err) });
        subscriberCheckFailed = true;
      }
    }

    let message = `Are you sure you want to delete the plan "${plan.name}"? This action cannot be undone.`;
    if (subscriberCount > 0) {
      message = `This plan has ${subscriberCount} active subscription(s). Deleting it will affect these organizations.\n\n${message}`;
    } else if (subscriberCheckFailed) {
      message = `Warning: Could not verify whether this plan has active subscribers.\n\n${message}`;
    }

    const approval = await promptPrivilegedAction({
      title: 'Delete Plan',
      message,
      confirmText: 'Delete',
      requirePassword: true,
    });
    if (!approval) return;

    try {
      await api.deletePlan(plan.id);
      success('Plan Deleted', `Plan "${plan.name}" has been deleted`);
      loadPlans();
    } catch (err: unknown) {
      logger.error('Failed to delete plan', { component: 'PlansPage', error: err instanceof Error ? err.message : String(err) });
      showError('Failed to delete plan', err instanceof Error ? err.message : 'Please try again.');
    }
  };

  return (
    <PermissionGuard role={['admin', 'super_admin', 'owner']}>
      <ResponsiveLayout>
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
                <CreditCard className="h-6 w-6" />
                Plans
              </h1>
              <p className="text-muted-foreground">
                Manage subscription plans and pricing
              </p>
            </div>
            <div className="flex gap-2">
              {billingEnabled && plans.length > 1 && (
                <Button variant={compareMode ? 'default' : 'outline'} onClick={() => setCompareMode(!compareMode)}>
                  {compareMode ? 'Card View' : 'Compare'}
                </Button>
              )}
              {billingEnabled && (
                <Button onClick={openCreate}>
                  <Plus className="mr-2 h-4 w-4" />
                  Create Plan
                </Button>
              )}
            </div>
          </div>

          {!billingEnabled ? (
            <Card>
              <CardContent className="py-12 text-center">
                <p className="text-muted-foreground">
                  Billing is not enabled. Set <code>NEXT_PUBLIC_BILLING_ENABLED=true</code> to manage plans.
                </p>
              </CardContent>
            </Card>
          ) : loading ? (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              <CardSkeleton />
              <CardSkeleton />
              <CardSkeleton />
            </div>
          ) : plans.length === 0 ? (
            <Card>
              <CardContent className="py-12 text-center">
                <p className="text-muted-foreground">No plans found. Create your first plan to get started.</p>
              </CardContent>
            </Card>
          ) : compareMode ? (
            <Card>
              <CardContent className="pt-6 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2 font-semibold">Feature</th>
                      {plans.map(p => (
                        <th key={p.id} className="text-center p-2 font-semibold">
                          {p.name} <PlanBadge tier={p.tier} />
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b">
                      <td className="p-2 text-muted-foreground">Price</td>
                      {plans.map(p => <td key={p.id} className="text-center p-2 font-medium">{formatCents(p.monthly_price_cents)}/mo</td>)}
                    </tr>
                    <tr className="border-b">
                      <td className="p-2 text-muted-foreground">Token Credits</td>
                      {plans.map(p => <td key={p.id} className="text-center p-2">{p.included_token_credits.toLocaleString()}</td>)}
                    </tr>
                    <tr className="border-b">
                      <td className="p-2 text-muted-foreground">Overage Rate</td>
                      {plans.map(p => <td key={p.id} className="text-center p-2">{formatCents(p.overage_rate_per_1k_tokens_cents)}/1k</td>)}
                    </tr>
                    <tr>
                      <td className="p-2 text-muted-foreground">Features</td>
                      {plans.map(p => <td key={p.id} className="text-center p-2">{p.features?.length ?? 0}</td>)}
                    </tr>
                  </tbody>
                </table>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {plans.map((plan) => (
                <Card key={plan.id}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">{plan.name}</CardTitle>
                      <PlanBadge tier={plan.tier} />
                    </div>
                    <CardDescription>
                      {formatCents(plan.monthly_price_cents)}/mo
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <dl className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <dt className="text-muted-foreground">Included tokens</dt>
                        <dd>{plan.included_token_credits.toLocaleString()}</dd>
                      </div>
                      <div className="flex justify-between">
                        <dt className="text-muted-foreground">Overage rate</dt>
                        <dd>{formatCents(plan.overage_rate_per_1k_tokens_cents)}/1k tokens</dd>
                      </div>
                      {plan.overage_rate_per_1k_tokens_cents > 0 && (
                        <p className="text-xs italic text-muted-foreground">
                          e.g., 100k excess tokens = {formatCents(plan.overage_rate_per_1k_tokens_cents * 100)}
                        </p>
                      )}
                      <div className="flex justify-between">
                        <dt className="text-muted-foreground">Features</dt>
                        <dd>{plan.features?.length ?? 0}</dd>
                      </div>
                    </dl>
                    <div className="mt-4 flex gap-2">
                      <Button variant="outline" size="sm" onClick={() => openEdit(plan)}>
                        <Pencil className="mr-1 h-3 w-3" />
                        Edit
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => handleDelete(plan)}>
                        <Trash2 className="mr-1 h-3 w-3" />
                        Delete
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>

        <Dialog open={showDialog} onOpenChange={setShowDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>{editingPlan ? 'Edit Plan' : 'Create Plan'}</DialogTitle>
              <DialogDescription>
                {editingPlan ? 'Update plan details.' : 'Add a new subscription plan.'}
              </DialogDescription>
            </DialogHeader>
            <FormProvider {...form}>
              <Form onSubmit={handleSubmit}>
                <FormInput<PlanFormData> name="name" label="Name" required placeholder="e.g. Pro" />
                <FormSelect<PlanFormData> name="tier" label="Tier" required options={tierOptions} />
                <FormInput<PlanFormData> name="monthly_price_cents" label="Monthly Price (cents)" type="number" required />
                <FormInput<PlanFormData> name="included_token_credits" label="Included Token Credits" type="number" required />
                <FormInput<PlanFormData> name="overage_rate_per_1k_tokens_cents" label="Overage Rate per 1k Tokens (cents)" type="number" required />
                <FormInput<PlanFormData> name="stripe_product_id" label="Stripe Product ID" placeholder="prod_..." />
                <FormInput<PlanFormData> name="stripe_price_id" label="Stripe Price ID" placeholder="price_..." />
                <DialogFooter>
                  <Button type="button" variant="outline" onClick={() => setShowDialog(false)}>
                    Cancel
                  </Button>
                  <Button type="submit" disabled={submitting}>
                    {submitting ? 'Saving...' : editingPlan ? 'Save Changes' : 'Create Plan'}
                  </Button>
                </DialogFooter>
              </Form>
            </FormProvider>
          </DialogContent>
        </Dialog>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
