'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { PermissionGuard } from '@/components/PermissionGuard';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useToast } from '@/components/ui/toast';
import { PlanBadge } from '@/components/PlanBadge';
import { api } from '@/lib/api-client';
import { isBillingEnabled } from '@/lib/billing';
import type { Plan } from '@/types';
import { Check } from 'lucide-react';

const orgSchema = z.object({
  name: z.string().min(1, 'Organization name is required'),
  slug: z
    .string()
    .min(1, 'Slug is required')
    .regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/, 'Slug must contain only lowercase letters, numbers, and hyphens'),
  owner_email: z.string().email('Invalid email address').optional().or(z.literal('')),
});

type OrgFormData = z.infer<typeof orgSchema>;

const STEPS = [
  { number: 1, label: 'Organization' },
  { number: 2, label: 'Select Plan' },
  { number: 3, label: 'Confirm' },
];

function StepIndicator({ currentStep }: { currentStep: number }) {
  return (
    <div
      className="flex items-center justify-center gap-2 mb-8"
      data-testid="step-indicator"
      role="group"
      aria-label={`Onboarding progress, step ${currentStep} of ${STEPS.length}`}
    >
      {STEPS.map((step, idx) => (
        <div key={step.number} className="flex items-center gap-2" aria-current={currentStep === step.number ? 'step' : undefined}>
          <div className="flex items-center gap-2">
            <div
              className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-medium ${
                currentStep > step.number
                  ? 'bg-green-600 text-white'
                  : currentStep === step.number
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-500 dark:bg-gray-700 dark:text-gray-400'
              }`}
              aria-hidden="true"
            >
              {currentStep > step.number ? <Check className="h-4 w-4" /> : step.number}
            </div>
            <span
              className={`text-sm ${
                currentStep >= step.number ? 'font-medium text-gray-900 dark:text-gray-100' : 'text-gray-500 dark:text-gray-400'
              }`}
              aria-current={currentStep === step.number ? 'step' : undefined}
            >
              {step.label}
            </span>
          </div>
          {idx < STEPS.length - 1 && (
            <div className="mx-2 h-px w-12 bg-gray-300 dark:bg-gray-600" aria-hidden="true" />
          )}
        </div>
      ))}
    </div>
  );
}

function OnboardingPageContent() {
  const { success, error: showError } = useToast();
  const billingEnabled = isBillingEnabled();
  const [currentStep, setCurrentStep] = useState(1);
  const [plans, setPlans] = useState<Plan[]>([]);
  const [selectedPlanId, setSelectedPlanId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const {
    register,
    getValues,
    trigger,
    formState: { errors },
  } = useForm<OrgFormData>({
    resolver: zodResolver(orgSchema),
    defaultValues: { name: '', slug: '', owner_email: '' },
  });

  const [slugAvailable, setSlugAvailable] = useState<boolean | null>(null);
  const [slugChecking, setSlugChecking] = useState(false);
  const slugCheckSequence = useRef(0);
  const slugCheckPromise = useRef<Promise<boolean | null> | null>(null);
  const latestRequestedSlug = useRef<string | null>(null);
  const latestResolvedSlug = useRef<string | null>(null);
  const latestResolvedAvailability = useRef<boolean | null>(null);

  const checkSlugAvailability = useCallback(async (slug: string): Promise<boolean | null> => {
    const normalizedSlug = slug.trim();
    const sequence = ++slugCheckSequence.current;
    latestRequestedSlug.current = normalizedSlug;
    if (!normalizedSlug || normalizedSlug.length < 2) {
      if (sequence === slugCheckSequence.current) {
        setSlugAvailable(null);
        setSlugChecking(false);
        latestResolvedSlug.current = normalizedSlug;
        latestResolvedAvailability.current = null;
        slugCheckPromise.current = null;
      }
      return null;
    }
    setSlugChecking(true);
    const requestPromise = (async () => {
      try {
        const orgs = await api.getOrganizations({ q: normalizedSlug });
        const taken = (Array.isArray(orgs) ? orgs : []).some(
          (o: { slug?: string }) => o.slug === normalizedSlug
        );
        const available = !taken;
        if (sequence === slugCheckSequence.current) {
          setSlugAvailable(available);
          latestResolvedSlug.current = normalizedSlug;
          latestResolvedAvailability.current = available;
        }
        return available;
      } catch {
        if (sequence === slugCheckSequence.current) {
          setSlugAvailable(null);
          latestResolvedSlug.current = normalizedSlug;
          latestResolvedAvailability.current = null;
        }
        return null;
      } finally {
        if (sequence === slugCheckSequence.current) {
          setSlugChecking(false);
          slugCheckPromise.current = null;
        }
      }
    })();
    slugCheckPromise.current = requestPromise;
    return requestPromise;
  }, []);

  const fetchPlans = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.getPlans();
      setPlans(data);
    } catch {
      showError('Failed to load plans');
    } finally {
      setLoading(false);
    }
  }, [showError]);

  useEffect(() => {
    if (billingEnabled) {
      fetchPlans();
    }
  }, [billingEnabled, fetchPlans]);

  if (!billingEnabled) {
    return (
      <div className="mx-auto max-w-2xl p-8">
        <Alert>
          <AlertDescription>Billing is not enabled</AlertDescription>
        </Alert>
      </div>
    );
  }

  const handleNext = async () => {
    if (currentStep === 1) {
      const valid = await trigger(['name', 'slug', 'owner_email']);
      if (!valid) return;
      const slug = getValues('slug').trim();
      const availability =
        latestResolvedSlug.current === slug
          ? latestResolvedAvailability.current
          : latestRequestedSlug.current === slug && slugCheckPromise.current
            ? await slugCheckPromise.current
            : await checkSlugAvailability(slug);
      if (availability !== true) {
        showError(availability === false ? 'Slug is already taken' : 'Unable to verify slug availability');
        return;
      }
      setCurrentStep(2);
    } else if (currentStep === 2) {
      if (!selectedPlanId) {
        showError('Please select a plan');
        return;
      }
      setCurrentStep(3);
    }
  };

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleConfirm = async () => {
    const values = getValues();
    setSubmitting(true);
    try {
      const result = await api.createOnboardingSession({
        org_name: values.name,
        org_slug: values.slug,
        plan_id: selectedPlanId!,
        owner_email: values.owner_email || undefined,
      });
      if (result.checkout_url && result.checkout_url.startsWith('https://')) {
        window.location.href = result.checkout_url;
      } else if (!result.checkout_url) {
        success('Organization created successfully');
        setSelectedPlanId(null);
        setCurrentStep(1);
      }
    } catch {
      showError('Failed to create organization');
    } finally {
      setSubmitting(false);
    }
  };

  const selectedPlan = plans.find((p) => p.id === selectedPlanId);
  const formValues = getValues();

  return (
    <div className="mx-auto max-w-3xl p-8">
      <h1 className="mb-2 text-2xl font-bold">Onboarding</h1>
      <p className="mb-6 text-gray-500 dark:text-gray-400">Set up your organization and select a plan.</p>

      <StepIndicator currentStep={currentStep} />

      {/* Step 1: Organization Details */}
      {currentStep === 1 && (
        <Card>
          <CardHeader>
            <CardTitle>Organization Details</CardTitle>
            <CardDescription>Enter the details for your new organization.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="org-name">Organization Name</Label>
              <Input
                id="org-name"
                data-testid="org-name-input"
                placeholder="My Organization"
                {...register('name')}
              />
              {errors.name && <p className="mt-1 text-sm text-red-600">{errors.name.message}</p>}
            </div>
            <div>
              <Label htmlFor="org-slug">Slug</Label>
              <Input
                id="org-slug"
                data-testid="org-slug-input"
                placeholder="my-organization"
                {...register('slug', {
                  onBlur: (e) => { void checkSlugAvailability(e.target.value); },
                })}
              />
              {errors.slug && <p className="mt-1 text-sm text-red-600">{errors.slug.message}</p>}
              {slugChecking && <p className="mt-1 text-xs text-muted-foreground">Checking availability...</p>}
              {slugAvailable === true && !errors.slug && <p className="mt-1 text-xs text-green-600">Slug is available</p>}
              {slugAvailable === false && !errors.slug && <p className="mt-1 text-xs text-red-600">Slug is already taken</p>}
            </div>
            <div>
              <Label htmlFor="owner-email">Owner Email (optional)</Label>
              <Input
                id="owner-email"
                data-testid="owner-email-input"
                type="email"
                placeholder="admin@example.com"
                {...register('owner_email')}
              />
              {errors.owner_email && (
                <p className="mt-1 text-sm text-red-600">{errors.owner_email.message}</p>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 2: Select Plan */}
      {currentStep === 2 && (
        <div>
          <h2 className="mb-4 text-lg font-semibold">Select a Plan</h2>
          {loading ? (
            <p role="status" aria-live="polite">Loading plans...</p>
          ) : plans.length === 0 ? (
            <p>No plans available.</p>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {plans.map((plan) => (
                <Card
                  key={plan.id}
                  data-testid={`plan-card-${plan.id}`}
                  className={`cursor-pointer transition-colors ${
                    selectedPlanId === plan.id
                      ? 'border-blue-500 ring-2 ring-blue-500'
                      : 'hover:border-gray-400'
                  }`}
                  onClick={() => setSelectedPlanId(plan.id)}
                >
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-base">{plan.name}</CardTitle>
                      <PlanBadge tier={plan.tier} />
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-2xl font-bold">
                      ${(plan.monthly_price_cents / 100).toFixed(2)}
                      <span className="text-sm font-normal text-gray-500">/mo</span>
                    </p>
                    <p className="mt-2 text-sm text-gray-500">
                      {plan.included_token_credits.toLocaleString()} token credits included
                    </p>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Step 3: Confirm */}
      {currentStep === 3 && (
        <Card>
          <CardHeader>
            <CardTitle>Confirm</CardTitle>
            <CardDescription>Review your details and confirm.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-2 text-sm">
              <div className="flex justify-between">
                <span className="font-medium">Organization Name</span>
                <span>{formValues.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="font-medium">Slug</span>
                <span>{formValues.slug}</span>
              </div>
              {formValues.owner_email && (
                <div className="flex justify-between">
                  <span className="font-medium">Owner Email</span>
                  <span>{formValues.owner_email}</span>
                </div>
              )}
              {selectedPlan && (
                <>
                  <hr className="my-2" />
                  <div className="flex justify-between">
                    <span className="font-medium">Plan</span>
                    <span className="flex items-center gap-2">
                      {selectedPlan.name} <PlanBadge tier={selectedPlan.tier} />
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium">Price</span>
                    <span>${(selectedPlan.monthly_price_cents / 100).toFixed(2)}/mo</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium">Token Credits</span>
                    <span>{selectedPlan.included_token_credits.toLocaleString()}</span>
                  </div>
                </>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Navigation */}
      <div className="mt-6 flex justify-between">
        <Button variant="outline" onClick={handleBack} disabled={currentStep === 1}>
          Back
        </Button>
        {currentStep < 3 ? (
          <Button onClick={handleNext}>Next</Button>
        ) : (
          <Button onClick={handleConfirm} disabled={submitting}>
            {submitting ? 'Creating...' : 'Create Organization'}
          </Button>
        )}
      </div>
    </div>
  );
}

export default function OnboardingPage() {
  return (
    <PermissionGuard role={['admin', 'super_admin', 'owner']}>
      <OnboardingPageContent />
    </PermissionGuard>
  );
}
