## admin-ui Release Checklist

This checklist covers release readiness for the admin UI (Next.js). Treat it as a living document and trim/expand based on scope.

---

## 1. Versioning & Metadata

- [ ] Bump the version in `admin-ui/package.json` and confirm release tags/notes match.
- [ ] Update `admin-ui/README.md` with user-visible changes and setup steps.
- [ ] Document new environment variables in `admin-ui/README.md` (and `.env.example` if present).
- [ ] Verify any UI-visible version strings or “What’s New” notes are current.

---

## 2. Documentation Health

- [ ] Walk through the `admin-ui/README.md` setup from a clean clone (install → dev → build).
- [ ] Verify screenshots, feature lists, and links are current.
- [ ] Update any admin UI notes or audit docs to reflect major changes.

---

## 3. Code Review & Hygiene

- [ ] Remove debug code, temporary flags, and unused feature toggles.
- [ ] Ensure components follow existing patterns for hooks, layout, and state handling.
- [ ] Confirm type coverage and lint rules are satisfied.
- [ ] Review accessibility annotations (aria labels, focus states, keyboard navigation).
- [ ] Validate client-only code is gated appropriately for Next.js.

---

## 4. Build & Install

- [ ] Run `bun install` from `admin-ui/`.
- [ ] Run `bun run lint` and address warnings.
- [ ] Run `bun run typecheck` and fix all TypeScript errors.
- [ ] Run `bun run build` and confirm the build completes without errors.
- [ ] Start the production build (`bun run start`) and confirm pages load correctly.

---

## 5. Test Matrix (Frontend)

- [ ] Run unit tests (`bun run test` or `bunx vitest run`).
- [ ] Run accessibility coverage (`bun run test:a11y`).
- [ ] Run browser smoke coverage (`bun run test:smoke`).
- [ ] Confirm React Testing Library coverage for new UI flows and error states.
- [ ] If snapshots are used, review and update them intentionally.
- [ ] Verify key pages (dashboard, monitoring, auth/config views) render without console errors.
- [ ] Verify the smoke suite still covers password login, MFA completion, and privileged user actions.

---

## 6. Backend Integration

- [ ] Verify admin auth and role gating against the current backend (single-user and multi-user if supported).
- [ ] Confirm API base URL configuration works in dev and prod (`NEXT_PUBLIC_API_URL`).
- [ ] Validate common admin flows (users, orgs, teams, monitoring) against the backend.
- [ ] Check for breaking changes in API schemas or routes and update UI accordingly.

---

## 6a. Feature-Specific Validation

### Resource Governor

- [ ] Policy CRUD (create, edit, delete) completes without errors.
- [ ] Policy resolution shows correct winner for user/org/global scopes.
- [ ] Rate limit analytics card renders throttle event counts when data is present.
- [ ] User autocomplete populates in the policy resolution section.
- [ ] Simulation impact shows affected users and requests.

### Incidents

- [ ] Create, update status, and resolve an incident end-to-end.
- [ ] SLA metric cards (MTTA, MTTR, P95, resolved count) render with backend data.
- [ ] Notify stakeholders dialog opens, sends notification, and shows delivery results.
- [ ] Runbook URL link renders when set on an incident and is absent when not set.
- [ ] Post-mortem fields (root cause, impact, action items) save correctly.
- [ ] Assignment changes persist and drafts are preserved.

### Organizations

- [ ] Tab navigation (Members, Teams, Keys, Billing) works and preserves URL state.
- [ ] Member search filters the member table correctly.
- [ ] Member role changes via Select component persist to the backend.
- [ ] Billing tab shows subscription, usage meter, and invoices for enabled orgs.
- [ ] Billing data clears when navigating to an org without billing.

### Webhooks

- [ ] Create webhook with URL and event subscriptions.
- [ ] Delivery history displays status, latency, and timestamps.
- [ ] Test event sends successfully and appears in delivery log.

### Compliance

- [ ] Posture score and letter grade render correctly.
- [ ] MFA adoption and key rotation metrics reflect backend data.
- [ ] Report scheduling (create, edit, delete schedule) works.

### AI Operations

- [ ] AI spend dashboard shows costs by provider, model, and user.
- [ ] Agent session list renders with correct status and token counts.
- [ ] Token budget creation and enforcement work as configured.

---

## 6b. Regression Checks

- [ ] Dashboard page loads without console errors.
- [ ] Login, logout, and MFA flows complete successfully.
- [ ] Permission guards prevent unauthorized page access.
- [ ] Responsive layout works on mobile and tablet viewports.
- [ ] Export functionality (CSV, JSON) works for incidents and audit logs.
- [ ] Pagination controls work across all list pages.
- [ ] Toast notifications appear for success and error states.
- [ ] Privileged action dialog prompts before destructive operations.

---

## 7. Performance & UX

- [ ] Check Lighthouse or basic performance metrics for key pages.
- [ ] Verify tables/lists scale without regressions.
- [ ] Confirm loading, empty, and error states are consistent.
- [ ] Validate mobile and tablet layouts for primary flows.

---

## 8. Accessibility & Visual QA

- [ ] Keyboard navigation works for primary flows (tabs, dialogs, dropdowns).
- [ ] Focus states are visible and consistent.
- [ ] Color contrast meets WCAG AA for critical UI elements.
- [ ] Verify dialogs, toasts, and alerts are accessible.

---

## 9. Security & Privacy

- [ ] Ensure no secrets are exposed in client bundles (API keys, tokens).
- [ ] Validate auth token storage and logout flow.
- [ ] Confirm security headers (if managed at the frontend level) are intact.
- [ ] Review any new third-party dependencies for risk and license compatibility.

---

## 10. Release Hygiene

- [ ] Update changelog/release notes with UI changes and fixes.
- [ ] Close or update related issues/PRs and document known issues.
- [ ] Tag the release and verify CI/CD artifacts (if applicable).

---

### Using This Checklist

- Use a subset for small fixes and expand for major releases.
- Keep the checklist current as the admin UI evolves.
