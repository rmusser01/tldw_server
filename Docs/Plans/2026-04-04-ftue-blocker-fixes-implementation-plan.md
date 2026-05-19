# FTUE Blocker Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the 4 BLOCKER-severity issues identified in the FTUE audit so that first-time users can discover all three pages (/chat, /watchlists, /admin/monitoring), configure LLM providers from the WebUI, and see helpful error messages instead of dead-ends.

**Architecture:** Four independent but logically ordered changes: (A) flip config defaults so the setup wizard works on fresh installs, (B) register /admin/monitoring in navigation, (C) update NoProviderBanner to link to the server settings page with non-technical copy, (D) the banner copy and link target changes are the minimal viable fix for the "no provider UI" blocker — a full `/settings/providers` page is a follow-on.

**Tech Stack:** Python (FastAPI, ConfigParser), TypeScript/React (Next.js, lucide-react, react-router-dom, i18n)

---

### Task 1: Enable setup wizard by default for fresh installs (BLOCKER #1.2)

**Files:**
- Modify: `tldw_Server_API/Config_Files/config.txt` (lines 1-3)

**Step 1: Change the default config values**

In `tldw_Server_API/Config_Files/config.txt`, change:

```ini
[Setup]
enable_first_time_setup = true
setup_completed = false
```

The existing code in `setup_manager.py:403-414` (`get_setup_flags()`) already derives `needs_setup = enabled AND NOT completed`. With these defaults, `needs_setup` will be `True` on first start, causing:
- `main.py:5928` to redirect `/` → `/setup`
- `main.py:5534` to log setup instructions at startup
- All `/api/v1/setup/*` endpoints to be accessible

When setup completes, `POST /setup/complete` writes `setup_completed = true` to config.txt, so this only triggers once.

**Step 2: Verify the change is safe for existing installs**

Existing users already have their own `config.txt` with `setup_completed = true`. The shipped file is only used for fresh installs or as a reference. The `setup_completed = true` in their existing config means `needs_setup` stays `False` — no impact on existing users.

**Step 3: Commit**

```bash
git add tldw_Server_API/Config_Files/config.txt
git commit -m "fix(setup): enable first-time setup wizard by default for fresh installs

BLOCKER #1.2 - The config.txt shipped with enable_first_time_setup=false
and setup_completed=true, making it impossible for new users to see the
setup wizard without manually editing config.txt first."
```

---

### Task 2: Register /admin/monitoring in navigation (BLOCKER #6.1)

**Files:**
- Modify: `apps/packages/ui/src/routes/route-registry.tsx` (~line 103, ~line 506)
- Modify: `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts` (~line 39 imports, ~line 465)
- Modify: `apps/packages/ui/src/services/settings/ui-settings.ts` (~line 386)
- Create: `apps/tldw-frontend/extension/routes/option-admin-monitoring.tsx`
- Modify: `apps/tldw-frontend/extension/routes/route-registry.tsx` (~line 151, ~line 738)

**Step 1: Add lazy import and route to main route registry**

In `apps/packages/ui/src/routes/route-registry.tsx`:

After line 103 (`const OptionAdminRuntimeConfig = ...`), add:
```typescript
const OptionAdminMonitoring = lazy(() => import("./option-admin-monitoring"))
```

After the `/admin/runtime-config` route definition (~line 506), add:
```typescript
  {
    kind: "options",
    path: "/admin/monitoring",
    element: <OptionAdminMonitoring />,
    targets: ALL_TARGETS,
  },
```

**Step 2: Add monitoring to header shortcut items**

In `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`:

Add `Activity` to the icon imports at the top (line 2-39 import block):
```typescript
import {
  Activity,
  // ... existing imports ...
```

In the `admin` group items array (after the `admin-mlx` entry, before `settings`), add:
```typescript
      {
        id: "admin-monitoring",
        to: "/admin/monitoring",
        icon: Activity,
        labelKey: "option:header.adminMonitoring",
        labelDefault: "Monitoring",
        descriptionKey: "option:header.adminMonitoringDesc",
        descriptionDefault: "Server health alerts, security status, and system metrics"
      },
```

**Step 3: Add the ID to the HeaderShortcutId type**

In `apps/packages/ui/src/services/settings/ui-settings.ts`, add `"admin-monitoring"` to the `HEADER_SHORTCUT_IDS` array (after `"admin-mlx"` at line 386):
```typescript
  "admin-mlx",
  "admin-monitoring",
  "settings",
```

**Step 4: Create extension route wrapper**

Create `apps/tldw-frontend/extension/routes/option-admin-monitoring.tsx`:
```typescript
import OptionLayout from "@web/components/layout/WebLayout"
import MonitoringDashboardPage from "@/components/Option/Admin/MonitoringDashboardPage"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"

const OptionAdminMonitoring = () => {
  return (
    <RouteErrorBoundary routeId="admin-monitoring" routeLabel="Monitoring">
      <OptionLayout>
        <MonitoringDashboardPage />
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAdminMonitoring
```

**Step 5: Register route in extension route registry**

In `apps/tldw-frontend/extension/routes/route-registry.tsx`:

After line 151 (`const OptionAdminMlx = ...`), add:
```typescript
const OptionAdminMonitoring = lazy(() => import("./option-admin-monitoring"))
```

Add the `Activity` icon to the lucide-react import at the top of the file.

After the `/admin/mlx` route definition (~line 738), add:
```typescript
  {
    kind: "options",
    path: "/admin/monitoring",
    element: <OptionAdminMonitoring />,
    targets: ALL_TARGETS,
    nav: {
      group: "server",
      labelToken: "option:header.adminMonitoring",
      icon: Activity,
      order: 9
    }
  },
```

**Step 6: Run existing header shortcut tests**

Run:
```bash
cd apps/packages/ui && npx vitest run src/components/Layouts/__tests__/header-shortcut-descriptions.test.ts src/components/Layouts/__tests__/HeaderShortcuts.test.tsx src/services/__tests__/ui-settings.header-shortcuts.test.ts --reporter=verbose
```

Expected: All tests pass. The descriptions test only checks items in its `JARGON_IDS` list, so the new `admin-monitoring` entry won't fail it (it has a `descriptionDefault` but isn't in the jargon list).

**Step 7: Commit**

```bash
git add apps/packages/ui/src/routes/route-registry.tsx \
       apps/packages/ui/src/components/Layouts/header-shortcut-items.ts \
       apps/packages/ui/src/services/settings/ui-settings.ts \
       apps/tldw-frontend/extension/routes/option-admin-monitoring.tsx \
       apps/tldw-frontend/extension/routes/route-registry.tsx
git commit -m "fix(nav): register /admin/monitoring in route registry and navigation

BLOCKER #6.1 - The monitoring dashboard page existed but was unreachable
from the UI navigation. Users could only find it by typing the URL
directly. Added to both web and extension route registries, header
shortcut items, and the HeaderShortcutId type."
```

---

### Task 3: Fix NoProviderBanner copy and link target (BLOCKERs #4.1 & #4.8)

**Files:**
- Modify: `apps/packages/ui/src/components/Common/NoProviderBanner.tsx` (lines 33-50)
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundChat.tsx` (~lines 1024-1037, the "no models" fallback)

**Step 1: Update NoProviderBanner copy and link**

In `apps/packages/ui/src/components/Common/NoProviderBanner.tsx`:

Change the body text (line 39-42) from:
```typescript
            {t(
              "playground:noProviderBanner.body",
              "Chat requires an LLM provider API key (OpenAI, Anthropic, etc.). Add one in your server's .env file or through the admin panel, then restart."
            )}
```
To:
```typescript
            {t(
              "playground:noProviderBanner.body",
              "Chat requires an LLM provider API key (OpenAI, Anthropic, etc.). Open your server's settings to add one, then restart the server."
            )}
```

Change the Link target (line 45-49) from:
```typescript
            <Link
              to="/settings/model"
```
To:
```typescript
            <Link
              to="/settings/tldw"
```

The `/settings/tldw` page is the primary server config page where users can configure the connection, see health status, and find guidance. While it doesn't yet have an inline provider key form, it's a meaningful destination (unlike `/settings/model` which is read-only). The label "Open Settings" remains accurate.

**Step 2: Update the "No AI models available" fallback in PlaygroundChat**

Find the secondary alert in `PlaygroundChat.tsx` (~lines 1024-1037) that shows when `chatModels.length === 0`. Its copy says:

> "Add an LLM provider API key to your server's .env file and restart, then refresh models"

Change this to:

> "Add an LLM provider API key in your server settings and restart, then refresh models"

The exact location depends on the current code — search for `"No AI models available"` or `"noModelsAvailable"` in `PlaygroundChat.tsx`.

**Step 3: Commit**

```bash
git add apps/packages/ui/src/components/Common/NoProviderBanner.tsx \
       apps/packages/ui/src/components/Option/Playground/PlaygroundChat.tsx
git commit -m "fix(chat): fix NoProviderBanner dead-end link and remove .env jargon

BLOCKER #4.1 - 'Open Settings' linked to /settings/model which is
read-only. Changed to /settings/tldw which is the primary server config
page. Also replaced '.env file' references with plain language for
non-technical users."
```

---

### Task 4: Verify all changes end-to-end

**Step 1: Verify config.txt change**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_server2
head -5 tldw_Server_API/Config_Files/config.txt
```

Expected:
```
[Setup]
enable_first_time_setup = true
setup_completed = false
```

**Step 2: Verify monitoring route exists in navigation**

```bash
grep -c "admin/monitoring" apps/packages/ui/src/routes/route-registry.tsx
grep -c "admin-monitoring" apps/packages/ui/src/components/Layouts/header-shortcut-items.ts
grep -c "admin-monitoring" apps/packages/ui/src/services/settings/ui-settings.ts
grep -c "admin/monitoring" apps/tldw-frontend/extension/routes/route-registry.tsx
```

Expected: All return `1` or more.

**Step 3: Verify NoProviderBanner links to /settings/tldw**

```bash
grep "settings/tldw" apps/packages/ui/src/components/Common/NoProviderBanner.tsx
grep -c ".env" apps/packages/ui/src/components/Common/NoProviderBanner.tsx
```

Expected: First grep shows the Link, second grep returns `0` (no .env references).

**Step 4: Run all related tests**

```bash
cd apps/packages/ui && npx vitest run src/components/Layouts/__tests__/ src/services/__tests__/ui-settings.header-shortcuts.test.ts --reporter=verbose
```

Expected: All pass.

**Step 5: Commit verification notes**

No commit needed — this task verifies the prior commits.
