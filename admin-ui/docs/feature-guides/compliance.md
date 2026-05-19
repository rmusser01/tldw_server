# Compliance Dashboard

The Compliance page provides a centralized view of your organization's security posture, MFA adoption, key hygiene, and scheduled compliance reporting.

---

## Posture Score

The posture score summarizes overall compliance health as a letter grade from **A** (best) to **F** (needs attention). It is computed from weighted factors:

| Factor                | Weight | Description                                           |
|-----------------------|--------|-------------------------------------------------------|
| MFA adoption          | 30%    | Percentage of active users with MFA enabled.          |
| Key rotation          | 25%    | Percentage of API keys rotated within policy window.  |
| Inactive user cleanup | 20%    | Percentage of stale accounts disabled or removed.     |
| Password policy       | 15%    | Strength of enforced password requirements.           |
| Session policy        | 10%    | Session timeout and token lifetime configuration.     |

### Grade thresholds

| Grade | Score range |
|-------|-------------|
| A     | 90-100      |
| B     | 80-89       |
| C     | 70-79       |
| D     | 60-69       |
| F     | 0-59        |

The score recalculates on page load and whenever underlying data changes. A trend arrow shows improvement or regression compared to the previous snapshot.

---

## MFA Adoption Metrics

The MFA card shows:

- **Enrolled users**: count and percentage of users with at least one MFA method configured.
- **Enforcement status**: whether MFA is required for all users, admin-only, or optional.
- **Method breakdown**: TOTP, WebAuthn/passkey, and recovery code usage.

### Improving MFA adoption

1. Navigate to **Settings > Authentication**.
2. Set MFA enforcement to **Required for all users**.
3. Communicate the change to your team and set a grace period if needed.
4. Monitor the compliance dashboard for adoption progress.

---

## Key Rotation Metrics

The Key Rotation card tracks API key hygiene:

- **Keys within policy**: number of keys rotated within the configured rotation window.
- **Overdue keys**: keys that have exceeded the rotation window, ordered by staleness.
- **Rotation window**: the configured maximum key age (default 90 days).
- **Average key age**: mean age across all active keys.

Click an overdue key to navigate to the key detail page and rotate it.

---

## Report Scheduling

Compliance reports can be generated on demand or scheduled for automatic delivery.

### Creating a scheduled report

1. Click **Schedule Report** on the Compliance page.
2. Configure:
   - **Frequency**: daily, weekly, or monthly.
   - **Format**: PDF or CSV.
   - **Recipients**: comma-separated email addresses.
3. Click **Save Schedule**.

### Report contents

Each report includes:

- Current posture score and grade.
- MFA adoption summary.
- Key rotation status and overdue key list.
- Inactive user summary.
- Changes since the previous report period.

### Managing schedules

- View active schedules in the **Scheduled Reports** section.
- Edit frequency, format, or recipients at any time.
- Disable or delete a schedule to stop future deliveries.

---

## Best Practices

- **Review weekly.** Check the compliance dashboard at least once a week to catch regressions early.
- **Enforce MFA.** Set MFA to required for all users to maximize the posture score.
- **Automate key rotation.** Use the API to rotate keys programmatically before they become overdue.
- **Distribute reports.** Schedule reports to security and team leads so compliance stays visible.
