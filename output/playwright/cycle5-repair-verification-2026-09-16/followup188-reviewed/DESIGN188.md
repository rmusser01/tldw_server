# UAT188 / TASK13260.126

Bounded metric-localization repair. Native analytics reports study_streak_days=1 and the actual Study dashboard renders1 days. The component passes the numeric value correctly; both the existing English studyStreakDays resource and its component default contain an unconditional plural days.

Proposed production scope: replace those two messages with the existing ICU plural convention, {count, plural, one {# day} other {# days}}. Keep summary.study_streak_days, other metrics, deck ordering/selection and visibility unchanged. Do not alter the distinct remaining-card or Manage-count messages or generate extension locale artifacts.

Permanent causal test: ReviewAnalyticsSummary.plurals.test.tsx renders the actual component with real i18next, the existing ICU wrapper and production English resources. It checks0/1/6 days for resource-present and missing-key component fallback, plus same-instance0→1→6 updates and unchanged other metric values. No response/method/class mirror or mocked translation implementation.

The adjacent existing ReviewTab.analytics-summary test has an interpolation-only translation mock and an existing6 days assertion. If GREEN reaches that limitation, upgrade only its fixture to real ICU message formatting while preserving every assertion. Root owns native capture, tasks/tracker, shared docs and integration. Production remains on HOLD; only new tests/private evidence are authorized at this stage.

## Final status

Root released production after frozen native capture. Both strings and the causally required adjacent translation fixture are now frozen; final18/4 pass. See REPAIR188-REVIEW.md for exact RED/GREEN and limits.
