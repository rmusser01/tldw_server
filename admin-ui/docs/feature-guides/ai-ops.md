# AI Operations

The AI Ops pages help administrators monitor AI resource consumption, enforce budgets, and troubleshoot agent behavior across the platform.

---

## AI Spend Tracking

The **AI Spend** card on the dashboard shows real-time and historical cost data for LLM, embedding, transcription, and TTS usage.

### Key metrics

| Metric                | Description                                                  |
|-----------------------|--------------------------------------------------------------|
| Total spend (period)  | Aggregated cost in USD for the selected time window.         |
| Spend by provider     | Breakdown by LLM provider (OpenAI, Anthropic, etc.).         |
| Spend by model        | Per-model cost breakdown within each provider.               |
| Spend by user / org   | Attribution of cost to individual users or organizations.    |
| Daily trend           | Line chart of daily spend over the past 30 days.             |

### Filtering

Use the controls at the top of the page to filter by:

- **Time range**: last 24h, 7d, 30d, or custom date range.
- **Provider**: restrict to a single provider.
- **Organization**: restrict to a single org (super-admin only).
- **User**: restrict to a single user.

---

## Agent Metrics and Session Monitoring

The **Agent Sessions** section tracks MCP and ACP agent activity.

### Session list

Each active or recent session shows:

| Column          | Description                                          |
|-----------------|------------------------------------------------------|
| Session ID      | Unique identifier for the agent session.             |
| Agent           | Name and version of the connected agent.             |
| User            | The user who initiated the session.                  |
| Started At      | Session start timestamp.                             |
| Duration        | Elapsed time (or total if completed).                |
| Tool Calls      | Number of tool invocations during the session.       |
| Tokens Used     | Total input + output tokens consumed.                |
| Status          | `active`, `completed`, `error`, or `timed_out`.      |

### Session detail

Click a session row to inspect:

- **Tool call log**: ordered list of tool invocations with input/output and latency.
- **Token breakdown**: input vs. output tokens per call.
- **Error trace**: stack trace and context if the session ended in error.

### Alerts

Configure alerts for:

- Sessions exceeding a token threshold.
- Sessions exceeding a duration threshold.
- Tool call error rate above a percentage.

---

## Token Budget Enforcement

Token budgets let administrators set hard or soft limits on LLM consumption.

### Budget types

| Type       | Behavior                                                        |
|------------|-----------------------------------------------------------------|
| Hard limit | Requests are rejected once the budget is exhausted.             |
| Soft limit | Requests continue but an alert is sent when the limit is hit.   |

### Creating a budget

1. Navigate to **Resource Governor > Token Budgets**.
2. Click **New Budget**.
3. Configure:
   - **Scope**: global, organization, user, or role.
   - **Period**: daily, weekly, or monthly.
   - **Token limit**: maximum tokens for the period.
   - **Limit type**: hard or soft.
4. Click **Save**.

### Budget status

The budget list shows:

- **Used / Limit**: current consumption against the cap.
- **Utilization %**: progress bar showing how close to the limit.
- **Resets at**: when the budget period resets.
- **Enforcement**: whether the budget is hard or soft.

### Overage handling

When a hard budget is exhausted:

- API requests return `429 Too Many Requests` with a `Retry-After` header.
- The admin UI shows a banner on the user/org detail page.
- An email notification is sent to configured recipients.

When a soft budget is exceeded:

- Requests continue normally.
- An alert is recorded in the incident log.
- The budget card shows a warning indicator.

---

## Best Practices

- **Start with soft limits.** Use soft budgets initially to understand usage patterns before enforcing hard limits.
- **Monitor daily.** Review the spend dashboard daily during rollouts to catch unexpected spikes.
- **Set per-org budgets.** In multi-tenant deployments, assign budgets per organization to prevent noisy-neighbor issues.
- **Alert early.** Configure alerts at 80% utilization so you have time to adjust before budgets are hit.
- **Review agent sessions.** Periodically audit agent session logs to identify inefficient tool usage or excessive token consumption.
