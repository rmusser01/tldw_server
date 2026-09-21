# Automation, Admin, And Operations

Use these pages when you want recurring work, integrations, server operations, tool/MCP setup, moderation, or administrator controls.

## Automation And Integration Pages

| Page or feature | Surface/status | What it lets you do | Common uses |
| --- | --- | --- | --- |
| `/integrations` | Advanced self-hosted | Discover and configure integration surfaces. | External services, connector setup, feature availability. |
| `/scheduled-tasks`, `/scheduled-tasks/results` | Advanced self-hosted | Manage schedules and inspect scheduled run results. | Recurring jobs, automation status, repeated imports. |
| `/watchlists` | Experimental/labs | Monitor sources, recurring runs, and alert rules. | Watchlists, reports, monitored topics. |
| `/workflow-editor` | Advanced self-hosted | Edit workflow definitions. | Multi-step processing, reusable automations. |
| `/mcp-hub` | Advanced self-hosted | Configure MCP hub profiles, servers, and tools. | Tool access, external MCP servers, workspace trust. |
| `/acp-playground` | Advanced self-hosted | Test ACP sessions, tools, permissions, workspaces, and MCP server config. | Agent protocol experiments, ACP diagnostics. |
| `/model-playground` | Advanced self-hosted | Test and compare model behavior. | Model selection, provider checks, prompt experiments. |
| `/skills` | Advanced self-hosted | Inspect or manage skill-facing surfaces. | Skill discovery, capability review. |
| `/notifications` | WebUI | View notification inbox and alert state. | Background job notices, task alerts. |

## Safety And Review Pages

| Page or feature | Surface/status | What it lets you do | Common uses |
| --- | --- | --- | --- |
| `/moderation` | Advanced self-hosted | Review moderation queues or supervised safety state. | Human review, policy enforcement. |
| `/moderation/rules` | Advanced self-hosted | Manage content rule surfaces. | Safety policy setup. |
| `/moderation-playground` | Legacy or advanced safety route | Test moderation behavior. | Policy testing, safety experiments. |
| `/claims-review` | Advanced self-hosted | Review extracted claims and evidence. | Claim verification, evidence workflows. |

## Admin And Operator Pages

| Page or feature | Surface/status | What it lets you do | Common uses |
| --- | --- | --- | --- |
| `/admin` | Admin/operator | Open the admin operations overview. | Shared-server administration. |
| `/admin/server` | Admin/operator | Inspect server status and server-level controls. | Deployment diagnostics. |
| `/admin/api-keys` | Admin/operator | Manage API keys where enabled. | Key lifecycle, admin key review. |
| `/admin/billing` | Admin/operator or hosted | Inspect billing administration surfaces. | Billing support, plan review. |
| `/admin/data-ops` | Admin/operator | Run or inspect data operations. | Cleanup, migration, maintenance. |
| `/admin/integrations`, `/admin/sources` | Admin/operator | Administer integrations and sources. | Connector oversight, source governance. |
| `/admin/llamacpp`, `/admin/mlx` | Admin/operator | Manage local model runtime surfaces. | Local model readiness, runtime controls. |
| `/admin/maintenance` | Admin/operator | Use maintenance and repair controls. | Cleanup, operational recovery. |
| `/admin/monitoring` | Admin/operator | Inspect monitoring dashboards. | Metrics and health review. |
| `/admin/orgs`, `/admin/rbac`, `/admin/rate-limiting` | Admin/operator | Manage organizations, permissions, and rate limits. | Multi-user administration. |
| `/admin/usage` | Admin/operator | Inspect usage state. | Quota and usage review. |
| `/admin/watchlists-items`, `/admin/watchlists-runs` | Admin/operator | Inspect watchlist item and run administration. | Watchlist operations. |

## Recovering an interrupted scheduled executor

Scheduled question/agent Jobs hold a durable execution claim for their schedule
slot. A replaced or expired Jobs lease does not prove the previous executor has
stopped, so another delivery cannot steal that claim. Normal completion, failure,
and cancellation release the claim after execution stops. A failed terminal-state
write is retryable; a hard process exit may require operator reconciliation.

If worker logs report `needs execution reconciliation`, first stop all automation
workers for that deployment, disable their automatic restart, and verify that the
previous executor process has exited. Inspect the owner, run, and Jobs records and
retain their identifiers before changing anything. If the process may still be
running, leave the claim intact.

After confirming execution stopped, use the deployment's configured environment
and virtual environment to inspect and release the exact claim through the
database interface:

```python
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import ScheduledTasksDatabase

db = ScheduledTasksDatabase.for_user(user_id=OWNER_ID)
run = db.get_run(owner_id=OWNER_ID, run_id=RUN_ID)
claim = run.run_summary.get("execution_claim") if run else None
# Inspect this claim and verify it belongs to the stopped attempt first.
if claim:
    db.release_scheduled_task_run_claim(
        owner_id=OWNER_ID, run_id=RUN_ID, claim_id=claim["id"]
    )
```

Use the existing Jobs retry controls for the same job and schedule slot, then
restart workers. Keep the original run identity; starting a new slot would create
new work. This recovery does not promise exactly-once arbitrary external effects;
the current consumer continues to prohibit unapproved tool execution.

## Related Docs

- [Workflows examples](../WebUI_Extension/Workflows_Examples.md)
- [Getting started with ACP](../Integrations_Experiments/Getting_Started_with_ACP.md)
- [MCP Unified guide](https://github.com/rmusser01/tldw_server/blob/main/Docs/MCP/Unified/Developer_Guide.md)
- [Organization administration](../Server/Organization_Administration.md)
- [Organizations and sharing](../Server/Organizations_and_Sharing.md)
- [Usage module](../Server/Usage_Module.md)
- [Metrics cheatsheet](../../Monitoring/Metrics_Cheatsheet.md)
- [Reminder and notifications API](../../API-related/Reminder_Notifications_API.md)
- [Watchlists API](../../API-related/Watchlists_API.md)
