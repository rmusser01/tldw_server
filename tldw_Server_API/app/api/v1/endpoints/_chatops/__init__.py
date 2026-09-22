"""Transport-agnostic shell shared by the Discord and Slack ChatOps endpoints.

The two integrations were near-clones of each other. Measured by difflib after
normalising the provider and entity vocabulary:

    discord_oauth_admin.py vs slack_oauth_admin.py   370/407 = 90.9%
    discord_support.py     vs slack_support.py       533/677 = 78.6%
    discord.py             vs slack.py               368/600 = 61.3%

plus four test clone pairs, one of them identical.

The cost was not theoretical. The IDOR fix on ``GET /{discord|slack}/jobs/{job_id}``
had to be written four times with byte-identical comment text, and the policy
contract has already drifted into two vocabularies that now mean the same thing:
``team_quota_per_minute`` against ``workspace_quota_per_minute``, ``status_scope``
of ``{team, team_and_user}`` against ``{workspace, workspace_and_user}``.

Only two things are genuinely per-protocol and stay injected: the request signature
algorithm (Ed25519 for Discord, HMAC-SHA256 ``v0=`` for Slack) and the command
parser. Everything else -- the OAuth state machine, the installation record, policy
read/write, quota enforcement, receipt and dedupe wiring, metric labels and the
error envelope -- is one implementation here.

See Docs/ADR/050-chatops-shared-shell.md.
"""

from .oauth_admin import ChatOpsOAuthProvider

__all__ = ["ChatOpsOAuthProvider"]
