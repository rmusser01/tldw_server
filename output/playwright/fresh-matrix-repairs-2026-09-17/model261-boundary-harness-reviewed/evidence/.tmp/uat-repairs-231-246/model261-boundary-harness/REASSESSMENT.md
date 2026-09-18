# UAT261 harness startup reassessment

No provider adapter, `complete-v2` route, or patched
`perform_chat_api_call` seam was reached.

| Attempt | Result | Safe classification |
| --- | --- | --- |
| 1 | stopped during app startup | missing single-user test key in the standalone shell |
| 2 | stopped during app startup | `ValueError` at AuthNZ `settings.py:923` |
| 3 | stopped during app startup after disposable AuthNZ/user-db paths and synthetic test keys | same `ValueError` at AuthNZ `settings.py:923` |

The harness never prints or retains startup logs, response bodies, prompts,
provider reasoning, credentials, or configuration values. The only retained
failure classification is its stage, exception class, and source file/line.

The source tree's existing pytest configuration establishes a broader isolated
AuthNZ process environment. A fixture-based runner, using that existing setup
without changing production configuration, is the next bounded option. No
fourth run occurs until that approach is authorized after this reassessment.
