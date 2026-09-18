# Supplement — adjacent virtual-key failure attribution

This supplement preserves the original UAT257/259 review verdict and records root's subsequent baseline check.

## Baseline result

Using the official PostgreSQL runner, root loaded **both** changed AuthNZ modules from revision `2787043410` through a read-only module-selector overlay. The exact selected baseline modules were:

- `User_DB_Handling.py`: `22509619a9748fa0a8a00b91bdee5d7fae1186b9c8637d074a6d75577b3f86c3`
- `auth_principal_resolver.py`: `27c15307abbb6c7fca647926cef584e2983575ce90d6ca8f3d145e1a6d0e6cca`

The selector's module-load log confirms the overlay paths for both modules. The same adjacent virtual-key case failed before an authenticated request with the same `TransactionError` then `DatabaseError` from `APIKeyManager.create_virtual_key`: **1 failed, 4 warnings, 11.37s**. The runner log SHA-256 is `195d93716b76d95008b88ad8ce295ce356f530b6fd65118f8121f941ffb70e3c`.

## Causal attribution

Root's projection of the unchanged `api_keys_repo.py` SQL shows the read is accepted while both relevant `INSERT` statements are rejected as `ProfileUserWriteRejected` by the adjacent profile-user write guard. The error originates before authentication/dependency resolution. This separates the failure from the reviewed content-scope repair and assigns it to the independently tracked API-key SQL tokenization issue (Task 262 / Task 204).

## Artifact integrity

```text
8b6de3be9b6a78ab1da500a074221cd3ef11986fca93b2a8457a275452d2decf  baseline-inputs.json
1620eda0542ed6958654a65b3e57d184c37a613ce600c672cdbfcc2f590d12e5  module-loads.jsonl
0041660b1c821605077175be2ee721ec86e0e7823864f4ce95fea3d5041ba04a  adjacent-sql-guard-projection.json
```

No source, test, runtime, Git, or native artifact was modified for this supplement. The independent UAT257/259 focused validation remains 72 passed, zero skips.
