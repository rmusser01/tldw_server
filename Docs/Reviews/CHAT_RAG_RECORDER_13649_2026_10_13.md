# RAG metadata recorder — TASK-13649

Original recording PostgreSQL RAG merge failed with tuple index out of range. Real metadata updates now issue a one-parameter message-version increment; the old recorder routed it through a generic update path requiring multiple parameters. The exact translated statement now increments the recorded existing message once, returning zero rows for an absent message. Production is unchanged.

The regression retains lock-before-write ordering and exact interleaved pin/RAG merge assertions, adding exact message/history version increments. Focused RED1failure then GREEN1pass; full connected suite52passes0skips12warnings21.37s process0, including official PostgreSQL fixture cases. Independent bounded review clear. Ruff0; changed branch/regression Black clean, unrelated preexisting whole-file formatting retained; formatting AST matches tested AST. Scoped Bandit(excluding test assertions B101)0findings0errors; existing nosec parser warnings retained. Whitespace clean; normal exact hooks required.

ADR assessment:no newADR; test recorder qualification, existing production revision rules retained. PR:https://github.com/rmusser01/tldw_server/pull/3229. Remaining fixes, whole-PR review, final rebase and merge pending.
