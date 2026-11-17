# Private saved-model guidance — TASK-13675

After a private popup binds to its native conversation, its model picker changes only local defaults. Protected replies use saved server authority, so displaying model B as the active choice is misleading. Keep the picker for prospective setup. After binding replace it with localized saved-model/provider guidance and retain Continue in full chat for review. Do not invent frozen values, add an override protocol or mutate saved settings.

Extend the existing mounted Manager Quick Chat journey: fresh picker, bind and guidance, changed default without picker returning, close/reopen fresh with picker. Keep actual native wire/owner/request/draft proofs separate from the declared mounted UI adapters. ADR assessment: existing ADR-049; no new durable architecture rule.
