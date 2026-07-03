# Chat_Macros

`Chat_Macros` owns macro definition models, validation, and slash-style argument
parsing for chat macro commands.

Task 1 intentionally stops at the definition boundary:

- YAML macro definitions are loaded with `yaml.safe_load` and validated by
  Pydantic models.
- `/wrapup` is bundled as a built-in macro definition.
- Non-empty tool and skill permissions are rejected.
- Merge and post-result steps can only consume outputs produced by earlier
  steps.

Execution, storage, Jobs integration, API routing, and frontend rendering are
left to later implementation slices.
