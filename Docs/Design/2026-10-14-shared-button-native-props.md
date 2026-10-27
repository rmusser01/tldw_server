# Shared native button props — TASK-13672

The existing shared Button forwards only a small subset of native props. Installed AntD Tooltip/Popover clones hover/focus/pointer handlers onto its child; dropping those handlers breaks trigger behavior. Standard aria and form attributes are dropped at the same boundary.

Reuse React.ButtonHTMLAttributes with only the legacy type field overridden. Spread remaining native props onto the actual button before managed styling/type/disabled attributes. Preserve native aria-label when the legacy ariaLabel alias is absent, and preserve caller aria-busy unless loading explicitly takes ownership. Existing variant/type/htmlType, icon, size and shape stay compatible and never leak to DOM. No new dependency or wrapper.

Scope: shared Button, actual installed Tooltip hover/focus controls, form/ref/accessibility forwarding and existing contrast/caller tests. This is distinct from the TASK-13669 empty-textarea pointer layout shift. User authorization covers remaining root issues. ADR assessment: native DOM compatibility within existing component; no durable architecture decision required.
