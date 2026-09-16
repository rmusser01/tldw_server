# UAT118 early independent boundary review

These are source-review findings sent during implementation, before final freeze; no final approval is implied.

- Literal user text: existing history placeholder replacement can change saved {{user}}/{{char}} before explicit Retry overlap while the request remains literal. Author reports permanent real endpoint RED and bounded correction in progress.
- Incomplete attachments: strict DB errors/position checks alone do not prevent the old history encoder from skipping missing secondary image bytes before matching. Author reports actual saved-second-image RED.
- Opt-in client contract: when has_image=true but the explicit images field is absent/empty, an opt-in read must not silently downgrade to a complete image-free message. Preserve default/legacy absent-field semantics outside the opt-in contract; author reports adapter RED.
- Mounted full boundary exposed PNG local mirror being relabeled JPEG in normalChatMode. Root approved preserving qualified data URLs with the raw legacy fallback unchanged, adding one production hunk. Exact equality stays strict.
- Snapshot/cap: ordinary PostgreSQL BackendManagedTransaction is READ COMMITTED. A separate byte preflight and blob load cannot establish one snapshot. Root approved a bounded single-statement strict CTE query in message_store, CASE-gated blob projection and actual decoded-length recheck before base64 expansion; no shared isolation-policy change. Independent final review must inspect this and genuine backend fixture coverage.

Local model native limitation: GET9099/props returned200 and modalities vision/video/audio=false. Actual attachment persistence and failure/Retry can be tested, but successful vision inference cannot be certified with this model. No provider/model reconfiguration or native mock is authorized by this limitation.
