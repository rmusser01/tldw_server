# PostgreSQL sequence ownership repair

UAT147 now confines both Chat/Notes and Media sequence synchronization to their canonical tables. The first ChaCha tuple correction remains in commit41fcd5e4b3; this unit prevents Media from resetting or locking foreign shared-schema tables.

Author and independent runs each pass137tests across5files with zero skips; Ruff/Bandit have zero findings. The actual Media-only inventory covers all39 owned serial pairs, with foreign sequence preservation and held-table-lock controls.

Three consecutive normal startup cycles on preserved r3 single/multi profiles complete without the demonstrated deadlock. Each cycle returns authenticated Characters/Chats/Notes200 twice and healthy MCP Media database connection/write checks. These are targeted runtime checks, not full browser workflow or clean-machine installation certification. PostgreSQL official fixtures use a privileged role; database-level RLS enforcement is not established by these checks. Earlier failing traces remain under followup150.
