# UAT239 observed catalogue scope

TASK13260.181. Frozen8f877. Native Character form sends two catalogue reads with include_disabled=true, both500. Safe application log identifies list_world_books at line757 using with self.db.get_connection(). The PostgreSQL BackendConnectionWrapper does not implement this context-manager protocol. This is a read-only root source trace, not a causal regression test or implemented repair.

Prior214 task explicitly accepted two bounded reads (get_character_world_books and entry counts), with legacy CRUD residuals unverified. The newly exercised list_world_books catalogue failure is tracked separately; original214 acceptance is not relabeled as a failure. Use the existing supported read lifecycle and verify filtering/order and caller transaction ownership on actual PostgreSQL before acceptance.

Inputs:
- tldw_Server_API/app/core/Character_Chat/world_book_manager.py SHA256 039e29e72ada4c2a160f070d6262272f3db3d893826b9a97c3618ef1908bf691
- backlog/tasks/task-13260.153 - Load-Character-world-books-with-backend-compatible-read-transactions.md SHA256 2ec33cfc115505ed4f883f045ff7c995fb11de91e44bafabe07ca0bfe0f434d9
- testbot-entry-result.txt SHA256 9d3b84b0887a47fedb29c969e045c9ba4effe530602cba78de3a574bb30ef873
- worldbook-error-excerpt.json SHA256 17015edbf7ff9c871c1cf72086b14e88cc588d09662e0741a650f2a8b12e5fdb
