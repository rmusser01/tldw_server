# PostgreSQL World Book ownership upgrade

World Books in shared PostgreSQL storage now carry a `client_id` owner. Catalogue,
detail, entry, attachment, retrieval, chat-context and export-scope operations
are restricted to the current account;
book names are unique within that account. SQLite keeps its existing per-user
files and names.

## Existing PostgreSQL installations

The old World Book tables recorded no owner. The upgrade adds a nullable owner
column without deleting books, entries or attachments. Existing rows remain
unassigned and are hidden from application accounts until an administrator
establishes their ownership. Opening the application never claims these rows.

Before upgrading a populated shared database, back it up and record the owner
of each existing book using trusted installation records. In a maintenance
session, assign only verified book IDs to the corresponding account's string
user ID through the database administration interface. The maintenance update
must be conditional on `client_id IS NULL`; inspect its affected-row count.
Do not infer ownership from whichever user logs in first or from an attachment
alone: earlier versions could create cross-account associations.

After assignment, verify that the intended owner can read the book and its
entries, another account cannot read or change them, and all retained
attachments reference characters belonging to the same owner. Uncertain books
must remain unassigned until their owner is established. The upgrade does not
provide a public ownership-transfer endpoint.

## Verification

Maintained coverage is in `test_world_book_owner_isolation.py` and
`test_world_book_consumer_owner_isolation.py`. It exercises two
real owners with separate SQLite files or the official shared PostgreSQL
fixture, including hidden legacy rows, retained content, explicit maintenance
assignment, literal search, and caller-owned rollback. The full fresh-install
UAT continues separately after PR2967 merges.
