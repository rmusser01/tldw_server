# UAT174 reviewed repair

Trusted psycopg SQLSTATE23505 is reduced to a payload-free subclass, raised outside the catch. No driver context, query, parameters or constraint identity escape. Existing duplicate409 mapping is reused; non-unique500 and rollback controls retained. Independent real PostgreSQL run:29 passed,0 skips,7.46s. Native duplicate conflict/recovery remains pending.

Originals preserved. Manifest records original/retained hashes; text trailing whitespace normalized, PNG bytes unchanged. Known runtime credentials and JWT/PEM patterns scanned with zero matches.
