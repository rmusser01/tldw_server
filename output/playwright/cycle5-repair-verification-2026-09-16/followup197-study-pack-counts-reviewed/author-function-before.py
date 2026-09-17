def add_study_pack_cards(self, study_pack_id: int, flashcard_uuids: list[str]) -> int:
        """Attach flashcards to a study pack by UUID without relying on deck membership."""
        if not flashcard_uuids:
            return 0

        now = self._get_current_utc_timestamp_iso()
        if self.backend_type == BackendType.POSTGRESQL:
            insert_sql = (
                "INSERT INTO study_pack_cards("
                "study_pack_id, flashcard_uuid, created_at, last_modified, deleted, client_id, version"
                ") VALUES(?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT DO NOTHING"
            )
        else:
            insert_sql = (
                "INSERT OR IGNORE INTO study_pack_cards("
                "study_pack_id, flashcard_uuid, created_at, last_modified, deleted, client_id, version"
                ") VALUES(?, ?, ?, ?, ?, ?, ?)"
            )

        params = [
            (study_pack_id, card_uuid, now, now, False, self.client_id, 1)
            for card_uuid in flashcard_uuids
        ]
        try:
            with self.transaction() as conn:
                before_row = conn.execute(
                    "SELECT COUNT(*) FROM study_pack_cards WHERE study_pack_id = ? AND deleted = 0",
                    (study_pack_id,),
                ).fetchone()
                before_count = int(before_row[0]) if before_row else 0
                self.execute_many(insert_sql, params, commit=False)
                after_row = conn.execute(
                    "SELECT COUNT(*) FROM study_pack_cards WHERE study_pack_id = ? AND deleted = 0",
                    (study_pack_id,),
                ).fetchone()
                after_count = int(after_row[0]) if after_row else before_count
            return max(0, after_count - before_count)
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to add study pack cards: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to add study pack cards: {exc}") from exc
