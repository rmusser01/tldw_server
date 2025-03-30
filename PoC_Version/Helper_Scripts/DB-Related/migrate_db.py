import sqlite3
import logging
from datetime import datetime
import os
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import sys
from tqdm import tqdm


class DatabaseMigrator:
    def __init__(self, source_db_path: str, target_db_path: str, conversations_export_path: str):
        self.source_db_path = Path(source_db_path)
        self.target_db_path = Path(target_db_path)
        self.conversations_export_path = Path(conversations_export_path)
        self.source_conn = None
        self.target_conn = None

        # Tables to migrate (in order of dependencies)
        self.tables_to_migrate = [
            'Media',
            'Keywords',
            'MediaKeywords',
            'MediaVersion',
            'MediaModifications',
            'Transcripts',
            'MediaChunks',
            'UnvectorizedMediaChunks',
            'DocumentVersions'
        ]

        # Tables to explicitly ignore
        self.tables_to_ignore = {
            'media_fts',
            'media_fts_data',
            'media_fts_idx',
            'keyword_fts',
            'ChatConversations',
            'ChatMessages'
        }

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('migration.log'),
                logging.StreamHandler()
            ]
        )

    def connect_databases(self):
        """Establish connections to both databases"""
        try:
            self.source_conn = sqlite3.connect(self.source_db_path)
            self.target_conn = sqlite3.connect(self.target_db_path)
            # Enable foreign keys
            self.source_conn.execute("PRAGMA foreign_keys = ON")
            self.target_conn.execute("PRAGMA foreign_keys = ON")
        except Exception as e:
            logging.error(f"Failed to connect to databases: {str(e)}")
            raise

    def export_conversations(self):
        """Export all conversations to markdown files"""
        try:
            # Create export directory if it doesn't exist
            self.conversations_export_path.mkdir(parents=True, exist_ok=True)

            # Get all conversations
            conversations_query = """
                SELECT id, media_id, media_name, conversation_name, created_at
                FROM ChatConversations
            """
            messages_query = """
                SELECT sender, message, timestamp
                FROM ChatMessages
                WHERE conversation_id = ?
                ORDER BY timestamp
            """

            conversations = self.source_conn.execute(conversations_query).fetchall()

            # Add progress bar for conversation export
            print("Exporting chats:")
            for conv in tqdm(conversations):
                conv_id, media_id, media_name, conv_name, created_at = conv

                # Create filename from conversation details
                filename = f"{created_at}_{media_name or 'no_media'}_{conv_name or f'conversation_{conv_id}'}.md"
                filename = "".join(c if c.isalnum() or c in ".-_" else "_" for c in filename)

                messages = self.source_conn.execute(messages_query, (conv_id,)).fetchall()

                # Write conversation to markdown file
                with open(self.conversations_export_path / filename, 'w', encoding='utf-8') as f:
                    f.write(f"# Conversation: {conv_name or 'Untitled'}\n")
                    f.write(f"Media: {media_name or 'None'}\n")
                    f.write(f"Created: {created_at}\n\n")
                    f.write("---\n\n")

                    for sender, message, timestamp in messages:
                        f.write(f"**{sender}** ({timestamp}):\n")
                        f.write(f"{message}\n\n")

        except Exception as e:
            logging.error(f"Failed to export conversations: {str(e)}")
            raise

    def migrate_table(self, table_name: str):
        """Migrate a single table's data"""
        try:
            # Skip if table is in ignore list
            if table_name in self.tables_to_ignore:
                return

            # Get data
            data = self.source_conn.execute(f"SELECT * FROM {table_name}").fetchall()
            if not data:
                return

            # Get column names
            columns = [desc[0] for desc in self.source_conn.execute(f"SELECT * FROM {table_name} LIMIT 0").description]

            # Begin transaction in target database
            with self.target_conn:
                # Insert data with progress bar
                print(f"Migrating {table_name}:")
                placeholders = ','.join(['?' for _ in columns])
                insert_sql = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"

                for row in tqdm(data):
                    try:
                        self.target_conn.execute(insert_sql, row)
                    except Exception as e:
                        logging.error(f"Error inserting row in {table_name}: {str(e)}")
                        raise

            self.target_conn.commit()

        except Exception as e:
            logging.error(f"Failed to migrate table {table_name}: {str(e)}")
            raise

    def perform_migration(self):
        """Execute the complete migration process"""
        try:
            self.setup_logging()
            logging.info("Starting database migration")

            self.connect_databases()

            # Export conversations first
            self.export_conversations()

            # Migrate each table in order
            for table in self.tables_to_migrate:
                self.migrate_table(table)

            logging.info("Migration completed successfully")

        except KeyboardInterrupt:
            logging.error("Migration interrupted by user")
            raise
        except Exception as e:
            logging.error(f"Migration failed: {str(e)}")
            raise
        finally:
            if self.source_conn:
                self.source_conn.close()
            if self.target_conn:
                self.target_conn.close()


def validate_paths(source_db: str, target_db: str, export_path: str) -> None:
    """Validate the provided paths"""
    # Check source database exists
    if not os.path.isfile(source_db):
        raise ValueError(f"Source database does not exist: {source_db}")

    # Check target database path is writable
    target_dir = os.path.dirname(target_db) or '.'
    if not os.access(target_dir, os.W_OK):
        raise ValueError(f"Cannot write to target database location: {target_dir}")

    # Check export path is writable
    export_dir = Path(export_path)
    try:
        export_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ValueError(f"Cannot create export directory: {export_path}") from e


def parse_arguments():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(description='Migrate SQLite database and export conversations to markdown.')

    parser.add_argument(
        '--source-db',
        required=True,
        help='Path to the source database file'
    )

    parser.add_argument(
        '--target-db',
        required=True,
        help='Path where the new database will be created'
    )

    parser.add_argument(
        '--export-path',
        required=True,
        help='Directory where conversations will be exported as markdown'
    )

    args = parser.parse_args()

    try:
        validate_paths(args.source_db, args.target_db, args.export_path)
    except ValueError as e:
        parser.error(str(e))

    return args


def main():
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Create and run migrator
        migrator = DatabaseMigrator(args.source_db, args.target_db, args.export_path)
        migrator.perform_migration()

    except KeyboardInterrupt:
        print("\nMigration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Migration failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()