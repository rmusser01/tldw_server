# Import_RAG_Chat.py
# Functionality to import RAG Chats into the DB
#
# Imports
import os
import sqlite3
import sys
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List, Dict
#
# External Imports
from chardet import detect
import gradio as gr
#
# Local Imports
from PoC_Version.App_Function_Libraries.Utils.Utils import FileProcessor, ZipValidator, logging


#
########################################################################################################################
#
# RAG Chat Conversation Import Functionality

class RAGQABatchImporter:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.setup_logging()
        self.file_processor = FileProcessor()
        self.zip_validator = ZipValidator()

    def setup_logging(self):
        from loguru import logger
        logger.add('rag_qa_import.log', level='INFO', format='{time} - {level} - {message}')
        logger.add(sys.stdout, level='INFO', format='{time} - {level} - {message}')

    def process_markdown_content(self, content: str) -> List[Dict[str, str]]:
        """Process markdown content into a conversation format."""
        messages = []
        sections = content.split('\n\n')

        for section in sections:
            if section.strip():
                messages.append({
                    'role': 'user',
                    'content': section.strip()
                })

        return messages

    def process_keywords(self, db: sqlite3.Connection, conversation_id: str, keywords: str):
        """Process and link keywords to a conversation."""
        if not keywords:
            return

        keyword_list = [k.strip() for k in keywords.split(',')]
        for keyword in keyword_list:
            # Insert keyword if it doesn't exist
            db.execute("""
                INSERT OR IGNORE INTO rag_qa_keywords (keyword)
                VALUES (?)
            """, (keyword,))

            # Get keyword ID
            keyword_id = db.execute("""
                SELECT id FROM rag_qa_keywords WHERE keyword = ?
            """, (keyword,)).fetchone()[0]

            # Link keyword to conversation
            db.execute("""
                INSERT INTO rag_qa_conversation_keywords 
                (conversation_id, keyword_id)
                VALUES (?, ?)
            """, (conversation_id, keyword_id))

    def import_single_file(
            self,
            db: sqlite3.Connection,
            content: str,
            filename: str,
            keywords: str,
            custom_prompt: Optional[str] = None,
            rating: Optional[int] = None
    ) -> str:
        """Import a single file's content into the database"""
        conversation_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()

        # Process filename into title
        title = FileProcessor.process_filename_to_title(filename)
        if title.lower().endswith(('.md', '.txt')):
            title = title[:-3] if title.lower().endswith('.md') else title[:-4]

        # Insert conversation metadata
        db.execute("""
            INSERT INTO conversation_metadata 
            (conversation_id, created_at, last_updated, title, rating)
            VALUES (?, ?, ?, ?, ?)
        """, (conversation_id, current_time, current_time, title, rating))

        # Process content and insert messages
        messages = self.process_markdown_content(content)
        for msg in messages:
            db.execute("""
                INSERT INTO rag_qa_chats 
                (conversation_id, timestamp, role, content)
                VALUES (?, ?, ?, ?)
            """, (conversation_id, current_time, msg['role'], msg['content']))

        # Process keywords
        self.process_keywords(db, conversation_id, keywords)

        return conversation_id

    def extract_zip(self, zip_path: str) -> List[Tuple[str, str]]:
        """Extract and validate files from zip"""
        is_valid, error_msg, valid_files = self.zip_validator.validate_zip_file(zip_path)
        if not is_valid:
            raise ValueError(error_msg)

        files = []
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for filename in valid_files:
                with zip_ref.open(filename) as f:
                    content = f.read()
                    # Try to decode with detected encoding
                    try:
                        detected_encoding = detect(content)['encoding'] or 'utf-8'
                        content = content.decode(detected_encoding)
                    except UnicodeDecodeError:
                        content = content.decode('utf-8', errors='replace')

                    filename = os.path.basename(filename)
                    files.append((filename, content))
        return files

    def import_files(
            self,
            files: List[str],
            keywords: str = "",
            custom_prompt: Optional[str] = None,
            rating: Optional[int] = None,
            progress=gr.Progress()
    ) -> Tuple[bool, str]:
        """Import multiple files or zip files into the RAG QA database."""
        try:
            imported_files = []

            with sqlite3.connect(self.db_path) as db:
                # Process each file
                for file_path in progress.tqdm(files, desc="Processing files"):
                    filename = os.path.basename(file_path)

                    # Handle zip files
                    if filename.lower().endswith('.zip'):
                        zip_files = self.extract_zip(file_path)
                        for zip_filename, content in progress.tqdm(zip_files, desc=f"Processing files from {filename}"):
                            conv_id = self.import_single_file(
                                db=db,
                                content=content,
                                filename=zip_filename,
                                keywords=keywords,
                                custom_prompt=custom_prompt,
                                rating=rating
                            )
                            imported_files.append(zip_filename)

                    # Handle individual markdown/text files
                    elif filename.lower().endswith(('.md', '.txt')):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        conv_id = self.import_single_file(
                            db=db,
                            content=content,
                            filename=filename,
                            keywords=keywords,
                            custom_prompt=custom_prompt,
                            rating=rating
                        )
                        imported_files.append(filename)

                db.commit()

            return True, f"Successfully imported {len(imported_files)} files:\n" + "\n".join(imported_files)

        except Exception as e:
            logging.error(f"Import failed: {str(e)}")
            return False, f"Import failed: {str(e)}"


def create_conversation_import_tab() -> gr.Tab:
    """Create the import tab for the Gradio interface"""
    with gr.Tab("Import RAG Chats") as tab:
        gr.Markdown("# Import RAG Chats into the Database")
        gr.Markdown("""
        Import your RAG Chat markdown/text files individually or as a zip archive

        Supported file types:
        - Markdown (.md)
        - Text (.txt)
        - Zip archives containing .md or .txt files

        Maximum zip file size: 100MB
        Maximum files per zip: 100
        """)
        with gr.Row():
            with gr.Column():
                import_files = gr.File(
                    label="Upload Files",
                    file_types=["txt", "md", "zip"],
                    file_count="multiple"
                )

                keywords_input = gr.Textbox(
                    label="Keywords",
                    placeholder="Enter keywords to apply to all imported files (comma-separated)"
                )

                custom_prompt_input = gr.Textbox(
                    label="Custom Prompt",
                    placeholder="Enter a custom prompt for processing (optional)"
                )

                rating_input = gr.Slider(
                    minimum=1,
                    maximum=3,
                    step=1,
                    label="Rating (1-3)",
                    value=None
                )

            with gr.Column():
                import_button = gr.Button("Import Files")
                import_output = gr.Textbox(
                    label="Import Status",
                    lines=10
                )

        def handle_import(files, keywords, custom_prompt, rating):
            importer = RAGQABatchImporter("rag_qa.db")  # Update with your DB path
            success, message = importer.import_files(
                files=[f.name for f in files],
                keywords=keywords,
                custom_prompt=custom_prompt,
                rating=rating
            )
            return message

        import_button.click(
            fn=handle_import,
            inputs=[
                import_files,
                keywords_input,
                custom_prompt_input,
                rating_input
            ],
            outputs=import_output
        )

    return tab

#
# End of Conversation Import Functionality
#######################################################################################################################
