# API Notes


Existing Pages:
- Transcribe/Analyze/Ingestion
	1. Video Transcription + Summarization
	2. Audio File Transcription + Summarization
	3. Podcast
	4. Ebook Files
	5. Import Plain Text & .docx files
	6. Import XML Files
	7. Website Scraping
	8. PDF Ingestion
	9. Re-Summarize
	10. Live Recording & TRanscription
	11. Audio Generation Playground
	12. Arxiv Search & Ingest
	13. Semantic Scholar Search
- RAG Chat/Search
	1. RAG Search
	2. RAG QA Chat (Chat + notes)
	3. Notes Management
	4. Chat Management
- Chat with an LLM
	1. Remote LLM Chat (Horizontal)
	2. Remote LLM Chat (Vertical)
	3. One prompt, multiple APIs
	4. Four independent API Chats
	5. Chat workflows
- Web Search & Review
	1. Web Search & Review
- Character Chat
	1. Chat with a character card
	2. Character Chat Management
	3. Create a New Character Card
	4. Validate a Character Card
	5. Multi-Character Chat
- Writing Tools
	1. Writing Feedback
	2. Grammar & Style Checker
	3. Tone Analyzer & Editor
	4. Creative Writing Assistant
	5. Mikupad
- Search / View DB Items
	1. Media DB Search/ Detailed View
	2. Media DB Search/View Title + Summary
	3. View all MediaDB Items
	4. View Media Database Entries
	5. Search MediaDB by keyword
	6. View all RAG notes/Conversation items
	7. View RAG DB Entries
	8. View RAG Notes/Conversations by keyword
- Prompts
	1. View Prompt Database
	2. Search Prompts
	3. Add & Edit Prompts
	4. Clone & Edit prompts
	5. Add & Edito prompts
	6. Export Prompts
	7. Clone & Edit prompts
	8. Prompt Suggestion/Creation
	9. Export Prompts
- Manage DB Items
	1. Edit existing items in the Media DB
	2. Edit/Manage DB Items
	3. Clone and edit existing items in the media DB
- Embeddings Management
	1. Create Embeddings
	2. View/Update Embeddings
	3. Purge Embeddings
- Keywords
	1. View MediaDB Keywords
	2. Add MediaDB Keywords
	3. Delete MediaDB Keywords
	4. Export MEdiaDB Keywords
	5. Character Keywords
	6. RAG QA Keywords
	7. Meta-Keywords
	8. Prompt Keywords
- Import
	1. Import Markdown/Text Files
	2. Import Obsidian Vault
	3. Import a Prompt
	4. Import multiple prompts
	5. MediaWiki Import
	6. MediaWiki import config
	7. Import RAG chats
- Export
	1. Media DB Export
	2. RAG Conversations Export
	3. RAG Notes Export
	4. Export Prompts
	5. Export Prompts
- DB Mgmt
	1. Media DB
	2. RAG Chat DB
	3. Character Chat DB
- Utils
	1. MindMap Gen
	2. Youtube vid downloader
	3. Youtube audio downloader
	4. Youtube timestamp URL gen
- Anki
	1. Anki Deck Generator
	2. Anki Flash Card Validator
- Local LLM
	1. Local LLM with Llamafile
	2. Ollama Model Serving
- Trashcan
	1. Search and mark as trash
	2. View trash
	3. Delete DB Item
	4. Empty trash
- Evals
	1. G-Eval
	2. Infinite Bench
- Intro/Help
	1. Intro/Help Page
- Config Editor
	1. Config Editor



### Old
```
Here’s the important part. We’ll create:

    A global asyncio.Queue of “write tasks.”
    A WriteTask class that holds the SQL, parameters, and an asyncio.Future to signal completion.
    A background worker (writer_worker) that pops tasks from the queue, executes them, and sets the result in the Future.
    Endpoints that push a WriteTask onto the queue, then await the Future before returning.

# main.py
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Tuple, Union

from database import get_db_connection

app = FastAPI()

# -----------------------------
# 1) A global queue + task class
# -----------------------------
class WriteTask:
    """Holds SQL, parameters, and a Future to let the enqueuing code wait for completion."""
    def __init__(self, sql: str, params: tuple[Any, ...]):
        self.sql = sql
        self.params = params
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()

write_queue: asyncio.Queue[WriteTask] = asyncio.Queue()


# -----------------------------
# 2) The background worker
# -----------------------------
async def writer_worker():
    """Continuously processes write tasks from the queue, one at a time."""
    while True:
        task: WriteTask = await write_queue.get()
        try:
            # Perform the write
            with get_db_connection() as conn:
                conn.execute(task.sql, task.params)
                conn.commit()

            # If success, set the result of the Future
            task.future.set_result(True)
        except Exception as e:
            # If failure, set the exception so the caller can handle it
            task.future.set_exception(e)
        finally:
            write_queue.task_done()


# -----------------------------
# 3) Start the worker on startup
# -----------------------------
@app.on_event("startup")
async def startup_event():
    # Launch the writer worker as a background task
    asyncio.create_task(writer_worker())


# -----------------------------
# 4) Pydantic model for input
# -----------------------------
class ItemCreate(BaseModel):
    name: str


# -----------------------------
# 5) Write endpoint (POST)
# -----------------------------
@app.post("/items")
async def create_item(item: ItemCreate):
    """Queue a write to the database, then wait for its completion."""
    sql = "INSERT INTO items (name) VALUES (?)"
    params = (item.name,)

    # Create a WriteTask
    write_task = WriteTask(sql, params)

    # Put the task in the queue
    await write_queue.put(write_task)

    # Wait for the task to complete
    try:
        result = await write_task.future  # This will be True if successful
        return {"status": "success", "name": item.name}
    except Exception as exc:
        # If the DB write failed for some reason, raise a 500
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------
# 6) Read endpoint (GET)
# -----------------------------
@app.get("/items")
def read_items():
    """Simple read operation that does not need the queue."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM items")
        rows = cursor.fetchall()
        return [{"id": row[0], "name": row[1]} for row in rows]

Explanation

    WriteTask stores (sql, params, future). The future is how we pass success/failure back to the original request.
    When a request hits POST /items, we:
        Construct a WriteTask.
        put() it on the write_queue.
        Immediately await write_task.future. We don’t return until the DB operation is done.
    The writer_worker loop picks tasks in FIFO order and executes them one-by-one, guaranteeing no concurrency for writes (thus avoiding locks).
    On success, task.future.set_result(True) is called. On failure, task.future.set_exception(e).
    The awaiting endpoint sees either a success (and returns HTTP 200) or an exception (and returns HTTP 500).

    This pattern means each request is effectively serialized for writes, but the user still gets a definitive success/failure response in the same request/response cycle.
```










