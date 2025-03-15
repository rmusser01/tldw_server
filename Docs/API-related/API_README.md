# API Documentation

## Overview

API uses FastAPI framework.
Designed to be simple and easy to use.
Generative endpoints follow openai API spec where possible.

- **URLs**
    - Main page: http://127.0.0.1:8000
    - API Documentation page: http://127.0.0.1:8000/docs




## Endpoints



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