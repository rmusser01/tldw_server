# Chatbook Developer Guide

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Structure](#module-structure)
3. [Database Schema](#database-schema)
4. [Core Components](#core-components)
5. [Job Queue System](#job-queue-system)
6. [File Handling](#file-handling)
7. [Security Implementation](#security-implementation)
8. [Testing](#testing)
9. [Extending the Module](#extending-the-module)
10. [Integration Guide](#integration-guide)

## Architecture Overview

### System Design

```mermaid
graph TB
    subgraph "API Layer"
        API[FastAPI Endpoints]
        Schema[Pydantic Schemas]
        Auth[Authentication]
    end

    subgraph "Service Layer"
        Service[ChatbookService]
        Validator[ChatbookValidator]
        Quota[QuotaManager]
    end

    subgraph "Core Components"
        Models[Chatbook Models]
        JobQueue[Job Queue]
        FileHandler[File Handler]
    end

    subgraph "Data Layer"
        DB[(SQLite DB)]
        FS[File System]
        Vector[(ChromaDB) - optional]
    end

    API --> Service
    Service --> Models
    Service --> JobQueue
    Service --> FileHandler
    Service --> DB
    FileHandler --> FS
    Service --> Vector
```

### Design Principles

1. **Separation of Concerns**: API, Service, and Data layers are clearly separated
2. **User Isolation**: All operations are scoped to authenticated users
3. **Async Support**: Long-running operations use background jobs
4. **Security First**: Input validation, path traversal protection, quota management
5. **Extensibility**: Easy to add new content types and export formats

## Module Structure

```
tldw_Server_API/app/core/Chatbooks/
├── __init__.py
├── chatbook_service.py       # Main service class
├── chatbook_models.py        # Data models and enums
├── chatbook_format_v1_1.py   # v1.1 feature registry, envelopes, inventory, preview/import validation
├── chatbook_validators.py    # Input validation
├── quota_manager.py          # User quota management
├── jobs_adapter.py          # Core Jobs adapter (queue/status integration)
└── exceptions.py             # Custom exceptions

tldw_Server_API/app/api/v1/
├── endpoints/
│   └── chatbooks.py         # API endpoints
└── schemas/
    └── chatbook_schemas.py  # Request/response schemas
```

### Key Files

- **chatbook_service.py**: Core business logic for export/import operations
- **chatbook_models.py**: Defines ChatbookManifest, ExportJob, ImportJob models
- **chatbook_format_v1_1.py**: Shared helpers for the v1.1 format contract
- **chatbook_validators.py**: Input validation and sanitization
- **quota_manager.py**: Manages user quotas and rate limiting
- **jobs_adapter.py**: Core Jobs integration for enqueueing and status mapping

## Database Schema

### Export Jobs Table (current schema)

```sql
CREATE TABLE IF NOT EXISTS export_jobs (
    job_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    chatbook_name TEXT NOT NULL,
    output_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    progress_percentage INTEGER DEFAULT 0,
    total_items INTEGER DEFAULT 0,
    processed_items INTEGER DEFAULT 0,
    file_size_bytes INTEGER,
    download_url TEXT,
    expires_at TIMESTAMP
);

CREATE INDEX idx_export_jobs_user_id ON export_jobs(user_id);
CREATE INDEX idx_export_jobs_status ON export_jobs(status);
CREATE INDEX idx_export_jobs_created_at ON export_jobs(created_at);
```

### Import Jobs Table (current schema)

```sql
CREATE TABLE IF NOT EXISTS import_jobs (
    job_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    chatbook_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    progress_percentage INTEGER DEFAULT 0,
    total_items INTEGER DEFAULT 0,
    processed_items INTEGER DEFAULT 0,
    successful_items INTEGER DEFAULT 0,
    failed_items INTEGER DEFAULT 0,
    skipped_items INTEGER DEFAULT 0,
    conflicts TEXT,  -- JSON array
    warnings TEXT    -- JSON array
);

CREATE INDEX idx_import_jobs_user_id ON import_jobs(user_id);
CREATE INDEX idx_import_jobs_status ON import_jobs(status);
CREATE INDEX idx_import_jobs_created_at ON import_jobs(created_at);
```

## Core Components

### ChatbookService

The main service class that handles all chatbook operations:

```python
class ChatbookService:
    def __init__(self, user_id: str, db: CharactersRAGDB):
        """Initialize service with user context and database."""
        self.user_id = user_id
        self.db = db

        # Global shim-backed queue; db is attached for handlers
        self.job_queue = get_job_queue()
        self.job_queue.db = db

        # Per-user secure directories under base data dir
        # (see Directory Setup details below)
        self.export_dir = ...
        self.import_dir = ...
        self.temp_dir = ...

        self._init_job_tables()
        self._register_job_handlers()
```

#### Key Methods

```python
async def create_chatbook(
    self,
    name: str,
    description: str,
    content_selections: Dict[ContentType, List[str]],
    **kwargs
) -> Tuple[bool, str, Optional[str]]:
    """Create a chatbook from selected content."""

async def import_chatbook(
    self,
    file_path: str,
    content_selections: Optional[Dict[ContentType, List[str]]],
    conflict_resolution: ConflictResolution,
    **kwargs
) -> Tuple[bool, str, Optional[str]]:
    """Import content from a chatbook file."""

def preview_chatbook(
    self,
    file_path: str
) -> Tuple[Optional[ChatbookManifest], Optional[str]]:
    """Preview chatbook contents without importing."""
```

### Chatbook v1.1 Format Helpers

`chatbook_format_v1_1.py` owns the shared v1.1 contract logic so producer,
preview, and importer behavior stays consistent:

- `FEATURE_REGISTRY` and `ensure_known_features()` define stable feature tokens
  and report unsupported features without crashing on malformed manifest input.
- `sha256_file()` and `build_file_inventory()` hash bundled files and omit
  `manifest.json` and archive checksum sidecars to avoid self-referential
  hashes.
- `build_content_envelope()` creates the content-envelope shape used by v1.1
  producers for structured payloads, rendered representations, integrity,
  lossiness, source references, and redaction metadata.
- `build_preview_report()` creates the deterministic preview report fields:
  `compatibility`, `features`, `integrity`, `lossiness`, `source_refs`,
  `warnings`, and `errors`.
- `validate_v1_1_before_import()` converts preview/inventory findings into
  pre-write import warnings or blocking errors according to the manifest
  compatibility policy.

Keep new v1.1 behaviors in this module when they are shared across content
types. Content-specific exporters and importers should call these helpers
instead of duplicating feature checks, hash formats, or report shapes.

### Chatbook v1.1 Flow

Export remains v1.0.0 by default. `CreateChatbookRequest.format_version` and
`ChatbookService.create_chatbook(..., format_version=...)` are coerced through
the shared version helper; v1.1 output is produced only when callers request
`"1.1.0"`.

For v1.1 export, `ChatbookService` builds the normal v1-compatible manifest and
then adds `features_used`, `producer`, `source_instance`, `compatibility`, and
`file_inventory`. Producer-specific content envelopes can be added
incrementally for content types that have stable structured restore payloads;
until then, content payloads keep the v1-compatible `file_path` layout.

Preview keeps `preview_chatbook()` as the legacy two-tuple. The API endpoint
prefers `preview_chatbook_with_report()`, which reuses the safe extraction flow
and adds the v1.1 preview report from `build_preview_report()`. This keeps
existing service doubles and v1.0 clients compatible while exposing v1.1
feature, integrity, lossiness, and source-reference summaries.

Import parses the manifest and runs `validate_v1_1_before_import()` immediately
for v1.1 archives, before building selections or calling content import methods.
Validation checks inventory hashes, missing listed files, unsafe inventory
paths, unsupported feature policy, and required inventory coverage for each
payload path that can be imported. Coverage includes explicit
`ContentItem.file_path` values where applicable, legacy fallback paths used by
current importers, and bundled conversation image attachment paths discovered
from verified conversation payloads.

### Chatbook Models

#### ChatbookManifest

```python
@dataclass
class ChatbookManifest:
    """Metadata for a chatbook archive."""
    version: ChatbookVersion
    name: str
    description: str
    author: Optional[str]
    user_id: str
    export_id: str
    created_at: datetime
    updated_at: datetime
    content_items: List[ContentItem]

    # Statistics
    total_conversations: int = 0
    total_notes: int = 0
    total_characters: int = 0
    total_media_items: int = 0

    # Options
    include_media: bool = False
    include_embeddings: bool = False
    media_quality: str = "compressed"
```

#### ExportJob

```python
@dataclass
class ExportJob:
    """Tracks export job status."""
    job_id: str
    user_id: str
    status: ExportStatus
    chatbook_name: str
    output_path: Optional[str]
    created_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    progress_percentage: int = 0
    total_items: int = 0
    processed_items: int = 0
    file_size_bytes: Optional[int]
    download_url: Optional[str]
    expires_at: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Content Collection

The service collects content based on selections:

```python
def _collect_conversations(
    self,
    selection: List[str],
    work_dir: Path,
    manifest: ChatbookManifest,
    content: ChatbookContent
):
    """Collect conversations for export."""
    if not selection:  # Empty means all
        conversations = self.db.get_all_conversations(self.user_id)
    else:
        conversations = [
            self.db.get_conversation(conv_id)
            for conv_id in selection
        ]

    for conv in conversations:
        # Process and add to content
        content_item = ContentItem(
            id=conv['id'],
            type=ContentType.CONVERSATION,
            title=conv.get('title', 'Untitled'),
            created_at=conv.get('created_at')
        )
        manifest.content_items.append(content_item)
        content.conversations.append(conv)
```

### Content Import

Import process with conflict resolution:

```python
def _import_conversations(
    self,
    extract_dir: Path,
    manifest: ChatbookManifest,
    selection: List[str],
    conflict_resolution: ConflictResolution,
    prefix_imported: bool,
    import_status: ImportJob
):
    """Import conversations from chatbook."""
    conv_dir = extract_dir / "content" / "conversations"

    for item in manifest.content_items:
        if item.type != ContentType.CONVERSATION:
            continue
        if selection and item.id not in selection:
            continue

        # Load conversation data
        file_path = conv_dir / f"conversation_{item.id}.json"
        with open(file_path, 'r') as f:
            conv_data = json.load(f)

        # Handle conflicts
        existing = self._get_conversation_by_name(conv_data['name'])
        if existing:
            if conflict_resolution == ConflictResolution.SKIP:
                import_status.skipped_items += 1
                continue
            elif conflict_resolution == ConflictResolution.OVERWRITE:
                self.db.update_conversation(existing['id'], conv_data)
            elif conflict_resolution == ConflictResolution.RENAME:
                conv_data['name'] = self._generate_unique_name(
                    conv_data['name'],
                    'conversation'
                )
                self.db.create_conversation(conv_data)
        else:
            self.db.create_conversation(conv_data)

        import_status.successful_items += 1
```

## Job Queue System

### Current Implementation (Shim)

A temporary synchronous implementation that will be replaced with a proper queue:

```python
class JobQueueShim:
    """Temporary job queue implementation."""

    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.handlers: Dict[str, Callable] = {}

    def submit_job(self, job: Job) -> str:
        """Submit a job for processing."""
        self.jobs[job.job_id] = job

        # Execute synchronously for now
        if job.job_type in self.handlers:
            handler = self.handlers[job.job_type]
            try:
                result = handler(job)
                job.status = JobStatus.COMPLETED
                job.result = result
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = str(e)

        return job.job_id
```

### Future Implementation

Will use Celery or similar for proper async processing:

```python
# Future implementation with Celery
@celery_app.task
def export_chatbook_task(job_id: str, params: dict):
    """Async task for chatbook export."""
    db = ...  # Resolve CharactersRAGDB for the user
    service = ChatbookService(params['user_id'], db)
    return service.process_export(job_id, params)
```

## File Handling

### Security Measures

1. **Path Traversal Protection**:
Use the production validator `ChatbookValidator.validate_zip_file(path)` which performs:
- ZIP magic check and integrity test
- Total and per-file size limits; zip-bomb compression ratio checks
- Path traversal, symlink, and dangerous extension checks
- Required files presence (e.g., `manifest.json`)

2. **Secure File Storage**:
```python
# Directory setup in ChatbookService.__init__
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

user_data_dir = DatabasePaths.get_user_chatbooks_dir(user_id)
exports_dir = DatabasePaths.get_user_chatbooks_exports_dir(user_id)
imports_dir = DatabasePaths.get_user_chatbooks_imports_dir(user_id)
temp_dir = DatabasePaths.get_user_chatbooks_temp_dir(user_id)
for directory in (user_data_dir, exports_dir, imports_dir, temp_dir):
    directory.chmod(0o700)
```

3. **Filename Sanitization**:
Supported extensions: `.zip`, `.chatbook`. Use `ChatbookValidator.validate_filename(name)` which:
- Enforces max length and allowed characters
- Accepts `.zip` or `.chatbook` (forces `.zip` on sanitize if missing)
- Returns `(is_valid, error_message, safe_filename)`

### Archive Creation

```python
def _create_chatbook_archive(
    self,
    work_dir: Path,
    output_path: Path
) -> bool:
    """Create ZIP archive from work directory."""
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add manifest
        manifest_path = work_dir / 'manifest.json'
        zf.write(manifest_path, 'manifest.json')

        # Add content recursively
        content_dir = work_dir / 'content'
        for root, dirs, files in os.walk(content_dir):
            for file in files:
                file_path = Path(root) / file
                arc_path = file_path.relative_to(work_dir)
                zf.write(file_path, arc_path)

        # Add README
        readme_path = work_dir / 'README.md'
        if readme_path.exists():
            zf.write(readme_path, 'README.md')

    return output_path.exists()
```

## Security Implementation

### Input Validation

All inputs are validated using Pydantic schemas and custom validators:

```python
class CreateChatbookRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=1, max_length=1000)
    content_selections: Dict[ContentType, List[str]]

    @validator('name')
    def validate_name(cls, v):
        if not re.match(r'^[\w\s.-]+$', v):
            raise ValueError('Invalid characters in name')
        return v

    @validator('content_selections')
    def validate_selections(cls, v):
        for content_type, ids in v.items():
            if len(ids) > 1000:  # Max items per type
                raise ValueError(f'Too many items for {content_type}')
        return v
```

### Quota Management

```python
class QuotaManager:
    """Manages user quotas and limits."""

    async def check_export_quota(self) -> Tuple[bool, Optional[str]]:
        """Check if user can create an export."""
        # Check daily limit
        today_exports = await self._count_today_exports()
        if today_exports >= self.tier_limits['exports_per_day']:
            return False, "Daily export limit reached"

        # Check storage quota
        used_storage = await self._calculate_storage_used()
        if used_storage >= self.tier_limits['storage_bytes']:
            return False, "Storage quota exceeded"

        return True, None

    async def check_file_size(self, size_bytes: int) -> Tuple[bool, Optional[str]]:
        """Check if file size is within limits."""
        max_size = self.tier_limits['max_file_size_bytes']
        if size_bytes > max_size:
            return False, f"File too large. Max: {max_size} bytes"

        return True, None
```

### Authentication & Authorization

```python
def get_chatbook_service(
    user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db)
) -> ChatbookService:
    """Get service instance for authenticated user."""
    # User is automatically injected and validated
    return ChatbookService(str(user.id), db)

# In endpoints
@router.post("/export")
async def create_chatbook(
    request: CreateChatbookRequest,
    service: ChatbookService = Depends(get_chatbook_service),
    user: User = Depends(get_request_user)
):
    # Service is already scoped to authenticated user
    # All operations will be isolated to this user
```

### Job IDs and Downloads
- Export/import job IDs are UUIDv4 (validated by `ChatbookValidator.validate_job_id`).
- Downloads are served by `GET /api/v1/chatbooks/download/{job_id}` once an export is `completed`.
- Prefer `GET /api/v1/chatbooks/export/jobs/{job_id}` to obtain the canonical `download_url`.

### Rate Limiting
- Export/Import endpoints: 5 requests per minute (per IP)
- Download endpoint: 20 requests per minute (per IP)
- Implemented via Resource Governor (RG) ingress policies; tests typically disable RG via `TEST_MODE`

## Testing

### Unit Tests

Test individual components:

```python
class TestChatbookService:
    """Unit tests for ChatbookService."""

    def test_init_creates_tables(self, service, mock_db):
        """Test that initialization creates required tables."""
        # Verify execute_query was called with CREATE TABLE
        calls = mock_db.execute_query.call_args_list
        create_export = any('CREATE TABLE' in str(call) and 'export_jobs' in str(call)
                          for call in calls)
        create_import = any('CREATE TABLE' in str(call) and 'import_jobs' in str(call)
                          for call in calls)
        assert create_export
        assert create_import

    @pytest.mark.asyncio
    async def test_export_chatbook_sync(self, service, mock_db):
        """Test synchronous export."""
        # Setup mock data
        mock_db.search_conversations_by_title.return_value = []
        mock_db.search_notes.return_value = []

        # Call export
        success, message, file_path = await service.create_chatbook(
            name="Test Export",
            description="Test",
            content_selections={},
            async_mode=False
        )

        assert success is True
        assert file_path is not None
        assert "chatbook" in file_path
```

### Integration Tests

Test with real database:

```python
class TestChatbookIntegration:
    """Integration tests with real database."""

    @pytest.fixture
    def test_db(self, tmp_path):
        """Create real test database."""
        db_path = tmp_path / "test.db"
        db = CharactersRAGDB(db_path=str(db_path))
        yield db
        # Cleanup
        if db_path.exists():
            db_path.unlink()

    @pytest.mark.asyncio
    async def test_export_import_roundtrip(self, service, tmp_path):
        """Test full export and import cycle."""
        # Export
        export_result = await service.create_chatbook(
            name="Roundtrip Test",
            description="Test",
            content_selections={
                ContentType.CONVERSATION: [],
                ContentType.NOTE: []
            },
            async_mode=False
        )

        success, message, export_path = export_result
        assert success is True

        # Import back
        import_result = await service.import_chatbook(
            file_path=export_path,
            conflict_resolution="rename"
        )

        assert import_result[0] is True
```

### Testing Best Practices

1. **Use Fixtures**: Share common test setup
2. **Mock External Dependencies**: Database, file system
3. **Test Edge Cases**: Empty exports, large files, conflicts
4. **Test Security**: Path traversal, file size limits
5. **Test Async Operations**: Job queue, progress tracking

## Extending the Module

### Adding New Content Types

1. **Update Models**:
```python
# In chatbook_models.py
class ContentType(str, Enum):
    # ... existing types ...
    CUSTOM_TYPE = "custom_type"
```

2. **Add Collection Method**:
```python
def _collect_custom_type(
    self,
    selection: List[str],
    work_dir: Path,
    manifest: ChatbookManifest,
    content: ChatbookContent
):
    """Collect custom type for export."""
    # Implementation
```

3. **Add Import Method**:
```python
def _import_custom_type(
    self,
    extract_dir: Path,
    manifest: ChatbookManifest,
    selection: List[str],
    conflict_resolution: ConflictResolution,
    prefix_imported: bool,
    import_status: ImportJob
):
    """Import custom type from chatbook."""
    # Implementation
```

4. **Update Service**:
```python
# In create_chatbook method
if ContentType.CUSTOM_TYPE in content_selections:
    self._collect_custom_type(
        content_selections[ContentType.CUSTOM_TYPE],
        work_dir, manifest, content
    )
```

5. **Add v1.1 Coverage When Applicable**:
- Exporters that produce v1.1 content should call `build_content_envelope()`
  and store the result in `content_items[].metadata.envelope`.
- Keep v1-compatible `file_path` and `checksum` populated when there is a
  bundled primary payload.
- Add rendered, binary, attachment, or external-reference representations to
  the envelope instead of inventing ad hoc metadata fields.
- Ensure every bundled file is covered by `file_inventory`; `build_file_inventory()`
  handles the final archive-wide inventory.
- Extend `validate_v1_1_before_import()` when a new importer reads fallback
  paths, attachment paths, or representation paths before writing.
- Add tests that prove missing inventory coverage, missing files, and checksum
  mismatches fail before import writes.

### Adding Export Formats

To support new formats (e.g., CSV, JSON Lines):

```python
class ExportFormat(str, Enum):
    CHATBOOK = "chatbook"  # Default ZIP
    CSV = "csv"
    JSONL = "jsonl"

class ExportFormatter:
    """Format exports for different outputs."""

    @staticmethod
    def format_csv(content: ChatbookContent) -> bytes:
        """Format content as CSV."""
        # Implementation

    @staticmethod
    def format_jsonl(content: ChatbookContent) -> bytes:
        """Format content as JSON Lines."""
        # Implementation
```

### Custom Conflict Resolution

Implement custom strategies:

```python
class SmartMergeResolver:
    """Intelligent content merging."""

    def resolve_conversation(
        self,
        existing: Dict,
        imported: Dict
    ) -> Dict:
        """Merge conversation intelligently."""
        # Combine messages
        existing_msgs = set(msg['id'] for msg in existing['messages'])
        for msg in imported['messages']:
            if msg['id'] not in existing_msgs:
                existing['messages'].append(msg)

        # Update metadata
        existing['updated_at'] = max(
            existing['updated_at'],
            imported['updated_at']
        )

        return existing
```

## Integration Guide

### Integrating with Other Modules

#### RAG Integration
Note: Example integration; ChatbookService does not directly depend on ChromaDB. Treat this as optional/future work.

```python
# Export embeddings with content
def _export_embeddings(self, content_ids: List[str], work_dir: Path):
    """Export ChromaDB embeddings."""
    embeddings_dir = work_dir / 'embeddings'
    embeddings_dir.mkdir(exist_ok=True)

    for content_id in content_ids:
        # Get embedding from ChromaDB
        embedding = self.chroma_client.get(
            collection_name="conversations",
            ids=[content_id]
        )

        # Save to file
        emb_file = embeddings_dir / f"{content_id}.npy"
        np.save(emb_file, embedding['embeddings'][0])
```

#### Media Processing

```python
# Compress media during export
def _process_media_for_export(
    self,
    media_path: Path,
    quality: str
) -> Path:
    """Process media file for export."""
    if quality == "thumbnail":
        return self._create_thumbnail(media_path)
    elif quality == "compressed":
        return self._compress_media(media_path)
    else:  # original
        return media_path
```

### API Integration

#### Webhook Notifications (Future)

```python
async def _notify_webhook(self, job: ExportJob):
    """Send webhook notification on job completion."""
    webhook_url = self.user_settings.get('webhook_url')
    if not webhook_url:
        return

    payload = {
        'event': 'export.completed',
        'job_id': job.job_id,
        'user_id': job.user_id,
        'timestamp': datetime.utcnow().isoformat(),
        'data': {
            'file_path': job.output_path,
            'total_items': job.total_items
        }
    }

    async with aiohttp.ClientSession() as session:
        await session.post(webhook_url, json=payload)
```

#### Client SDK Generation

Using OpenAPI spec for auto-generation:

```yaml
# chatbooks_openapi.yaml
openapi: 3.0.0
info:
  title: Chatbooks API
  version: 1.0.0
paths:
  /api/v1/chatbooks/export:
    post:
      summary: Export chatbook
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateChatbookRequest'
```

Generate client:
```bash
# Python client
openapi-generator generate -i chatbooks_openapi.yaml -g python -o ./sdk/python

# TypeScript client
openapi-generator generate -i chatbooks_openapi.yaml -g typescript-axios -o ./sdk/typescript
```

## Performance Optimization

### Database Optimization

1. **Batch Operations**:
```python
def _batch_insert_conversations(self, conversations: List[Dict]):
    """Insert multiple conversations efficiently."""
    # Cross-backend placeholder guidance:
    # - Prefer Postgres-style placeholders ($1,$2,...); project adapters normalize to SQLite automatically.
    # - If using raw sqlite3, switch placeholders to "?".
    self.db.execute_many(
        """INSERT INTO conversations (id, title, content, user_id)
           VALUES ($1, $2, $3, $4)""",
        [(c['id'], c['title'], c['content'], self.user_id)
         for c in conversations]
    )
```

2. **Indexing**:
```sql
-- Add composite index for common queries
CREATE INDEX idx_export_jobs_user_status
ON export_jobs(user_id, status, created_at DESC);
```

### File System Optimization

1. **Streaming Large Files**:
```python
async def _stream_to_archive(
    self,
    source_path: Path,
    zf: zipfile.ZipFile,
    arc_name: str
):
    """Stream large files to archive."""
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks

    with open(source_path, 'rb') as src:
        with zf.open(arc_name, 'w') as dst:
            while chunk := src.read(CHUNK_SIZE):
                dst.write(chunk)
                # Update progress
                self._update_progress(len(chunk))
```

2. **Parallel Processing**:
```python
async def _collect_content_parallel(
    self,
    content_selections: Dict[ContentType, List[str]]
):
    """Collect content in parallel."""
    tasks = []

    for content_type, selection in content_selections.items():
        if content_type == ContentType.CONVERSATION:
            tasks.append(self._collect_conversations_async(selection))
        elif content_type == ContentType.NOTE:
            tasks.append(self._collect_notes_async(selection))

    results = await asyncio.gather(*tasks)
    return self._merge_results(results)
```

### Memory Management

1. **Generator for Large Collections**:
```python
def _iter_conversations(self, selection: List[str]):
    """Iterate conversations without loading all into memory."""
    if not selection:
        # Stream all conversations
        offset = 0
        limit = 100
        while True:
            batch = self.db.get_conversations_batch(
                self.user_id, offset, limit
            )
            if not batch:
                break
            yield from batch
            offset += limit
    else:
        # Load specific conversations
        for conv_id in selection:
            yield self.db.get_conversation(conv_id)
```

## Monitoring & Debugging

### Logging

Comprehensive logging throughout:

```python
import structlog

logger = structlog.get_logger()

class ChatbookService:
    def __init__(self, user_id: str, db: CharactersRAGDB):
        self.logger = logger.bind(
            service="chatbook",
            user_id=user_id
        )

    async def create_chatbook(self, **kwargs):
        self.logger.info(
            "Creating chatbook",
            name=kwargs.get('name'),
            content_types=list(kwargs.get('content_selections', {}).keys())
        )

        try:
            # ... operation ...
            self.logger.info(
                "Chatbook created successfully",
                file_path=file_path,
                size_bytes=file_size
            )
        except Exception as e:
            self.logger.error(
                "Failed to create chatbook",
                error=str(e),
                exc_info=True
            )
            raise
```

### Metrics

Track key metrics:

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
export_counter = Counter(
    'chatbook_exports_total',
    'Total number of chatbook exports',
    ['user_tier', 'status']
)

export_duration = Histogram(
    'chatbook_export_duration_seconds',
    'Time spent exporting chatbooks',
    ['content_type']
)

active_jobs = Gauge(
    'chatbook_active_jobs',
    'Number of active export/import jobs',
    ['job_type']
)

# Use in service
@export_duration.time()
async def create_chatbook(self, **kwargs):
    active_jobs.labels(job_type='export').inc()
    try:
        # ... operation ...
        export_counter.labels(
            user_tier=self.user_tier,
            status='success'
        ).inc()
    except Exception as e:
        export_counter.labels(
            user_tier=self.user_tier,
            status='failure'
        ).inc()
        raise
    finally:
        active_jobs.labels(job_type='export').dec()
```

### Health Checks

```python
@router.get("/health")
async def health_check(
    service: ChatbookService = Depends(get_chatbook_service)
):
    """Check module health."""
    checks = {
        'database': await service._check_database(),
        'storage': await service._check_storage(),
        'job_queue': await service._check_job_queue()
    }

    status = 'healthy' if all(checks.values()) else 'unhealthy'

    return {
        'status': status,
        'checks': checks,
        'timestamp': datetime.utcnow().isoformat()
    }
```

## Deployment Considerations

### Environment Variables

```bash
# Required
USER_DB_BASE_DIR=/app/Databases/user_databases
DATABASE_URL=sqlite:////app/Databases/users.db

# Optional
CHATBOOK_MAX_FILE_SIZE=104857600  # 100MB
CHATBOOK_EXPORT_RETENTION_DAYS=30
CHATBOOK_TEMP_DIR=/tmp/tldw_chatbooks
CHATBOOK_ENABLE_COMPRESSION=true
CHATBOOK_COMPRESSION_LEVEL=6
```

`DATABASE_URL` points to the central auth/users database (SQLite or Postgres). `USER_DB_BASE_DIR` is the per-user
storage root defined in `tldw_Server_API.app.core.config`, used by `tldw_Server_API/app/core/config.py` and
`tldw_Server_API/app/core/DB_Management/db_path_utils.py` to locate user-specific SQLite DBs and storage
directories (for example, `Media_DB_v2.db`, `ChaChaNotes.db`, and chatbook exports/imports). The defaults can vary
by environment: `Dockerfiles/docker-compose.yml` sets `DATABASE_URL=sqlite:///./Databases/users.db`, while
`config.py` falls back to `Databases/user_databases/<SINGLE_USER_FIXED_ID>/tldw.db` if `DATABASE_URL` is unset.
Override `USER_DB_BASE_DIR` via environment variable or `Config_Files/config.txt` as needed.

### Docker Configuration

```yaml
# docker-compose.yml (app service)
volumes:
  - app-data:/app/Databases
  - chroma-data:/app/Databases/user_databases
```

Volume setup is environment-specific (dev vs prod). `Dockerfiles/Dockerfile.prod` copies `Databases/` into
`/app/Databases`, while `Dockerfiles/Dockerfile.worker` runs `mkdir -p /app/Databases`. See
`Dockerfiles/docker-compose.yml` and `Dockerfiles/docker-compose.embeddings.yml` for the current volume mappings.

### Backup Strategy

```python
# Scheduled backup of job tables
@scheduler.scheduled_job('cron', hour=2)
async def backup_job_tables():
    """Backup job tables daily."""
    backup_path = Path('/backups') / f"jobs_{datetime.now():%Y%m%d}.sql"

    await db.execute(f"""
        .output {backup_path}
        .dump export_jobs import_jobs
    """)
```

## Troubleshooting Guide

### Common Issues

1. **Jobs Stuck in Pending**:
   - Check job queue status
   - Verify worker processes running
   - Check for database locks

2. **Import Failures**:
   - Validate chatbook file integrity
   - Check available disk space
   - Review conflict resolution logs

3. **Performance Issues**:
   - Monitor database query times
   - Check disk I/O metrics
   - Review memory usage

### Debug Mode

Enable detailed logging:

```python
# In development
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
TLDW_LOG_LEVEL=DEBUG
```

## Future Roadmap

### Planned Features

1. **Proper Job Queue**: Replace shim with Celery/RQ
2. **Incremental Exports**: Only export changes
3. **Encryption**: Built-in archive encryption
4. **Cloud Storage**: S3/GCS/Azure integration
5. **Sharing**: Direct user-to-user sharing
6. **Templates**: Export configuration templates
7. **Webhooks**: Event notifications
8. **Compression Options**: Variable compression levels
9. **Merge Conflict Resolution**: Smart content merging
10. **API v2**: GraphQL support

### Contributing

To contribute to the Chatbook module:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Update documentation
6. Submit a pull request

### Code Style

Follow project conventions:
- PEP 8 for Python code
- Type hints for all functions
- Docstrings for classes and methods
- Comprehensive error handling
- Logging for debugging

---

*Last updated: January 2024*
*Version: 1.0.0*
