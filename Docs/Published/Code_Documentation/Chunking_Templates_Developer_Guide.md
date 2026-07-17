# Chunking Templates Developer Guide

## Overview

The Chunking Templates system provides a flexible, extensible framework for defining reusable document chunking strategies. This guide covers the architecture, implementation details, and how to extend the system.

See also: Guides/Chunking_Code_Guide.md for the core Chunker APIs and streaming/hierarchical behaviors used by templates.

## Architecture

### Component Overview

```
┌─────────────────────────────────────────┐
│           API Layer                      │
│  /api/v1/endpoints/chunking_templates.py │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│         Schema Layer                     │
│  /api/v1/schemas/chunking_templates_     │
│         schemas.py                       │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│      Template Processing Layer           │
│   /core/Chunking/templates.py            │
│   /core/Chunking/template_               │
│        initialization.py                 │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│        Database Layer                    │
│   /core/DB_Management/media_db/native_class.py │
│   ChunkingTemplates table                │
└─────────────────────────────────────────┘
```

### Key Components

#### 1. Database Schema (`ChunkingTemplates` table)

```sql
CREATE TABLE ChunkingTemplates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    template_json TEXT NOT NULL,
    is_builtin BOOLEAN DEFAULT 0 NOT NULL,
    tags TEXT,  -- JSON array
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    last_modified DATETIME,
    version INTEGER DEFAULT 1,
    client_id TEXT NOT NULL,
    user_id TEXT,
    deleted BOOLEAN DEFAULT 0,
    prev_version INTEGER,
    merge_parent_uuid TEXT
);
```

#### 2. Template Structure

```python
@dataclass
class ChunkingTemplate:
    name: str
    description: str = ""
    base_method: str = "words"
    stages: List[TemplateStage] = field(default_factory=list)
    default_options: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TemplateStage:
    name: str  # 'preprocess', 'chunk', 'postprocess'
    operations: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
```

#### 3. Template Processor

The `TemplateProcessor` class manages the execution pipeline:

```python
class TemplateProcessor:
    def process_template(self, text: str, template: ChunkingTemplate, **options) -> List[str]:
        # 1. Preprocessing stage
        # 2. Chunking stage
        # 3. Postprocessing stage
        return chunks
```

## Database Operations

### MediaDatabase Methods

The `MediaDatabase` class provides these template-specific methods:

```python
# Create a new template
def create_chunking_template(self, name: str, template_json: str, ...) -> Dict

# Get template by ID, name, or UUID
def get_chunking_template(self, template_id: int = None, ...) -> Optional[Dict]

# List templates with filtering
def list_chunking_templates(self, include_builtin: bool = True, ...) -> List[Dict]

# Update existing template
def update_chunking_template(self, template_id: int = None, ...) -> bool

# Delete template (soft or hard)
def delete_chunking_template(self, template_id: int = None, ...) -> bool

# Seed built-in templates
def seed_builtin_templates(self, templates: List[Dict]) -> int
```

### Migration

Database migration is handled via `/core/DB_Management/migrations/004_add_chunking_templates.json`:

```json
{
  "version": 4,
  "name": "add_chunking_templates",
  "description": "Add ChunkingTemplates table for storing reusable chunking configuration templates",
  "up_sql": "CREATE TABLE IF NOT EXISTS ChunkingTemplates ...",
  "down_sql": "DROP TABLE IF EXISTS ChunkingTemplates ..."
}
```

## Template Operations

### Creating Custom Operations

To add a new preprocessing or postprocessing operation:

1. Add the operation to `TemplateProcessor`:

```python
def _my_custom_operation(self, data: Union[str, List[str]], options: Dict[str, Any]) -> Union[str, List[str]]:
    """Custom operation description."""
    # Process data
    return processed_data
```

2. Register it in `_register_builtin_operations`:

```python
def _register_builtin_operations(self):
    # ... existing operations ...
    self.register_operation("my_custom_operation", self._my_custom_operation)
```

### Creating Custom Chunking Methods

Custom chunking methods should be added to the main `Chunker` class in `/core/Chunking/chunker.py`.

## Built-in Templates

### Template Files Location

Built-in templates are stored as JSON files in:
```
/app/core/Chunking/template_library/
├── academic_paper.json
├── book_chapters.json
├── chat_conversation.json
├── code_documentation.json
├── legal_document.json
└── transcript_dialogue.json
```

### Template JSON Structure

```json
{
  "name": "template_name",
  "description": "Template description",
  "tags": ["tag1", "tag2"],
  "preprocessing": [
    {
      "operation": "operation_name",
      "config": {
        // operation-specific config
      }
    }
  ],
  "chunking": {
    "method": "chunking_method",
    "config": {
      "max_size": 100,
      "overlap": 20
    }
  },
  "postprocessing": [
    {
      "operation": "operation_name",
      "config": {
        // operation-specific config
      }
    }
  ]
}
```

### Template Initialization

Templates are automatically loaded on application startup via the lifespan event in `main.py`:

```python
from tldw_Server_API.app.core.Chunking.template_initialization import ensure_templates_initialized

# In lifespan function:
if ensure_templates_initialized():
    logger.info("Chunking templates initialized successfully")
```

## API Integration

### Endpoint Implementation

The API endpoints are implemented in `/api/v1/endpoints/chunking_templates.py`:

```python
router = APIRouter(prefix="/chunking/templates", tags=["Chunking Templates"])

@router.get("", response_model=ChunkingTemplateListResponse)
async def list_templates(...)

@router.get("/{template_name}", response_model=ChunkingTemplateResponse)
async def get_template(...)

@router.post("", response_model=ChunkingTemplateResponse, status_code=201)
async def create_template(...)

@router.put("/{template_name}", response_model=ChunkingTemplateResponse)
async def update_template(...)

@router.delete("/{template_name}", status_code=204)
async def delete_template(...)

@router.post("/apply", response_model=ApplyTemplateResponse)
async def apply_template(...)

@router.post("/validate", response_model=TemplateValidationResponse)
async def validate_template(...)
```

#### Additional Endpoints

```python
# Rank templates by simple metadata-based classifier
@router.post("/match")
async def match_templates(media_type: Optional[str] = Query(None),
                          title: Optional[str] = Query(None),
                          url: Optional[str] = Query(None),
                          filename: Optional[str] = Query(None)) -> Dict:
    ...

# Learn hierarchical boundary rules from a seed document
class LearnTemplateRequest(BaseModel):
    name: str
    example_text: Optional[str] = None
    description: Optional[str] = None
    save: bool = False
    classifier: Optional[Dict[str, Any]] = None

@router.post("/learn")
async def learn_template(req: LearnTemplateRequest) -> Dict:
    ...
```

Notes:
- `/match` uses POST with query parameters for convenience (e.g., `?filename=...&title=...`).
- `/learn` returns a minimal template with `chunking.config.hierarchical_template.boundaries`; set `save=true` to persist.

### Schema Definitions

Pydantic schemas in `/api/v1/schemas/chunking_templates_schemas.py`:

```python
class ChunkingTemplateCreate(BaseModel):
    name: str
    description: Optional[str]
    tags: Optional[List[str]]
    template: TemplateConfig

class TemplateConfig(BaseModel):
    preprocessing: Optional[List[Dict[str, Any]]]
    chunking: Dict[str, Any]
    postprocessing: Optional[List[Dict[str, Any]]]

class ChunkingTemplateResponse(BaseModel):
    id: int
    uuid: str
    name: str
    # ... other fields
```

## Integration with Existing Chunking API

### Using Templates in Chunking Endpoint

The existing chunking endpoint (`/api/v1/chunking/chunk_text`) has been modified to support templates:

```python
# In chunking_schema.py
class ChunkingOptionsRequest(BaseModel):
    template_name: Optional[str] = Field(None, description="Name of template to use")
    # ... other fields

# In chunking.py endpoint
if request_data.options and request_data.options.template_name:
    # Load and apply template
    template_data = db.get_chunking_template(name=request_data.options.template_name)
    # ... process with template
```

### Using Proposition Chunking

The chunking endpoint also supports propositional chunking (`method='propositions'`) with multiple engines and LLM prompt profiles.

Example request body for `/api/v1/chunking/chunk`:

```json
{
  "text_content": "Alice founded Acme Corp in 2020 and Bob joined in 2021.",
  "options": {
    "method": "propositions",
    "max_size": 3,
    "overlap": 1,
    "proposition_engine": "auto",
    "proposition_aggressiveness": 2,
    "proposition_min_proposition_length": 15,
    "proposition_prompt_profile": "claimify"
  }
}
```

Notes:
- `proposition_engine`: `heuristic` (default), `spacy`, `llm`, or `auto` (tries `spacy` then falls back).
- `proposition_prompt_profile`: `generic`, `claimify`, `gemma_aps` (LLM engine only).
- System-wide defaults can be set in `tldw_Server_API/Config_Files/config.txt` under `[Chunking]`.

## Testing

### Test Structure

Tests are located in `/tests/Chunking/test_chunking_templates.py`:

```python
class TestDatabaseOperations:
    """Test database CRUD operations"""

class TestTemplateInitialization:
    """Test template loading and seeding"""

class TestAPIEndpoints:
    """Test REST API endpoints"""

class TestTemplateProcessing:
    """Test template processing logic"""

class TestIntegration:
    """Test integration scenarios"""
```

### Running Tests

```bash
# Run all template tests
python -m pytest tldw_Server_API/tests/Chunking/test_chunking_templates.py -v

# Run specific test class
python -m pytest tldw_Server_API/tests/Chunking/test_chunking_templates.py::TestDatabaseOperations -v

# Run with coverage
python -m pytest tldw_Server_API/tests/Chunking/test_chunking_templates.py --cov=tldw_Server_API.app.core.Chunking
```

### Test Results

Current test status:
- Database Operations: 8/8 passing ✅
- Template Initialization: 2/2 passing ✅
- API Validation: 1/1 passing ✅

## Extending the System

### Adding a New Built-in Template

1. Create JSON file in `/app/core/Chunking/template_library/`:

```json
{
  "name": "my_template",
  "description": "My custom template",
  "tags": ["custom"],
  "chunking": {
    "method": "words",
    "config": {
      "max_size": 100
    }
  }
}
```

2. The template will be automatically loaded on next startup.

### Adding a New Operation

1. Implement the operation in `TemplateProcessor`:

```python
def _my_operation(self, data, options):
    # Implementation
    return processed_data
```

2. Register in `_register_builtin_operations()`:

```python
self.register_operation("my_operation", self._my_operation)
```

3. Use in templates:

```json
{
  "preprocessing": [
    {
      "operation": "my_operation",
      "config": {
        // options
      }
    }
  ]
}
```

### Creating a Custom Chunking Strategy

1. Implement in `/core/Chunking/strategies/`:

```python
class MyCustomStrategy(ChunkingStrategy):
    def chunk(self, text: str, config: Dict) -> List[str]:
        # Implementation
        return chunks
```

2. Register in `Chunker` class:

```python
self.strategies["my_custom"] = MyCustomStrategy()
```

3. Use in templates:

```json
{
  "chunking": {
    "method": "my_custom",
    "config": {
      // custom config
    }
  }
}
```

## Performance Considerations

### Caching

Templates are cached in memory after first load:

```python
class TemplateManager:
    def __init__(self):
        self._cache: Dict[str, ChunkingTemplate] = {}
```

### Database Indexes

Key indexes for performance:
- `idx_chunking_templates_name` - Fast lookup by name
- `idx_chunking_templates_uuid` - Fast lookup by UUID
- `idx_chunking_templates_deleted` - Filter active templates

### Batch Processing

For processing multiple documents with the same template:

```python
# Load template once
template = db.get_chunking_template(name="academic_paper")
processor = TemplateProcessor()

# Process multiple documents
for doc in documents:
    chunks = processor.process_template(doc.text, template)
```

## Troubleshooting

### Common Issues

1. **Template not found**
   - Check template name spelling
   - Verify template exists: `db.list_chunking_templates()`
   - Check if template is soft-deleted

2. **Invalid template JSON**
   - Validate JSON syntax
   - Use validation endpoint: `POST /api/v1/chunking/templates/validate`
   - Check required fields (chunking.method)

3. **Cannot modify built-in template**
   - Built-in templates have `is_builtin=1`
   - Create a copy with different name instead

4. **Database migration issues**
   - Check migration status in schema_version table
   - Run migrations manually if needed
   - Check logs for migration errors

### Debug Logging

Enable debug logging for detailed information:

```python
import logging
logging.getLogger("tldw_Server_API.app.core.Chunking").setLevel(logging.DEBUG)
```

## Security Considerations

1. **Input Validation**
   - All template JSON is validated before storage
   - Pydantic schemas enforce type checking
   - Template names are sanitized

2. **Access Control**
   - Templates track user_id for ownership
   - Built-in templates protected from modification
   - Implement auth middleware for production

3. **SQL Injection Prevention**
   - All database queries use parameterized statements
   - No string concatenation in SQL queries

## Future Enhancements

Planned improvements for the chunking templates system:

1. **Template Inheritance**: Allow templates to extend other templates
2. **Template Versioning**: Track and rollback template changes
3. **Template Marketplace**: Share templates between users
4. **Auto-detection**: Automatically select template based on content
5. **Performance Metrics**: Track template performance and usage
6. **Web UI**: Visual template editor and management interface

## API Reference

For complete API documentation, see:
- [Chunking Templates API Documentation](../API-related/Chunking_Templates_API_Documentation.md)

## Support

For issues or questions:
1. Check this developer guide
2. Review test cases for examples
3. Check logs for error details
4. Submit GitHub issue with details

---

*Last Updated: January 2025*
*Version: 1.0.0*
