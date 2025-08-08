# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context
This is a **legacy version** of the TLDW (Too Long Didn't Watch) project intended for stabilization and archiving. The goal is to make it stable and functional, then snapshot it - no new features or major refactors should be added.

TLDW is a comprehensive media ingestion, transcription, summarization, and RAG (Retrieval Augmented Generation) application that processes videos, audio, documents, and web content into a searchable local database.

## Essential Commands

### Running the Application
```bash
# Primary method - Enhanced launcher with better error handling
python app_fixed.py -gui

# Alternative - Original launcher
python summarize.py -gui

# Windows batch file
start_app.bat

# Linux/Mac (requires virtual environment)
source venv/bin/activate
python summarize.py -gui
```

### Testing
```bash
# Run all tests
pytest Tests/

# Run specific test categories
pytest Tests/SQLite_DB/
pytest Tests/Summarization/
pytest Tests/Character_Chat/
pytest Tests/Books/

# Test audio transcription
pytest Tests/Integration/test_audio_transcription.py

# Test database functionality
pytest Tests/SQLite_DB/test_sqlite_db.py
```

### Installation & Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data (auto-downloads on first run)
python -c "import nltk; nltk.download('punkt')"

# Create necessary directories
mkdir -p Databases Logs Config_Files

# For audio/video processing
# Linux: sudo apt install ffmpeg portaudio19-dev gcc build-essential python3-dev
# Windows: Ensure ffmpeg is in PATH
# MacOS: brew install ffmpeg portaudio
```

### Database Management
```bash
# Backup databases
cp Databases/*.db ./tldw_DB_Backups/

# Check database integrity
python Helper_Scripts/DB-Related/Inspect_DB.py

# Migrate old database format
python Helper_Scripts/DB-Related/migrate_db.py --source old.db --target new.db
```

## Architecture Overview

### Core Entry Points
- **`summarize.py`**: Main application entry, handles CLI and launches GUI
- **`app_fixed.py`**: Enhanced launcher with dependency checking and error handling
- **`App_Function_Libraries/Gradio_Related.py`**: GUI initialization and tab loading chain

### Directory Structure
```
App_Function_Libraries/       # Main application logic
├── Audio/                   # Audio transcription (faster_whisper)
├── Books/                   # Book/ePub ingestion
├── Character_Chat/          # Character card chat functionality
├── Chat/                    # LLM chat interfaces
├── DB/                      # Database managers
├── Gradio_UI/              # Individual UI tabs
├── Local_LLM/              # Local model inference
├── PDF/                    # PDF processing
├── RAG/                    # RAG implementation (ChromaDB + BM25)
├── Summarization/          # Content summarization
├── TTS/                    # Text-to-speech
├── Utils/                  # Utilities and system checks
└── Web_Scraping/           # Web content ingestion
```

### Key Processing Pipeline
1. **Input** → URL/File → `Video_DL_Ingestion_Lib.py` or `Local_File_Processing_Lib.py`
2. **Transcription** → `Audio_Transcription_Lib.py` (uses faster_whisper)
3. **Chunking** → `Chunk_Lib.py` (semantic/token/word-based)
4. **Summarization** → `Summarization_General_Lib.py` (LLM APIs)
5. **Storage** → `SQLite_DB.py` → SQLite databases
6. **RAG/Search** → `RAG_Library.py` + `ChromaDB_Library.py`

## Database Architecture

### Four SQLite Databases
1. **`media_summary.db`** (`Databases/media_summary.db`)
   - Main content database
   - Tables: Media, Transcripts, Summaries, MediaChunks, MediaKeywords
   - FTS5 full-text search enabled
   
2. **`prompts.db`** (`Databases/prompts.db`)
   - Prompt templates storage
   - Used for summarization and chat

3. **`RAG_QA_Chat.db`** (`Databases/RAG_QA_Chat.db`)
   - RAG-based Q&A conversations
   - Notes functionality
   - Conversation history

4. **`chatDB.db`** (`Databases/chatDB.db`)
   - Character card storage
   - Character chat history

### Database Manager Pattern
- `DB_Manager.py` provides unified interface
- All DB operations go through manager functions
- Supports SQLite with provisions for Elasticsearch (not implemented)

## Configuration

### Main Config File: `Config_Files/config.txt`
Key sections:
- `[Processing]`: CUDA/CPU selection
- `[Database]`: DB paths and backup locations
- `[Chunking]`: Default chunking methods and sizes
- `[API_KEYS]`: LLM API credentials
- `[Transcription]`: Whisper model selection

### Critical Settings for Stability
```ini
[Processing]
processing_choice = cpu  # Use 'cuda' only if properly configured

[Settings]
save_video_transcripts = True  # Keep all transcripts

[Database]
backup_path = ./tldw_DB_Backups/  # Regular backups essential
```

## Testing & Stabilization

### Pre-Archive Checklist
1. **Verify core functionality**:
   ```bash
   python summarize.py --test  # Basic functionality test
   pytest Tests/ -v            # Full test suite
   ```

2. **Check all ingestion types**:
   - YouTube video transcription
   - Local file processing
   - PDF ingestion
   - Book/ePub import
   - Web scraping

3. **Verify database operations**:
   - Search functionality (FTS5)
   - Keyword tagging
   - Content versioning
   - Backup/restore

4. **Test LLM integrations** (at least one):
   - Local: Ollama or llama.cpp
   - API: OpenAI, Anthropic, or Groq

### Known Issues & Solutions

1. **CUDA/cuDNN errors on Windows**:
   - Extract `cudnn_ops_infer64_8.dll` and `cudnn_cnn_infer64_8.dll` from Faster-Whisper-XXL package
   - Place in tldw root directory

2. **Missing ffmpeg**:
   - Required for audio/video processing
   - Must be in system PATH

3. **PyAudio installation fails**:
   - Linux: Install `portaudio19-dev`
   - Windows: Use `PyAudioWPatch` instead
   - Mac: `brew install portaudio`

4. **Gradio version conflicts**:
   - Locked to `gradio~=5.12.0` due to compatibility issues
   - Pydantic pinned to `2.10.6` for same reason

5. **Database migration needed**:
   - Use `migrate_db.py` for pre-Nov 1st databases
   - Always backup before migration

## Gradio UI Navigation

Main tabs flow (loaded from `Gradio_Related.py`):
1. **Introduction** → Overview and instructions
2. **Video Transcription** → Primary media ingestion
3. **Audio Processing** → Audio file transcription
4. **PDF/Book Import** → Document ingestion
5. **Website Scraping** → Web content extraction
6. **Chat Interfaces** → Multiple chat UI styles
7. **RAG Chat** → RAG-powered Q&A
8. **Search** → Database search interface
9. **Utilities** → Various tools and exports

## Critical Files to Never Modify

For archive stability, avoid modifying:
- `App_Function_Libraries/DB/SQLite_DB.py` - Core database schema
- `App_Function_Libraries/Gradio_Related.py` - UI initialization chain
- Database schema migrations
- `Config_Files/config.txt` structure

## Stabilization Commands

```bash
# Full system check
python -c "from App_Function_Libraries.Utils.System_Checks_Lib import *; cuda_check(); platform_check(); check_ffmpeg()"

# Verify imports
python -c "import gradio, torch, transformers, faster_whisper, chromadb"

# Test transcription
python summarize.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --whisper_model small

# Database integrity check
sqlite3 Databases/media_summary.db "PRAGMA integrity_check;"
```

## Notes for Archive Preparation

1. **Remove unnecessary files**:
   - Clear `Logs/` directory (keep structure)
   - Remove test outputs from `Tests/`
   - Clean `__pycache__` directories

2. **Document current state**:
   - Note working LLM endpoints in config
   - Document any workarounds applied
   - List tested functionality

3. **Create final backup**:
   ```bash
   tar -czf tldw_archive_$(date +%Y%m%d).tar.gz \
     --exclude='__pycache__' \
     --exclude='*.pyc' \
     --exclude='Logs/*.log' \
     .
   ```

Remember: This is a snapshot for archival. Focus on stability over features. Test thoroughly before final archive.