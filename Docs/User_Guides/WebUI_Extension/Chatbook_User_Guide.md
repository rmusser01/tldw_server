# Chatbook User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Exporting Content](#exporting-content)
4. [Importing Content](#importing-content)
5. [Managing Jobs](#managing-jobs)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

## Introduction

### What is a Chatbook?

A Chatbook is your personal content archive - a portable file that contains your conversations, notes, characters, and other content from the tldw_server platform. Think of it as a comprehensive backup or a way to share curated collections of your work. The same import area can also migrate normal OpenWebUI "Export Chats" JSON files and uploaded OpenWebUI `webui.db` databases into tldw conversations.

### Why Use Chatbooks?

- **Backup & Recovery**: Protect your valuable content from data loss
- **Migration**: Move content between servers or accounts
- **Sharing**: Share curated collections with colleagues or friends
- **Organization**: Archive completed projects or research
- **Version Control**: Maintain snapshots of your content at specific points in time
- **Compliance**: Meet data retention or portability requirements

### What Can Be Included?

Chatbooks can contain:
- 💬 **Conversations**: Chat histories and message threads
- 📝 **Notes**: Personal notes and documentation
- 👤 **Characters**: AI character definitions and personalities
- 🌍 **World Books**: Lore and context for creative projects
- 📚 **Dictionaries**: Text replacement rules
- 📄 **Documents**: Generated summaries and reports
- 🎨 **Media**: Images, audio, and video files (optional)
- 🔢 **Embeddings**: Vector representations for search (optional)

OpenWebUI imports are handled as conversation imports: each valid OpenWebUI chat becomes a tldw conversation with message branches, timestamps, models, file references, folder metadata, and source metadata preserved where available. Database imports also mirror source folders into tldw folders under `OpenWebUI / <selected user>`.

## Getting Started

### Prerequisites

Before using Chatbooks, ensure you have:
1. An active account on the tldw_server platform
2. Authentication credentials (API token or login)
3. Sufficient quota for export/import operations (check your tier)

### Understanding File Format

Chatbooks use the `.chatbook` extension and are essentially ZIP archives containing:
- `manifest.json`: Metadata about the chatbook
- `content/`: Directory containing exported content
- `media/`: Directory containing media files (if included)
- `README.md`: Human-readable description

## Exporting Content

### Quick Export (Everything)

To export all your content:

1. Navigate to the Chatbooks section
2. Click "Create New Export"
3. Enter a name (e.g., "Complete Backup - January 2024")
4. Add a description for future reference
5. Leave content selections empty (exports everything)
6. Click "Export"

### Selective Export

For more control over what to export:

1. **Select Content Types**: Choose which types to include
   - Conversations
   - Notes
   - Characters
   - World Books
   - Dictionaries

2. **Select Specific Items**:
   - Click "Select Items" for each content type
   - Use checkboxes to choose individual items
   - Or leave empty to include all items of that type

3. **Configure Options**:
   - **Include Media**: Adds images, audio, video (increases file size)
   - **Media Quality**:
     - `thumbnail`: Smallest size, preview quality
     - `compressed`: Balanced size and quality (recommended)
     - `original`: Full quality, largest size
   - **Include Embeddings**: Adds vector data for search (technical users)
   - **Include Generated Content**: Adds AI-generated documents

4. **Add Metadata**:
   - **Author**: Your name or username
   - **Tags**: Keywords for organization (e.g., "research", "project-x")
   - **Categories**: Broader classifications (e.g., "work", "personal")

### Export Modes

#### Synchronous Export (Small Collections)
- Best for: Small exports (<100 items)
- Behavior: Waits for completion, returns file immediately
- Set: `async_mode: false`

#### Asynchronous Export (Large Collections)
- Best for: Large exports or when you don't want to wait
- Behavior: Returns job ID, processes in background
- Set: `async_mode: true`
- Monitor progress via job status endpoint

### Example Export Scenarios

#### Weekly Backup
```json
{
  "name": "Weekly Backup - Week 4",
  "description": "Regular weekly backup of all content",
  "content_selections": {},  // Empty = everything
  "tags": ["backup", "weekly", "automatic"]
}
```

#### Project Archive
```json
{
  "name": "Project Alpha - Final",
  "description": "Complete archive of Project Alpha conversations and notes",
  "content_selections": {
    "conversation": ["conv_alpha_1", "conv_alpha_2"],
    "note": ["note_alpha_summary", "note_alpha_report"]
  },
  "include_media": true,
  "tags": ["project-alpha", "completed", "2024"]
}
```

#### Research Collection
```json
{
  "name": "AI Research Papers Discussion",
  "description": "Conversations about recent AI papers",
  "content_selections": {
    "conversation": [],  // All conversations
    "note": []  // All notes
  },
  "include_generated_content": true,
  "categories": ["research", "ai"]
}
```

## Importing Content

### Basic Chatbook Import

1. Click "Import Chatbook"
2. Select "Chatbook archive" as the source
3. Select your `.chatbook` file
4. Choose conflict resolution strategy
5. Click "Import"

### OpenWebUI Chat Import

The same import tab can also import OpenWebUI chats from two sources:

- **OpenWebUI JSON**: the normal JSON file produced by OpenWebUI's "Export Chats" action.
- **OpenWebUI database**: an uploaded `webui.db` or `.sqlite` database copied from an OpenWebUI instance.

#### Importing OpenWebUI JSON

1. Click "Import Chatbook"
2. Select "OpenWebUI JSON" as the source
3. Select the `.json` file produced by OpenWebUI's chat export
4. Preview the file and review the chat, message, branch, duplicate, attachment-reference, malformed-chat, and warning counts
5. Choose `skip` to avoid reimporting existing chats, or `rename` to intentionally create a second copy
6. Click "Import"

OpenWebUI import creates one tldw conversation for each valid OpenWebUI chat. It preserves all valid message branches as parent-linked message trees, not only the currently selected branch. Duplicate detection uses the OpenWebUI source reference stored on imported conversations, so repeated imports default to skipping previously imported chats.

#### Importing an OpenWebUI Database

Use database import when you have local access to the OpenWebUI SQLite database and want to preserve source folder organization:

1. Click "Import Chatbook"
2. Select "OpenWebUI database" as the source
3. Select the copied `webui.db`, `.db`, or `.sqlite` file
4. Preview the database and review the detected users, chat counts, message counts, folder counts, and attachment-reference counts
5. Select exactly one selected OpenWebUI user to import
6. Confirm the destination namespace shown in preview, such as `OpenWebUI / <selected user> / source folders`
7. Choose `skip` to avoid reimporting existing chats, or `rename` to intentionally create a second copy
8. Click "Import"

Database import reads chats only for the selected source user. It does not import every user from the database by default. Folder paths are mirrored into tldw folders under `OpenWebUI / <selected user> / ...`; if the same folder name is already used elsewhere, tldw keeps the displayed path but safely disambiguates the backing collection name.

OpenWebUI import first preserves files, images, and artifacts as metadata references. The import step does not copy binaries from the OpenWebUI attachment store. Use the separate attachment hydration workflow below after the conversations are imported. The WebUI shows the OpenWebUI attachment hydration panel directly under the OpenWebUI import preview and results in the Chatbooks import tab. Use it when preview reports attachment references or when imported messages show missing OpenWebUI images/files.

#### Hydrating OpenWebUI Attachments

OpenWebUI attachment hydration is a follow-up step for imported conversations. V1 hydrates only referenced files preserved during OpenWebUI JSON or database import; it does not scan every file in an OpenWebUI instance and does not connect to a live OpenWebUI server.

Before hydrating, copy or mount the OpenWebUI data directory on the tldw_server host. The data root must contain:

- `webui.db`
- `uploads/`

The data root and all resolved files must be under configured allowed roots. Configure those roots with `Files.ingestion_source_allowed_roots` in `config.txt`, or with `INGESTION_SOURCE_ALLOWED_ROOTS` / `TLDW_INGESTION_SOURCE_ALLOWED_ROOTS`. Multiple roots can be separated with commas or the platform path separator. If no allowed roots are configured, or if the OpenWebUI data root resolves outside them, preview and jobs are rejected.

To hydrate from the WebUI:

1. Import the OpenWebUI JSON or database first.
2. In the Chatbooks import tab, use the OpenWebUI attachment hydration panel.
3. Enter the server-local OpenWebUI data root.
4. Enter the imported tldw conversation IDs to scan. For database imports, keep the selected OpenWebUI source user when shown.
5. Click "Preview attachments" and review referenced, resolved, missing, image, media, and warning counts.
6. Leave "Process supported files" off unless you want supported non-image files processed after registration.
7. Click "Run hydration job" only after the preview matches the intended root and scope.

Image files are copied into tldw-owned storage and message metadata is updated so imported image references can resolve from tldw. Non-image files are registered in the Media DB when available. Processing supported non-image files is opt-in and defaults to false; without that opt-in, hydration registers the file reference but does not run ingestion processing.

Hydration output storage uses `OPENWEBUI_HYDRATION_MEDIA_STORAGE_PATH` when set, otherwise `MEDIA_STORAGE_PATH`, otherwise the server's default media storage directory.

Hydration requires a single-user session or a user/admin allowed to operate on the imported conversations. Common warnings include missing files, unsupported file types, oversized files, path rejected, and path outside allowed roots.

OpenWebUI v1 import does not import admin exports, import embeddings, apply content selections, or import unreferenced OpenWebUI files.

### Conflict Resolution Strategies

When importing content that already exists:

#### Skip (Default)
- **Behavior**: Ignores items that already exist
- **Use When**: You want to avoid duplicates
- **Example**: Importing a backup when some content still exists

#### Overwrite
- **Behavior**: Replaces existing items with imported versions
- **Use When**: Imported version is more recent or authoritative
- **Example**: Restoring from a backup after data corruption
- **OpenWebUI**: Not available for OpenWebUI JSON or database imports in v1

#### Rename
- **Behavior**: Adds imported items with modified names
- **Use When**: You want to keep both versions
- **Example**: Importing content from another user
- **Result**: "My Note" becomes "My Note (Imported)"

#### Merge (Future Feature)
- **Behavior**: Intelligently combines content
- **Use When**: Both versions have valuable changes
- **Status**: Coming soon

### Import Options

- **Prefix Imported**: Adds prefix to all imported item names
  - Example: "Research Note" becomes "[Imported] Research Note"
  - Useful for identifying imported content

- **Import Media**: Not supported yet
  - Default: false
  - Keep this set to false; the server rejects true values in v1
  - OpenWebUI JSON and database imports preserve attachment references first; use OpenWebUI attachment hydration after import to copy referenced images and register supported files

- **Import Embeddings**: Not supported yet
  - Default: false
  - Keep this set to false; the server rejects true values in v1
  - OpenWebUI JSON and database imports do not import embeddings

### Preview Before Import

Always preview a chatbook or OpenWebUI export before importing:

1. Click "Preview Chatbook"
2. Select the file
3. Review:
   - Total items by type
   - OpenWebUI chat/message/branch counts when importing JSON
   - OpenWebUI database users, folder counts, and selected-user destination namespace when importing `webui.db`
   - OpenWebUI hydration referenced/resolved file counts before running a hydration job
   - Duplicate and malformed-chat counts
   - Creation date
   - Author information
   - Size
   - Warnings
4. Decide on import strategy

## Managing Jobs

### Monitoring Export Jobs

For async exports, monitor progress:

1. Navigate to "Export Jobs"
2. Find your job by ID or name
3. Check status:
   - `pending`: Waiting to start
   - `in_progress`: Currently processing
   - `completed`: Ready for download
   - `failed`: Error occurred
   - `cancelled`: Manually stopped

### Job Details Include

- **Progress Percentage**: How much is complete
- **Items Processed**: X of Y items done
- **Time Elapsed**: How long it's been running
- **Estimated Time Remaining**: When it might finish
- **Error Messages**: If something went wrong

### Downloading Completed Exports

1. Wait for status: `completed`
2. Click "Download" button
3. Save file to secure location
4. Verify file integrity

### Cancelling Jobs

To cancel a running job:
1. Find job in list
2. Click "Cancel" button
3. Confirm cancellation
4. Note: Partial exports are not saved

## Best Practices

### Backup Strategy

1. **Regular Backups**: Weekly or monthly full exports
2. **Project Archives**: Export completed projects
3. **Before Major Changes**: Export before bulk operations
4. **3-2-1 Rule**:
   - 3 copies of important data
   - 2 different storage media
   - 1 offsite backup

### Organization Tips

1. **Naming Convention**:
   - Include date: "Backup_2024-01-15"
   - Include purpose: "ProjectX_Final"
   - Include version: "Research_v2"

2. **Use Tags Effectively**:
   - Type tags: "backup", "archive", "share"
   - Time tags: "2024", "january", "week-4"
   - Project tags: "project-x", "research", "personal"

3. **Description Best Practices**:
   - Include what's included
   - Note why it was created
   - Mention any exclusions
   - Add relevant dates

### Performance Optimization

1. **Large Exports**:
   - Use async mode
   - Export during off-peak hours
   - Exclude media if not needed
   - Split into multiple smaller exports

2. **Selective Exports**:
   - Export only what you need
   - Use date filters (when available)
   - Exclude generated content if regeneratable

3. **Import Optimization**:
   - Preview first to understand contents
   - Use "skip" for faster imports
   - Import during low-activity periods

### Security Recommendations

1. **Storage**:
   - Encrypt sensitive chatbooks
   - Store in secure locations
   - Don't share publicly
   - Use cloud storage with encryption

2. **Sharing**:
   - Review contents before sharing
   - Remove sensitive information
   - Use secure transfer methods
   - Set expiration dates

3. **Retention**:
   - Delete old exports regularly
   - Keep only necessary backups
   - Use cleanup endpoint for expired files

## Troubleshooting

### Common Issues and Solutions

#### Export Takes Too Long
- **Cause**: Large amount of content
- **Solution**:
  - Use async mode
  - Exclude media files
  - Export in smaller chunks
  - Check server status

#### Import Fails
- **Cause**: File corruption, conflicts, quota exceeded
- **Solution**:
  - Verify file integrity
  - Check available quota
  - Try different conflict resolution
  - Import in smaller batches

#### Can't Download Export
- **Cause**: File expired, job failed, authentication issue
- **Solution**:
  - Check job status
  - Verify authentication
  - Re-export if expired
  - Check error messages

#### Duplicate Content After Import
- **Cause**: Wrong conflict resolution strategy
- **Solution**:
  - Use "skip" strategy
  - Enable prefix_imported option
  - Review before importing
  - Clean up duplicates manually

#### Missing Content in Export
- **Cause**: Incorrect selection, permissions, or filters
- **Solution**:
  - Verify content selections
  - Check permissions
  - Remove filters
  - Try exporting everything

### Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| QUOTA_EXCEEDED | Hit usage limits | Wait or upgrade tier |
| FILE_TOO_LARGE | Export/import too big | Reduce content or split |
| INVALID_FILE | Corrupted chatbook | Re-export or repair |
| AUTH_REQUIRED | Not logged in | Authenticate first |
| NOT_FOUND | Job or file missing | Check ID, may be expired |
| RATE_LIMITED | Too many requests | Wait before retrying |

### Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review error messages carefully
3. Check the [FAQ](#faq) below
4. Search existing issues on GitHub
5. Contact support with:
   - Error messages
   - Job IDs
   - Steps to reproduce
   - Chatbook size and content types

## FAQ

### General Questions

**Q: How large can a chatbook be?**
A: Default limit is 100MB, but this can vary by user tier. Premium users may have higher limits.

**Q: How long are exports kept?**
A: Exports are retained for 30 days by default, then automatically deleted. Download important exports promptly.

**Q: Can I automate exports?**
A: Yes, using the API. Scheduled exports through the UI are coming soon.

**Q: Are chatbooks encrypted?**
A: Not by default, but you can encrypt them after download. Built-in encryption is planned.

**Q: Can I share chatbooks with other users?**
A: You can share the file, but they need their own account to import. Direct sharing features are planned.

### Export Questions

**Q: What happens if I export everything?**
A: All your content is included. This may create a large file and take time to process.

**Q: Can I export only recent content?**
A: Date filtering is planned. Currently, you must select specific items.

**Q: Do exports include deleted items?**
A: No, only active content is exported. Soft-deleted items are excluded.

**Q: Can I resume a failed export?**
A: Not currently. You need to start a new export job.

### Import Questions

**Q: Will importing duplicate my content?**
A: Depends on conflict resolution. Use "skip" to avoid duplicates. OpenWebUI JSON and database imports default to `skip` and detect existing imported chats by `source=openwebui` plus the OpenWebUI external reference. Use `rename` only when you want a second copy.

**Q: Can I preview what will be imported?**
A: Yes, use the preview endpoint to see contents without importing.

**Q: What happens to media files during import?**
A: Chatbook media import is not supported yet; keep `import_media=false`. OpenWebUI JSON and database imports preserve file, image, and artifact references first. Run OpenWebUI attachment hydration after import when you have the server-local OpenWebUI data root and want to copy referenced images or register supported files.

**Q: Can I undo an import?**
A: No automatic undo. Keep backups before importing.

**Q: How do I import from another user?**
A: They export and share the file, you import with "rename" strategy.

### Technical Questions

**Q: What format is the chatbook file?**
A: ZIP archive with JSON metadata and content files.

**Q: Can I edit chatbook contents manually?**
A: Yes, but be careful. Invalid edits may prevent import.

**Q: What are embeddings?**
A: Vector representations used for semantic search. Most users don't need to export these.

**Q: Is the API REST or GraphQL?**
A: REST with JSON payloads.

**Q: What authentication is required?**
A: JWT bearer tokens in the Authorization header.

### Quota and Limits

**Q: How many exports can I create?**
A: Depends on your tier:
- Free: 5 per day
- Basic: 20 per day
- Premium: 100 per day
- Enterprise: Unlimited

**Q: Is there a rate limit?**
A: Yes:
- Export: 5 per minute
- Import: 5 per minute
- Download: 20 per minute

**Q: How many concurrent jobs can I run?**
A: Usually 1-3 depending on your tier.

### Future Features

**Q: When will merge conflict resolution be available?**
A: Planned for Q2 2024.

**Q: Will you add incremental backups?**
A: Yes, this is on the roadmap.

**Q: Can I schedule automatic exports?**
A: Coming soon, likely Q2 2024.

**Q: Will cloud storage integration be added?**
A: Yes, S3/GCS/Azure integration is planned.

## Appendix

### Glossary

- **Chatbook**: Portable archive of your content
- **Manifest**: Metadata file describing chatbook contents
- **Job**: Background task for processing exports/imports
- **Conflict Resolution**: Strategy for handling duplicate content
- **Quota**: Usage limits based on your account tier
- **Embeddings**: Vector representations for semantic search

### Related Documentation

- [API Documentation](../../API-related/Chatbook_API_Documentation.md)
- [Developer Guide](../../Code_Documentation/Chatbook_Developer_Guide.md)
- [Security Best Practices](../Server/Production_Hardening_Checklist.md)

### Support Resources

- GitHub Issues: [Report bugs or request features](https://github.com/tldw_server/issues)
- Discord Community: [Get help from the community](https://discord.gg/tldw-server)
- Email Support: support@tldw-server.com (Premium users)

---

*Last updated: January 2024*
*Version: 1.0.0*
