# chunking_worker.py
# Worker for processing text chunking tasks

import hashlib
from typing import Any, Dict, List, Optional

from loguru import logger

from ..queue_schemas import (
    ChunkData,
    ChunkingMessage,
    EmbeddingMessage,
    JobStatus,
)
from .base_worker import BaseWorker, WorkerConfig


class ChunkingWorker(BaseWorker):
    """Worker that processes text chunking tasks"""
    
    def __init__(self, config: WorkerConfig):
        super().__init__(config)
        self.embedding_queue = config.queue_name.replace("chunking", "embedding")
        
    def _parse_message(self, data: Dict[str, Any]) -> ChunkingMessage:
        """Parse raw message data into ChunkingMessage"""
        return ChunkingMessage(**data)
    
    async def process_message(self, message: ChunkingMessage) -> Optional[EmbeddingMessage]:
        """Process chunking message and create chunks"""
        logger.info(f"Processing chunking job {message.job_id} for media {message.media_id}")
        
        try:
            # Update job status
            await self._update_job_status(message.job_id, JobStatus.CHUNKING)
            
            # Perform chunking
            chunks = self._chunk_text(
                message.content,
                message.chunking_config.chunk_size,
                message.chunking_config.overlap,
                message.chunking_config.separator
            )
            
            # Create chunk data objects
            chunk_data_list = []
            for i, (chunk_text, start_idx, end_idx) in enumerate(chunks):
                chunk_id = self._generate_chunk_id(message.job_id, i)
                
                chunk_data = ChunkData(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    metadata={
                        **message.source_metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "content_type": message.content_type
                    },
                    start_index=start_idx,
                    end_index=end_idx,
                    sequence_number=i
                )
                chunk_data_list.append(chunk_data)
            
            # Update job progress
            await self._update_job_progress(message.job_id, 25, len(chunks))
            
            # Create embedding message for next stage
            embedding_message = EmbeddingMessage(
                job_id=message.job_id,
                user_id=message.user_id,
                media_id=message.media_id,
                priority=message.priority,
                user_tier=message.user_tier,
                created_at=message.created_at,
                chunks=chunk_data_list,
                model_config={},  # Will be populated by embedding worker
                model_provider=""  # Will be populated by embedding worker
            )
            
            logger.info(f"Created {len(chunks)} chunks for job {message.job_id}")
            return embedding_message
            
        except Exception as e:
            logger.error(f"Error chunking content for job {message.job_id}: {e}")
            raise
    
    async def _send_to_next_stage(self, result: EmbeddingMessage):
        """Send chunked data to embedding queue"""
        await self.redis_client.xadd(
            self.embedding_queue,
            result.dict()
        )
        logger.debug(f"Sent job {result.job_id} to embedding queue")
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        separator: str
    ) -> List[tuple[str, int, int]]:
        """
        Split text into chunks with overlap.
        Returns list of (chunk_text, start_index, end_index) tuples.
        """
        if not text:
            return []
        
        chunks = []
        
        # Simple chunking by character count with overlap
        # In production, you might want more sophisticated chunking
        # that respects sentence/paragraph boundaries
        
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = min(start + chunk_size, text_length)
            
            # Try to find a good break point (separator)
            if end < text_length and separator:
                # Look for separator within last 20% of chunk
                search_start = max(start, end - int(chunk_size * 0.2))
                last_separator = text.rfind(separator, search_start, end)
                
                if last_separator != -1:
                    end = last_separator + len(separator)
            
            # Extract chunk
            chunk = text[start:end].strip()
            
            if chunk:  # Only add non-empty chunks
                chunks.append((chunk, start, end))
            
            # Move start position
            start = end - overlap if end < text_length else text_length
        
        return chunks
    
    def _generate_chunk_id(self, job_id: str, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        data = f"{job_id}:{chunk_index}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    async def _update_job_progress(self, job_id: str, percentage: float, total_chunks: int):
        """Update job progress information"""
        job_key = f"job:{job_id}"
        await self.redis_client.hset(
            job_key,
            mapping={
                "progress_percentage": percentage,
                "total_chunks": total_chunks
            }
        )