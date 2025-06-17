# storage_worker.py
# Worker for storing embeddings in ChromaDB and updating SQL database

from typing import Any, Dict, Optional

from loguru import logger

from ...DB_Management.SQLite_DB import (
    update_media_table_vector_processing_status,
    update_media_chunks_table_vector_processing_status
)
from ..ChromaDB_Library import ChromaDBManager
from ..queue_schemas import (
    JobStatus,
    StorageMessage,
)
from .base_worker import BaseWorker, WorkerConfig


class StorageWorker(BaseWorker):
    """Worker that stores embeddings in ChromaDB and updates database"""
    
    def __init__(self, config: WorkerConfig):
        super().__init__(config)
        self.chroma_manager = ChromaDBManager()
    
    def _parse_message(self, data: Dict[str, Any]) -> StorageMessage:
        """Parse raw message data into StorageMessage"""
        return StorageMessage(**data)
    
    async def process_message(self, message: StorageMessage) -> None:
        """Store embeddings and update database"""
        logger.info(
            f"Processing storage job {message.job_id} with {len(message.embeddings)} embeddings"
        )
        
        try:
            # Update job status
            await self._update_job_status(message.job_id, JobStatus.STORING)
            
            # Get or create ChromaDB collection for user
            collection = await self._get_or_create_collection(
                message.user_id,
                message.collection_name
            )
            
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for embedding_data in message.embeddings:
                ids.append(embedding_data.chunk_id)
                embeddings.append(embedding_data.embedding)
                documents.append("")  # ChromaDB requires documents, even if empty
                metadatas.append({
                    **embedding_data.metadata,
                    "media_id": str(message.media_id),
                    "model_used": embedding_data.model_used,
                    "dimensions": str(embedding_data.dimensions)
                })
            
            # Store in ChromaDB in batches
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_end = min(i + batch_size, len(ids))
                
                await self._store_batch(
                    collection,
                    ids[i:batch_end],
                    embeddings[i:batch_end],
                    documents[i:batch_end],
                    metadatas[i:batch_end]
                )
                
                # Update progress
                progress = 75 + (25 * batch_end / len(ids))
                await self._update_job_progress(message.job_id, progress)
            
            # Update SQL database
            await self._update_database(message.media_id, message.total_chunks)
            
            # Mark job as completed
            await self._update_job_status(message.job_id, JobStatus.COMPLETED)
            
            logger.info(f"Successfully stored embeddings for job {message.job_id}")
            
        except Exception as e:
            logger.error(f"Error storing embeddings for job {message.job_id}: {e}")
            raise
    
    async def _send_to_next_stage(self, result: Any):
        """Storage is the final stage, no next stage"""
        pass
    
    async def _get_or_create_collection(self, user_id: str, collection_name: str):
        """Get or create ChromaDB collection for user"""
        import asyncio
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.chroma_manager.get_or_create_collection,
            user_id,
            collection_name
        )
    
    async def _store_batch(
        self,
        collection,
        ids: list,
        embeddings: list,
        documents: list,
        metadatas: list
    ):
        """Store a batch of embeddings in ChromaDB"""
        import asyncio
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            collection.add,
            ids,
            embeddings,
            metadatas,
            documents
        )
    
    async def _update_database(self, media_id: int, total_chunks: int):
        """Update SQL database with vector processing status"""
        import asyncio
        
        loop = asyncio.get_event_loop()
        
        # Update media table
        await loop.run_in_executor(
            None,
            update_media_table_vector_processing_status,
            media_id,
            "completed"
        )
        
        # Update chunks table
        await loop.run_in_executor(
            None,
            update_media_chunks_table_vector_processing_status,
            media_id,
            "completed"
        )
    
    async def _update_job_progress(self, job_id: str, percentage: float):
        """Update job progress information"""
        job_key = f"job:{job_id}"
        await self.redis_client.hset(
            job_key,
            mapping={
                "progress_percentage": percentage
            }
        )