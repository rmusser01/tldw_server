# embedding_worker.py
# Worker for generating embeddings from chunks

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger

from ..Embeddings_Server.Embeddings_Create import (
    HFModelCfg,
    ONNXModelCfg,
    OpenAIModelCfg,
    LocalAPICfg,
    create_embeddings_batch
)
from ..queue_schemas import (
    EmbeddingData,
    EmbeddingMessage,
    JobStatus,
    StorageMessage,
)
from .base_worker import BaseWorker, WorkerConfig


class EmbeddingWorkerConfig(WorkerConfig):
    """Extended configuration for embedding workers"""
    default_model_provider: str = "huggingface"
    default_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_batch_size: int = 32
    gpu_id: Optional[int] = None


class EmbeddingWorker(BaseWorker):
    """Worker that generates embeddings from text chunks"""
    
    def __init__(self, config: EmbeddingWorkerConfig):
        super().__init__(config)
        self.embedding_config = config
        self.storage_queue = config.queue_name.replace("embedding", "storage")
        
        # Model configuration cache
        self.model_configs = {
            "huggingface": HFModelCfg(
                model_name_or_path=config.default_model_name,
                trust_remote_code=False
            ),
            "openai": OpenAIModelCfg(
                model_name_or_path="text-embedding-3-small"
            ),
            # Add more default configs as needed
        }
        
        # Set GPU if specified
        if config.gpu_id is not None:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    
    def _parse_message(self, data: Dict[str, Any]) -> EmbeddingMessage:
        """Parse raw message data into EmbeddingMessage"""
        return EmbeddingMessage(**data)
    
    async def process_message(self, message: EmbeddingMessage) -> Optional[StorageMessage]:
        """Process embedding message and generate embeddings"""
        logger.info(f"Processing embedding job {message.job_id} with {len(message.chunks)} chunks")
        
        start_time = time.time()
        
        try:
            # Update job status
            await self._update_job_status(message.job_id, JobStatus.EMBEDDING)
            
            # Get model configuration
            model_config = self._get_model_config(message)
            
            # Process chunks in batches
            embedding_data_list = []
            batch_size = message.batch_size or self.embedding_config.max_batch_size
            
            for i in range(0, len(message.chunks), batch_size):
                batch_chunks = message.chunks[i:i + batch_size]
                batch_texts = [chunk.content for chunk in batch_chunks]
                
                # Generate embeddings
                embeddings = await self._generate_embeddings(
                    batch_texts,
                    model_config,
                    message.model_provider
                )
                
                # Create embedding data objects
                for chunk, embedding in zip(batch_chunks, embeddings):
                    embedding_data = EmbeddingData(
                        chunk_id=chunk.chunk_id,
                        embedding=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                        model_used=model_config.model_name_or_path,
                        dimensions=len(embedding),
                        metadata=chunk.metadata
                    )
                    embedding_data_list.append(embedding_data)
                
                # Update progress
                progress = 25 + (50 * (i + len(batch_chunks)) / len(message.chunks))
                await self._update_job_progress(
                    message.job_id,
                    progress,
                    chunks_processed=i + len(batch_chunks)
                )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Create storage message
            storage_message = StorageMessage(
                job_id=message.job_id,
                user_id=message.user_id,
                media_id=message.media_id,
                priority=message.priority,
                user_tier=message.user_tier,
                created_at=message.created_at,
                embeddings=embedding_data_list,
                collection_name=f"user_{message.user_id}_media_{message.media_id}",
                total_chunks=len(message.chunks),
                processing_time_ms=processing_time_ms,
                metadata={
                    "model_provider": message.model_provider,
                    "model_name": model_config.model_name_or_path
                }
            )
            
            logger.info(
                f"Generated {len(embedding_data_list)} embeddings for job {message.job_id} "
                f"in {processing_time_ms}ms"
            )
            return storage_message
            
        except Exception as e:
            logger.error(f"Error generating embeddings for job {message.job_id}: {e}")
            raise
    
    async def _send_to_next_stage(self, result: StorageMessage):
        """Send embeddings to storage queue"""
        await self.redis_client.xadd(
            self.storage_queue,
            result.dict()
        )
        logger.debug(f"Sent job {result.job_id} to storage queue")
    
    def _get_model_config(self, message: EmbeddingMessage) -> Union[HFModelCfg, ONNXModelCfg, OpenAIModelCfg, LocalAPICfg]:
        """Get or create model configuration"""
        if message.model_config:
            # Use provided config
            provider = message.model_provider
            if provider == "huggingface":
                return HFModelCfg(**message.model_config)
            elif provider == "onnx":
                return ONNXModelCfg(**message.model_config)
            elif provider == "openai":
                return OpenAIModelCfg(**message.model_config)
            elif provider == "local_api":
                return LocalAPICfg(**message.model_config)
        
        # Use default config based on user tier
        if message.user_tier == "enterprise":
            return self.model_configs.get("openai", self.model_configs["huggingface"])
        else:
            return self.model_configs["huggingface"]
    
    async def _generate_embeddings(
        self,
        texts: List[str],
        config: Union[HFModelCfg, ONNXModelCfg, OpenAIModelCfg, LocalAPICfg],
        provider: str
    ) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts"""
        # Use the existing create_embeddings_batch function
        # This runs in a thread pool to avoid blocking the event loop
        import asyncio
        
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            create_embeddings_batch,
            texts,
            config.model_name_or_path,
            provider,
            config.api_url if hasattr(config, 'api_url') else None,
            config.api_key if hasattr(config, 'api_key') else None
        )
        
        return embeddings
    
    async def _update_job_progress(self, job_id: str, percentage: float, chunks_processed: int):
        """Update job progress information"""
        job_key = f"job:{job_id}"
        await self.redis_client.hset(
            job_key,
            mapping={
                "progress_percentage": percentage,
                "chunks_processed": chunks_processed
            }
        )
    
    async def _calculate_load(self) -> float:
        """Calculate current worker load based on GPU utilization"""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            if self.embedding_config.gpu_id is not None:
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.embedding_config.gpu_id)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return util.gpu / 100.0
            
        except Exception:
            # Fallback to base implementation if GPU monitoring fails
            pass
        
        return await super()._calculate_load()