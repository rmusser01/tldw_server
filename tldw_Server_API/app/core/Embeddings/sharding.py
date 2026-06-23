# sharding.py
# Sharding implementation for distributed embedding storage and retrieval

import hashlib
import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings
from loguru import logger


@dataclass
class ShardInfo:
    """Information about a single shard"""
    shard_id: int
    name: str
    path: str
    collection_prefix: str
    item_count: int
    size_bytes: int
    is_active: bool
    client: Optional[Any] = None


class ConsistentHashRing:
    """
    Consistent hashing implementation for shard distribution.
    Ensures minimal data movement when shards are added/removed.
    """

    def __init__(self, virtual_nodes: int = 150):
        """
        Initialize consistent hash ring.

        Args:
            virtual_nodes: Number of virtual nodes per shard for better distribution
        """
        self.virtual_nodes = virtual_nodes
        self.ring: dict[int, int] = {}  # hash -> shard_id
        self.shards: dict[int, ShardInfo] = {}  # shard_id -> ShardInfo
        self._lock = threading.RLock()

    def add_shard(self, shard: ShardInfo):
        """Add a shard to the ring"""
        with self._lock:
            self.shards[shard.shard_id] = shard

            # Add virtual nodes for this shard
            for i in range(self.virtual_nodes):
                virtual_key = f"{shard.shard_id}:{i}"
                hash_value = self._hash(virtual_key)
                self.ring[hash_value] = shard.shard_id

            logger.info(f"Added shard {shard.shard_id} to ring with {self.virtual_nodes} virtual nodes")

    def remove_shard(self, shard_id: int):
        """Remove a shard from the ring"""
        with self._lock:
            if shard_id not in self.shards:
                return

            # Remove virtual nodes
            keys_to_remove = []
            for hash_value, sid in self.ring.items():
                if sid == shard_id:
                    keys_to_remove.append(hash_value)

            for key in keys_to_remove:
                del self.ring[key]

            del self.shards[shard_id]
            logger.info(f"Removed shard {shard_id} from ring")

    def get_shard(self, key: str) -> Optional[ShardInfo]:
        """Get the shard for a given key"""
        with self._lock:
            if not self.ring:
                return None

            hash_value = self._hash(key)

            # Find the next shard in the ring
            sorted_hashes = sorted(self.ring.keys())

            # Binary search for the appropriate shard
            for ring_hash in sorted_hashes:
                if ring_hash >= hash_value:
                    shard_id = self.ring[ring_hash]
                    return self.shards.get(shard_id)

            # Wrap around to the first shard
            shard_id = self.ring[sorted_hashes[0]]
            return self.shards.get(shard_id)

    def _hash(self, key: str) -> int:
        """Generate hash for a key"""
        return int(hashlib.md5(key.encode(), usedforsecurity=False).hexdigest(), 16)

    def get_shard_distribution(self) -> dict[int, int]:
        """Get current distribution of keys across shards"""
        distribution = defaultdict(int)
        for shard_id in self.ring.values():
            distribution[shard_id] += 1
        return dict(distribution)


class EmbeddingShardManager:
    """
    Manages sharding of embeddings across multiple ChromaDB instances.
    Provides transparent sharding for scalability.
    """

    def __init__(
        self,
        base_path: str = "./shards",
        num_shards: int = 4,
        replication_factor: int = 1,
        max_items_per_shard: int = 1000000
    ):
        """
        Initialize shard manager.

        Args:
            base_path: Base path for shard storage
            num_shards: Number of shards to create
            replication_factor: Number of replicas per shard
            max_items_per_shard: Maximum items per shard before splitting
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.num_shards = num_shards
        self.replication_factor = replication_factor
        self.max_items_per_shard = max_items_per_shard

        # Initialize consistent hash ring
        self.hash_ring = ConsistentHashRing()

        # Shard statistics
        self.stats = {
            'total_operations': 0,
            'shard_hits': defaultdict(int),
            'rebalances': 0
        }

        # Initialize shards
        self._initialize_shards()

        logger.info(
            f"Shard manager initialized with {num_shards} shards, "
            f"replication factor {replication_factor}"
        )

    def _initialize_shards(self):
        """Initialize shard instances"""
        for i in range(self.num_shards):
            shard_path = self.base_path / f"shard_{i}"
            shard_path.mkdir(exist_ok=True)

            # Create ChromaDB client for this shard
            client = chromadb.PersistentClient(
                path=str(shard_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            shard = ShardInfo(
                shard_id=i,
                name=f"shard_{i}",
                path=str(shard_path),
                collection_prefix=f"shard{i}_",
                item_count=0,
                size_bytes=0,
                is_active=True,
                client=client
            )

            self.hash_ring.add_shard(shard)

    def get_shard_for_key(self, key: str) -> ShardInfo:
        """
        Get the appropriate shard for a given key.

        Args:
            key: Key to shard (e.g., user_id, document_id)

        Returns:
            ShardInfo for the selected shard
        """
        shard = self.hash_ring.get_shard(key)

        if not shard:
            raise ValueError("No shards available")

        self.stats['shard_hits'][shard.shard_id] += 1
        self.stats['total_operations'] += 1

        return shard

    def get_shard_for_collection(self, collection_name: str) -> ShardInfo:
        """
        Get shard for a collection based on its name.

        Args:
            collection_name: Name of the collection

        Returns:
            ShardInfo for the selected shard
        """
        return self.get_shard_for_key(collection_name)

    def add_embedding(
        self,
        collection_name: str,
        embedding_id: str,
        embedding: list[float],
        metadata: dict[str, Any],
        document: str
    ) -> str:
        """
        Add an embedding to the appropriate shard.

        Args:
            collection_name: Collection to add to
            embedding_id: Unique ID for the embedding
            embedding: The embedding vector
            metadata: Associated metadata
            document: The source document

        Returns:
            Shard ID where embedding was stored
        """
        # Get shard for this collection
        shard = self.get_shard_for_collection(collection_name)

        # Get or create collection in shard
        full_collection_name = f"{shard.collection_prefix}{collection_name}"

        try:
            collection = shard.client.get_or_create_collection(
                name=full_collection_name
            )

            # Add embedding
            collection.add(
                ids=[embedding_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[document]
            )

            # Update shard statistics
            shard.item_count += 1

            # Check if shard needs splitting
            if shard.item_count > self.max_items_per_shard:
                self._split_shard(shard)

            logger.debug(f"Added embedding {embedding_id} to shard {shard.shard_id}")
            return f"shard_{shard.shard_id}"

        except Exception as e:
            logger.error(f"Failed to add embedding to shard {shard.shard_id}: {e}")
            raise

    def query_embedding(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 10,
        where: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Query embeddings across shards.

        Args:
            collection_name: Collection to query
            query_embedding: Query vector
            n_results: Number of results to return
            where: Optional filter conditions

        Returns:
            Query results merged from all relevant shards
        """
        all_results = {
            'ids': [],
            'distances': [],
            'metadatas': [],
            'documents': []
        }

        # Query all shards (in production, you might optimize this)
        for shard in self.hash_ring.shards.values():
            if not shard.is_active:
                continue

            full_collection_name = f"{shard.collection_prefix}{collection_name}"

            try:
                # Check if collection exists in this shard
                collections = shard.client.list_collections()
                collection_names = [c.name for c in collections]

                if full_collection_name not in collection_names:
                    continue

                collection = shard.client.get_collection(full_collection_name)

                # Query the shard
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where
                )

                # Merge results
                if results['ids'] and results['ids'][0]:
                    all_results['ids'].extend(results['ids'][0])
                    all_results['distances'].extend(results['distances'][0])
                    all_results['metadatas'].extend(results['metadatas'][0])
                    all_results['documents'].extend(results['documents'][0])

            except Exception as e:
                logger.error(f"Error querying shard {shard.shard_id}: {e}")
                continue

        # Sort by distance and return top n_results
        if all_results['ids']:
            sorted_indices = sorted(
                range(len(all_results['distances'])),
                key=lambda i: all_results['distances'][i]
            )[:n_results]

            return {
                'ids': [[all_results['ids'][i] for i in sorted_indices]],
                'distances': [[all_results['distances'][i] for i in sorted_indices]],
                'metadatas': [[all_results['metadatas'][i] for i in sorted_indices]],
                'documents': [[all_results['documents'][i] for i in sorted_indices]]
            }

        return {'ids': [[]], 'distances': [[]], 'metadatas': [[]], 'documents': [[]]}

    def _split_shard(self, shard: ShardInfo):
        """
        Split a shard when it gets too large.

        Args:
            shard: Shard to split
        """
        logger.info(f"Splitting shard {shard.shard_id} (size: {shard.item_count})")

        # Create new shard
        new_shard_id = max(self.hash_ring.shards.keys()) + 1
        new_shard_path = self.base_path / f"shard_{new_shard_id}"
        new_shard_path.mkdir(exist_ok=True)

        new_client = chromadb.PersistentClient(
            path=str(new_shard_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        new_shard = ShardInfo(
            shard_id=new_shard_id,
            name=f"shard_{new_shard_id}",
            path=str(new_shard_path),
            collection_prefix=f"shard{new_shard_id}_",
            item_count=0,
            size_bytes=0,
            is_active=True,
            client=new_client
        )

        # Add to hash ring
        self.hash_ring.add_shard(new_shard)

        self.stats['rebalances'] += 1

        logger.info(f"Created new shard {new_shard_id}")

        # Note: In production, you would migrate some data from the old shard
        # to the new shard to balance the load

    def rebalance_shards(self):
        """Rebalance data across shards for even distribution"""
        logger.info("Starting shard rebalancing...")

        # Calculate average items per shard
        total_items = sum(s.item_count for s in self.hash_ring.shards.values())
        avg_items = total_items / len(self.hash_ring.shards)

        # Identify overloaded and underloaded shards
        overloaded = []
        underloaded = []

        for shard in self.hash_ring.shards.values():
            if shard.item_count > avg_items * 1.2:  # 20% over average
                overloaded.append(shard)
            elif shard.item_count < avg_items * 0.8:  # 20% under average
                underloaded.append(shard)

        # Note: Actual data migration would be implemented here
        logger.info(
            f"Rebalancing: {len(overloaded)} overloaded, "
            f"{len(underloaded)} underloaded shards"
        )

        self.stats['rebalances'] += 1

    def get_statistics(self) -> dict[str, Any]:
        """Get sharding statistics"""
        shard_stats = []

        for shard in self.hash_ring.shards.values():
            shard_stats.append({
                'shard_id': shard.shard_id,
                'item_count': shard.item_count,
                'is_active': shard.is_active,
                'hits': self.stats['shard_hits'][shard.shard_id]
            })

        return {
            'num_shards': len(self.hash_ring.shards),
            'total_operations': self.stats['total_operations'],
            'rebalances': self.stats['rebalances'],
            'shards': shard_stats,
            'distribution': self.hash_ring.get_shard_distribution()
        }

    def add_replica(self, shard_id: int) -> ShardInfo:
        """
        Add a replica for a shard.

        Args:
            shard_id: ID of shard to replicate

        Returns:
            New replica ShardInfo
        """
        if shard_id not in self.hash_ring.shards:
            raise ValueError(f"Shard {shard_id} not found")

        original_shard = self.hash_ring.shards[shard_id]

        # Create replica
        replica_id = f"{shard_id}_replica_{self.replication_factor}"
        replica_path = self.base_path / f"shard_{replica_id}"
        replica_path.mkdir(exist_ok=True)

        # Note: In production, you would copy data from original shard

        logger.info(f"Created replica {replica_id} for shard {shard_id}")

        return original_shard  # Simplified for now

    def shutdown(self):
        """Gracefully shutdown all shards"""
        logger.info("Shutting down shard manager...")

        for shard in self.hash_ring.shards.values():
            if shard.client:
                try:
                    # Close ChromaDB client connections
                    pass  # ChromaDB doesn't have explicit close
                except Exception as e:
                    logger.error(f"Error closing shard {shard.shard_id}: {e}")

        logger.info("Shard manager shutdown complete")


# Global shard manager
_shard_manager: Optional[EmbeddingShardManager] = None


def get_shard_manager() -> EmbeddingShardManager:
    """Get or create the global shard manager."""
    global _shard_manager
    enabled = str(os.getenv("EMBEDDINGS_ENABLE_EXPERIMENTAL_SHARDING") or "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        raise RuntimeError("Embeddings experimental sharding is disabled")
    if _shard_manager is None:
        _shard_manager = EmbeddingShardManager()
    return _shard_manager


# Example configuration for sharding
SHARDING_CONFIG_EXAMPLE = """
sharding:
  enabled: true
  num_shards: 4
  replication_factor: 2
  max_items_per_shard: 1000000
  base_path: ./embedding_shards

  # Consistent hashing settings
  virtual_nodes: 150

  # Rebalancing settings
  auto_rebalance: true
  rebalance_threshold: 0.2  # 20% imbalance triggers rebalance

  # Replica settings
  read_replicas: true
  write_replicas: false  # Write to primary only
  replica_sync_interval: 60  # seconds
"""
