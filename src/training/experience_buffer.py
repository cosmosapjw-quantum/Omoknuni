"""
Memory-mapped experience buffer implementation with Parquet storage and LRU caching.

Provides efficient storage and sampling of training examples from self-play games.
Optimized for large-scale training data with memory-mapped access patterns.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict, Any, Optional
import mmap
import os
import pickle
import random
import time
from collections import OrderedDict
import logging
from threading import Lock

from specs.contracts.training_api import (
    ExperienceBuffer, TrainingExample, GameResult
)

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache for training examples."""

    def __init__(self, max_size_mb: int):
        self.max_size_mb = max_size_mb
        self.max_entries = max_size_mb * 1024 * 1024 // 4000  # ~4KB per example estimate
        self.cache = OrderedDict()
        self.lock = Lock()

    def get(self, key: str) -> Optional[TrainingExample]:
        """Get example from cache, moving to end if found."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None

    def put(self, key: str, value: TrainingExample) -> None:
        """Add example to cache, evicting LRU if needed."""
        with self.lock:
            if key in self.cache:
                # Update existing entry
                self.cache.pop(key)
            elif len(self.cache) >= self.max_entries:
                # Evict least recently used
                self.cache.popitem(last=False)

            self.cache[key] = value

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_entries,
                'utilization': len(self.cache) / self.max_entries if self.max_entries > 0 else 0.0
            }


class MemoryMappedExperienceBuffer(ExperienceBuffer):
    """Memory-mapped experience buffer with Parquet storage and LRU caching."""

    def __init__(self,
                 buffer_path: Path,
                 max_examples: int = 1_000_000,
                 cache_size_mb: int = 512):
        """Initialize memory-mapped experience buffer.

        Args:
            buffer_path: Directory for memory-mapped storage
            max_examples: Maximum training examples to store
            cache_size_mb: RAM cache size in megabytes
        """
        self.buffer_path = Path(buffer_path)
        self.max_examples = max_examples
        self.cache_size_mb = cache_size_mb

        # Create buffer directory
        self.buffer_path.mkdir(parents=True, exist_ok=True)

        # Initialize data structures
        self._parquet_file = self.buffer_path / "examples.parquet"
        self._metadata_file = self.buffer_path / "metadata.pkl"
        self._index_file = self.buffer_path / "index.pkl"

        # LRU cache for frequently accessed examples
        self.cache = LRUCache(cache_size_mb)

        # Thread safety
        self.lock = Lock()

        # Load existing data or initialize
        self._load_or_initialize()

        logger.info(f"ExperienceBuffer initialized: {len(self.index)} examples, "
                   f"cache size: {cache_size_mb}MB")

    def _load_or_initialize(self) -> None:
        """Load existing buffer data or initialize empty buffer."""
        try:
            # Load metadata
            if self._metadata_file.exists():
                with open(self._metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
            else:
                self.metadata = {
                    'total_examples': 0,
                    'total_games': 0,
                    'game_type_counts': {},
                    'created_at': time.time(),
                    'last_modified': time.time()
                }

            # Load index
            if self._index_file.exists():
                with open(self._index_file, 'rb') as f:
                    self.index = pickle.load(f)
            else:
                self.index = []  # List of (game_id, example_idx, game_type, file_offset)

        except Exception as e:
            logger.warning(f"Failed to load existing buffer data: {e}. Initializing empty buffer.")
            self.metadata = {
                'total_examples': 0,
                'total_games': 0,
                'game_type_counts': {},
                'created_at': time.time(),
                'last_modified': time.time()
            }
            self.index = []

    def _save_metadata(self) -> None:
        """Save metadata and index to disk."""
        self.metadata['last_modified'] = time.time()

        with open(self._metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)

        with open(self._index_file, 'wb') as f:
            pickle.dump(self.index, f)

    def add_games(self, games: List[GameResult]) -> None:
        """Add games to experience buffer.

        Args:
            games: List of completed self-play games
        """
        if not games:
            return

        with self.lock:
            # Convert games to training examples
            new_examples = []
            for game in games:
                game_type = game.examples[0].game_type if game.examples else "unknown"

                for example in game.examples:
                    # Create row data for Parquet
                    example_data = {
                        'game_id': example.game_id,
                        'game_type': example.game_type,
                        'move_number': example.move_number,
                        'value': example.value,
                        'state_data': example.state.tobytes(),  # Serialize numpy array
                        'state_shape': list(example.state.shape),
                        'policy_data': example.policy.tobytes(),  # Serialize numpy array
                        'policy_shape': list(example.policy.shape),
                        'timestamp': time.time()
                    }
                    new_examples.append(example_data)

                # Update game type counts
                self.metadata['game_type_counts'][game_type] = (
                    self.metadata['game_type_counts'].get(game_type, 0) + 1
                )

            if not new_examples:
                return

            # Convert to DataFrame for Parquet
            df = pd.DataFrame(new_examples)

            # Append to Parquet file
            table = pa.Table.from_pandas(df)

            if self._parquet_file.exists():
                # Read existing table and concatenate
                existing_table = pq.read_table(self._parquet_file)
                combined_table = pa.concat_tables([existing_table, table])
            else:
                combined_table = table

            # Handle buffer size limit
            if len(combined_table) > self.max_examples:
                # Keep only the most recent examples
                start_idx = len(combined_table) - self.max_examples
                combined_table = combined_table.slice(start_idx)

                # Clear cache as indices have changed
                self.cache.clear()
                logger.info(f"Buffer size limit reached. Keeping last {self.max_examples} examples.")

            # Write back to Parquet file
            pq.write_table(combined_table, self._parquet_file)

            # Update index
            start_offset = len(self.index)
            for i, example_data in enumerate(new_examples):
                if start_offset + i < self.max_examples:  # Only add if within limit
                    self.index.append((
                        example_data['game_id'],
                        example_data['move_number'],
                        example_data['game_type'],
                        start_offset + i  # File offset in Parquet table
                    ))

            # Trim index if needed
            if len(self.index) > self.max_examples:
                self.index = self.index[-self.max_examples:]

            # Update metadata
            self.metadata['total_examples'] = len(self.index)
            self.metadata['total_games'] += len(games)

            # Save metadata
            self._save_metadata()

            logger.info(f"Added {len(new_examples)} examples from {len(games)} games. "
                       f"Total: {len(self.index)} examples")

    def sample_batch(self,
                    batch_size: int,
                    game_types: Optional[List[str]] = None) -> List[TrainingExample]:
        """Sample training batch from buffer.

        Args:
            batch_size: Number of examples to sample
            game_types: Restrict to specific game types (None = all)

        Returns:
            List of training examples
        """
        with self.lock:
            if not self.index:
                return []

            # Filter indices by game type if specified
            if game_types:
                filtered_indices = [
                    (i, entry) for i, entry in enumerate(self.index)
                    if entry[2] in game_types  # entry[2] is game_type
                ]
            else:
                filtered_indices = list(enumerate(self.index))

            if not filtered_indices:
                return []

            # Sample random indices
            sample_size = min(batch_size, len(filtered_indices))
            sampled_indices = random.sample(filtered_indices, sample_size)

            # Load examples
            examples = []
            cache_hits = 0

            for idx, entry in sampled_indices:
                game_id, move_number, game_type, file_offset = entry
                cache_key = f"{game_id}_{move_number}"

                # Try cache first
                cached_example = self.cache.get(cache_key)
                if cached_example:
                    examples.append(cached_example)
                    cache_hits += 1
                    continue

                # Load from Parquet file
                try:
                    # Read single row from Parquet
                    table = pq.read_table(self._parquet_file)
                    if file_offset >= len(table):
                        logger.warning(f"File offset {file_offset} out of bounds for table length {len(table)}")
                        continue

                    row = table.slice(file_offset, 1).to_pandas().iloc[0]

                    # Reconstruct numpy arrays
                    state = np.frombuffer(row['state_data'], dtype=np.float32).reshape(row['state_shape'])
                    policy = np.frombuffer(row['policy_data'], dtype=np.float32).reshape(row['policy_shape'])

                    # Create TrainingExample
                    example = TrainingExample(
                        state=state,
                        policy=policy,
                        value=row['value'],
                        game_type=row['game_type'],
                        move_number=row['move_number'],
                        game_id=row['game_id']
                    )

                    # Cache the example
                    self.cache.put(cache_key, example)
                    examples.append(example)

                except Exception as e:
                    logger.warning(f"Failed to load example at offset {file_offset}: {e}")
                    continue

            logger.debug(f"Sampled {len(examples)} examples (cache hits: {cache_hits}/{sample_size})")
            return examples

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics.

        Returns:
            dict: Stats including size, distribution, memory usage
        """
        with self.lock:
            # Calculate storage size
            storage_size_mb = 0
            if self._parquet_file.exists():
                storage_size_mb = self._parquet_file.stat().st_size / (1024 * 1024)

            cache_stats = self.cache.stats()

            return {
                'total_examples': len(self.index),
                'total_games': self.metadata.get('total_games', 0),
                'game_type_distribution': dict(self.metadata.get('game_type_counts', {})),
                'storage_size_mb': round(storage_size_mb, 2),
                'cache_stats': cache_stats,
                'buffer_utilization': len(self.index) / self.max_examples,
                'created_at': self.metadata.get('created_at'),
                'last_modified': self.metadata.get('last_modified')
            }

    def cleanup(self, keep_last_n: int = 100_000) -> None:
        """Remove old examples to manage storage.

        Args:
            keep_last_n: Number of most recent examples to retain
        """
        with self.lock:
            if len(self.index) <= keep_last_n:
                logger.info(f"Buffer has {len(self.index)} examples, no cleanup needed (keeping {keep_last_n})")
                return

            logger.info(f"Cleaning up buffer: keeping last {keep_last_n} of {len(self.index)} examples")

            # Read current Parquet table
            if not self._parquet_file.exists():
                return

            table = pq.read_table(self._parquet_file)

            # Keep only the last keep_last_n examples
            if len(table) > keep_last_n:
                start_idx = len(table) - keep_last_n
                trimmed_table = table.slice(start_idx)

                # Write back to file
                pq.write_table(trimmed_table, self._parquet_file)

                # Update index
                self.index = self.index[-keep_last_n:]

                # Update metadata
                self.metadata['total_examples'] = len(self.index)
                self._save_metadata()

                # Clear cache as indices have changed
                self.cache.clear()

                logger.info(f"Cleanup complete: {len(self.index)} examples remaining")


def create_experience_buffer(buffer_path: Path,
                           max_examples: int = 1_000_000,
                           cache_size_mb: int = 512) -> ExperienceBuffer:
    """Factory function to create experience buffer.

    Args:
        buffer_path: Directory for buffer storage
        max_examples: Maximum examples to store
        cache_size_mb: RAM cache size in megabytes

    Returns:
        ExperienceBuffer instance
    """
    return MemoryMappedExperienceBuffer(
        buffer_path=buffer_path,
        max_examples=max_examples,
        cache_size_mb=cache_size_mb
    )