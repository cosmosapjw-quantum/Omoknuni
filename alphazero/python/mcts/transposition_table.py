import threading
from typing import Dict, Any, Optional, Tuple

class TranspositionTable:
    """
    Transposition table for MCTS to reuse evaluations of previously seen states.
    
    This is a simple hash table that maps game state hashes to nodes in the search tree.
    Thread-safe implementation for parallel MCTS.
    """
    def __init__(self, max_size: int = 1000000):
        """
        Initialize the transposition table.
        
        Args:
            max_size: Maximum number of entries in the table
        """
        self.max_size = max_size
        self.table: Dict[int, Any] = {}
        self.lock = threading.RLock()
    
    def lookup(self, hash_value: int) -> Optional[Any]:
        """
        Look up a node in the table.
        
        Args:
            hash_value: Hash value of the game state
            
        Returns:
            The node if found, None otherwise
        """
        with self.lock:
            return self.table.get(hash_value)
    
    def store(self, hash_value: int, node: Any) -> None:
        """
        Store a node in the table.
        
        Args:
            hash_value: Hash value of the game state
            node: Node to store
        """
        with self.lock:
            # If table is full, don't add new entries
            if len(self.table) >= self.max_size:
                return
            
            self.table[hash_value] = node
    
    def clear(self) -> None:
        """Clear the table."""
        with self.lock:
            self.table.clear()
    
    def size(self) -> int:
        """Return the number of entries in the table."""
        with self.lock:
            return len(self.table)
    
    def __contains__(self, hash_value: int) -> bool:
        """Check if a hash value is in the table."""
        with self.lock:
            return hash_value in self.table


class LRUTranspositionTable(TranspositionTable):
    """
    Transposition table with LRU (Least Recently Used) eviction policy.
    
    This is an enhanced version of the transposition table that removes
    the least recently used entries when the table is full.
    """
    def __init__(self, max_size: int = 1000000):
        """
        Initialize the LRU transposition table.
        
        Args:
            max_size: Maximum number of entries in the table
        """
        super().__init__(max_size)
        self.access_count = 0
        self.access_time: Dict[int, int] = {}
    
    def lookup(self, hash_value: int) -> Optional[Any]:
        """
        Look up a node in the table and update its access time.
        
        Args:
            hash_value: Hash value of the game state
            
        Returns:
            The node if found, None otherwise
        """
        with self.lock:
            node = self.table.get(hash_value)
            if node is not None:
                self.access_count += 1
                self.access_time[hash_value] = self.access_count
            return node
    
    def store(self, hash_value: int, node: Any) -> None:
        """
        Store a node in the table, evicting the least recently used entry if necessary.
        
        Args:
            hash_value: Hash value of the game state
            node: Node to store
        """
        with self.lock:
            # If table is full, remove the least recently used entry
            if len(self.table) >= self.max_size:
                # Find the least recently used entry
                lru_hash = min(self.access_time.items(), key=lambda x: x[1])[0]
                del self.table[lru_hash]
                del self.access_time[lru_hash]
            
            # Store the new entry
            self.table[hash_value] = node
            self.access_count += 1
            self.access_time[hash_value] = self.access_count
    
    def clear(self) -> None:
        """Clear the table and access times."""
        with self.lock:
            self.table.clear()
            self.access_time.clear()
            self.access_count = 0