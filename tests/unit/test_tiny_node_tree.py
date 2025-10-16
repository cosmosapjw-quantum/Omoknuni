"""
Unit tests for TinyNodeTree (T024f-1: TinyNode Storage Layer)

Tests allocation, deallocation, capacity management, and basic thread safety.
"""

import pytest
import sys
import os
import threading
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import after path setup
import mcts_py


class TestTinyNodeTreeBasicAllocation:
    """Test basic allocation and deallocation"""

    def test_create_tree(self):
        """Test tree creation"""
        tree = mcts_py.TinyNodeTree(1000)
        assert tree is not None
        assert tree.get_max_nodes() == 1000
        assert tree.get_node_count() == 0
        assert tree.get_bytes_per_node() == 64  # Aligned to 64 bytes

    def test_allocate_single_node(self):
        """Test single node allocation"""
        tree = mcts_py.TinyNodeTree(1000)

        idx = tree.allocate_node()
        assert idx == 0  # First allocation
        assert tree.get_node_count() == 1
        assert tree.is_valid_index(idx)

        # Get node and verify initialization
        node = tree.get_node(idx)
        assert node is not None
        assert node.get_visit_count() == 0
        assert node.get_total_value() == 0
        assert node.get_virtual_loss() == 0

    def test_allocate_multiple_nodes(self):
        """Test multiple node allocations"""
        tree = mcts_py.TinyNodeTree(100)

        indices = []
        for i in range(10):
            idx = tree.allocate_node()
            assert idx == i  # Sequential allocation
            assert tree.is_valid_index(idx)
            indices.append(idx)

        assert tree.get_node_count() == 10

        # Verify all nodes are accessible
        for idx in indices:
            node = tree.get_node(idx)
            assert node is not None

    def test_allocate_beyond_capacity(self):
        """Test allocation failure when capacity exceeded"""
        tree = mcts_py.TinyNodeTree(5)

        # Allocate up to capacity
        for i in range(5):
            idx = tree.allocate_node()
            assert idx >= 0

        # Next allocation should fail
        idx = tree.allocate_node()
        assert idx == -1  # Allocation failed
        assert tree.get_node_count() == 5  # Count unchanged

    def test_deallocate_node(self):
        """Test node deallocation"""
        tree = mcts_py.TinyNodeTree(100)

        # Allocate nodes
        idx1 = tree.allocate_node()
        idx2 = tree.allocate_node()
        idx3 = tree.allocate_node()

        assert tree.get_node_count() == 3

        # Deallocate middle node
        tree.deallocate_node(idx2)

        # Node count stays same (free list management)
        # But next allocation should reuse idx2
        idx4 = tree.allocate_node()
        assert idx4 == idx2  # Reused from free list

    def test_clear_tree(self):
        """Test tree clearing"""
        tree = mcts_py.TinyNodeTree(100)

        # Allocate several nodes
        for i in range(10):
            tree.allocate_node()

        assert tree.get_node_count() == 10

        # Clear tree
        tree.clear()

        assert tree.get_node_count() == 0
        assert tree.get_root_index() == -1  # No root


class TestTinyNodeTreeRootInitialization:
    """Test root node initialization"""

    def test_init_root(self):
        """Test root initialization"""
        tree = mcts_py.TinyNodeTree(1000)

        # Initialize root
        zobrist_hash = 0x123456789ABCDEF0
        root_idx = tree.init_root(zobrist_hash)

        assert root_idx == 0  # Root is always index 0
        assert tree.get_node_count() == 1
        assert tree.get_root_index() == 0

        # Verify root properties
        root = tree.get_node(root_idx)
        assert root is not None
        assert root.is_root()
        assert root.zobrist_hash == zobrist_hash
        assert root.parent_idx == 0  # Root points to self
        assert root.get_visit_count() == 1  # WU-UCT: root starts with 1 visit

    def test_init_root_clears_tree(self):
        """Test that init_root clears existing tree"""
        tree = mcts_py.TinyNodeTree(1000)

        # Allocate some nodes
        for i in range(5):
            tree.allocate_node()

        assert tree.get_node_count() == 5

        # Init root should clear tree first
        root_idx = tree.init_root(0xDEADBEEF)

        assert root_idx == 0
        assert tree.get_node_count() == 1  # Only root


class TestTinyNodeTreeCapacityManagement:
    """Test capacity and space checking"""

    def test_has_space_for(self):
        """Test space availability checking"""
        tree = mcts_py.TinyNodeTree(10)

        assert tree.has_space_for(10)
        assert tree.has_space_for(5)

        # Allocate 5 nodes
        for i in range(5):
            tree.allocate_node()

        assert tree.has_space_for(5)
        assert not tree.has_space_for(6)

        # Allocate remaining
        for i in range(5):
            tree.allocate_node()

        assert not tree.has_space_for(1)

    def test_memory_usage(self):
        """Test memory usage calculation"""
        tree = mcts_py.TinyNodeTree(1000)

        # Memory usage = max_nodes * bytes_per_node
        expected = 1000 * 64  # 64 bytes per node
        assert tree.get_memory_usage() == expected

        # Memory usage doesn't change with allocation count
        tree.allocate_node()
        assert tree.get_memory_usage() == expected


class TestTinyNodeTreeValidation:
    """Test tree structure validation"""

    def test_validate_empty_tree(self):
        """Test validation of empty tree"""
        tree = mcts_py.TinyNodeTree(1000)
        assert tree.validate()

    def test_validate_with_root(self):
        """Test validation with root node"""
        tree = mcts_py.TinyNodeTree(1000)
        tree.init_root(0x12345)

        assert tree.validate()

    def test_validate_with_nodes(self):
        """Test validation with multiple nodes"""
        tree = mcts_py.TinyNodeTree(1000)
        tree.init_root(0x12345)

        # Allocate some nodes
        for i in range(10):
            tree.allocate_node()

        assert tree.validate()


class TestTinyNodeTreeIndexValidation:
    """Test node index validation"""

    def test_valid_indices(self):
        """Test is_valid_index for valid indices"""
        tree = mcts_py.TinyNodeTree(100)

        # Allocate 5 nodes
        indices = [tree.allocate_node() for _ in range(5)]

        # All allocated indices should be valid
        for idx in indices:
            assert tree.is_valid_index(idx)

    def test_invalid_indices(self):
        """Test is_valid_index for invalid indices"""
        tree = mcts_py.TinyNodeTree(100)

        # Allocate 3 nodes (indices 0, 1, 2)
        for _ in range(3):
            tree.allocate_node()

        # Negative indices are invalid
        assert not tree.is_valid_index(-1)
        assert not tree.is_valid_index(-100)

        # Indices beyond allocated range are invalid
        assert not tree.is_valid_index(3)
        assert not tree.is_valid_index(100)

    def test_get_node_invalid_index(self):
        """Test get_node returns None for invalid indices"""
        tree = mcts_py.TinyNodeTree(100)

        tree.allocate_node()  # Allocate index 0

        # Invalid indices return None
        assert tree.get_node(-1) is None
        assert tree.get_node(1) is None
        assert tree.get_node(100) is None


class TestTinyNodeTreeNodeAccess:
    """Test TinyNode access and properties"""

    def test_node_structure_fields(self):
        """Test accessing TinyNode structure fields"""
        tree = mcts_py.TinyNodeTree(100)
        tree.init_root(0xABCDEF)

        root = tree.get_node(0)

        # Test field access
        assert root.move == 0  # Root has no move
        assert root.parent_idx == 0  # Root points to self
        assert root.first_child_idx == 0  # No children initially
        assert root.next_sibling_idx == 0  # No siblings
        assert root.zobrist_hash == 0xABCDEF

    def test_node_helper_methods(self):
        """Test TinyNode helper methods"""
        tree = mcts_py.TinyNodeTree(100)
        tree.init_root(0x12345)

        root = tree.get_node(0)

        # Test helper methods
        assert root.is_root()
        assert not root.is_terminal()
        assert not root.is_expanded()

        # Test value access
        assert root.get_visit_count() == 1  # WU-UCT: root starts with 1
        assert root.get_total_value() == 0
        assert root.get_virtual_loss() == 0
        assert root.get_q_value() == 0.0  # Q = W/N = 0/1 = 0


class TestTinyNodeTreeThreadSafety:
    """Test basic thread safety of allocation"""

    def test_concurrent_allocation(self):
        """Test concurrent node allocation from multiple threads"""
        tree = mcts_py.TinyNodeTree(1000)
        allocated = []
        errors = []
        lock = threading.Lock()

        def allocate_nodes(count):
            try:
                thread_indices = []
                for _ in range(count):
                    idx = tree.allocate_node()
                    if idx >= 0:
                        thread_indices.append(idx)
                    time.sleep(0.0001)  # Small delay to increase contention

                with lock:
                    allocated.extend(thread_indices)
            except Exception as e:
                with lock:
                    errors.append(str(e))

        # Launch 5 threads, each allocating 20 nodes
        threads = []
        for _ in range(5):
            t = threading.Thread(target=allocate_nodes, args=(20,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Check for errors
        assert len(errors) == 0, f"Errors during allocation: {errors}"

        # Verify all allocations succeeded
        assert len(allocated) == 100  # 5 threads * 20 nodes

        # Verify all indices are unique (no double-allocation)
        assert len(set(allocated)) == len(allocated), "Duplicate indices allocated!"

    def test_concurrent_allocation_with_deallocation(self):
        """Test concurrent allocation and deallocation"""
        tree = mcts_py.TinyNodeTree(100)
        operations = []
        lock = threading.Lock()

        def worker():
            try:
                # Allocate
                idx = tree.allocate_node()
                if idx >= 0:
                    with lock:
                        operations.append(('alloc', idx))

                    time.sleep(0.0001)

                    # Deallocate
                    tree.deallocate_node(idx)
                    with lock:
                        operations.append(('dealloc', idx))
            except Exception as e:
                with lock:
                    operations.append(('error', str(e)))

        # Launch 10 threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Check for errors
        errors = [op for op in operations if op[0] == 'error']
        assert len(errors) == 0, f"Errors: {errors}"

        # Verify all operations completed
        allocs = [op for op in operations if op[0] == 'alloc']
        deallocs = [op for op in operations if op[0] == 'dealloc']

        assert len(allocs) == 10
        assert len(deallocs) == 10


class TestTinyNodeTreePerformance:
    """Test allocation performance (O(1) verification)"""

    def test_allocation_is_o1(self):
        """Test that allocation time is constant (O(1))"""
        import time

        tree = mcts_py.TinyNodeTree(100000)

        # Measure allocation time for first batch
        start = time.perf_counter()
        for _ in range(1000):
            tree.allocate_node()
        time_first_1000 = time.perf_counter() - start

        # Measure allocation time for second batch (after 1000 allocations)
        start = time.perf_counter()
        for _ in range(1000):
            tree.allocate_node()
        time_second_1000 = time.perf_counter() - start

        # O(1) allocation: times should be similar (within 2x tolerance)
        # Allow 2x tolerance for warm-up effects
        assert time_second_1000 < time_first_1000 * 2, \
            f"Allocation time increased significantly: {time_first_1000:.6f}s -> {time_second_1000:.6f}s"


class TestTinyNodeTreeChildManagement:
    """Test child management with sibling linking (T024f-2)"""

    def test_add_single_child(self):
        """Test adding a single child to a node"""
        tree = mcts_py.TinyNodeTree(1000)
        tree.init_root(0x12345)

        # Add child to root
        child_idx = tree.add_child(
            parent_idx=0,
            move=42,
            prior_prob=0.5,
            zobrist_hash=0xABCDEF
        )

        assert child_idx > 0  # Child allocated
        assert tree.get_child_count(0) == 1  # Root has 1 child

        # Verify child properties
        child = tree.get_node(child_idx)
        assert child is not None
        assert child.move == 42
        assert child.parent_idx == 0
        assert child.zobrist_hash == 0xABCDEF
        assert abs(child.get_prior() - 0.5) < 0.001  # Scaled prior

        # Verify root is marked as expanded
        root = tree.get_node(0)
        assert root.is_expanded()

    def test_add_multiple_children(self):
        """Test adding multiple children to a node"""
        tree = mcts_py.TinyNodeTree(1000)
        tree.init_root(0x12345)

        # Add 5 children
        child_indices = []
        for i in range(5):
            child_idx = tree.add_child(
                parent_idx=0,
                move=i,
                prior_prob=0.2,
                zobrist_hash=0x1000 + i
            )
            assert child_idx > 0
            child_indices.append(child_idx)

        # Verify child count
        assert tree.get_child_count(0) == 5

        # Verify all children are unique
        assert len(set(child_indices)) == 5

    def test_get_children_list(self):
        """Test getting children as a list"""
        tree = mcts_py.TinyNodeTree(1000)
        tree.init_root(0x12345)

        # Add children
        expected_indices = []
        for i in range(3):
            child_idx = tree.add_child(0, i, 0.33, 0x1000 + i)
            expected_indices.append(child_idx)

        # Get children list
        children = tree.get_children(0)
        assert len(children) == 3

        # Children are added to front of list (reverse order)
        assert set(children) == set(expected_indices)

    def test_expand_node_with_numpy_arrays(self):
        """Test expand_node with numpy arrays"""
        import numpy as np

        tree = mcts_py.TinyNodeTree(1000)
        tree.init_root(0x12345)

        # Prepare expansion data
        num_children = 10
        moves = np.arange(num_children, dtype=np.uint16)
        priors = np.full(num_children, 1.0 / num_children, dtype=np.float32)
        zobrist_hashes = np.arange(0x1000, 0x1000 + num_children, dtype=np.uint64)

        # Expand root
        success = tree.expand_node(0, moves, priors, zobrist_hashes)
        assert success

        # Verify expansion
        assert tree.get_child_count(0) == num_children
        assert tree.get_node(0).is_expanded()

        # Verify each child
        children = tree.get_children(0)
        assert len(children) == num_children

        for child_idx in children:
            child = tree.get_node(child_idx)
            assert child.parent_idx == 0
            assert child.move < num_children
            assert abs(child.get_prior() - 0.1) < 0.01  # ~0.1 prior

    def test_expand_node_empty(self):
        """Test expand_node with zero children"""
        import numpy as np

        tree = mcts_py.TinyNodeTree(1000)
        tree.init_root(0x12345)

        # Empty arrays
        moves = np.array([], dtype=np.uint16)
        priors = np.array([], dtype=np.float32)
        zobrist_hashes = np.array([], dtype=np.uint64)

        # Should succeed (no-op)
        success = tree.expand_node(0, moves, priors, zobrist_hashes)
        assert success
        assert tree.get_child_count(0) == 0

    def test_expand_node_capacity_exceeded(self):
        """Test expand_node when capacity exceeded"""
        import numpy as np

        tree = mcts_py.TinyNodeTree(10)  # Small capacity
        tree.init_root(0x12345)

        # Try to expand with 20 children (exceeds capacity)
        moves = np.arange(20, dtype=np.uint16)
        priors = np.full(20, 0.05, dtype=np.float32)
        zobrist_hashes = np.arange(0x1000, 0x1000 + 20, dtype=np.uint64)

        # Should fail (partial expansion might occur)
        success = tree.expand_node(0, moves, priors, zobrist_hashes)
        assert not success

    def test_child_iteration_order(self):
        """Test that children maintain consistent iteration order"""
        tree = mcts_py.TinyNodeTree(1000)
        tree.init_root(0x12345)

        # Add children in order
        for i in range(5):
            tree.add_child(0, i, 0.2, 0x1000 + i)

        # Get children multiple times
        children1 = tree.get_children(0)
        children2 = tree.get_children(0)

        # Order should be consistent
        assert children1 == children2

    def test_nested_expansion(self):
        """Test expanding grandchildren"""
        import numpy as np

        tree = mcts_py.TinyNodeTree(1000)
        tree.init_root(0x12345)

        # Expand root
        root_moves = np.array([0, 1, 2], dtype=np.uint16)
        root_priors = np.array([0.3, 0.4, 0.3], dtype=np.float32)
        root_hashes = np.array([0x1000, 0x1001, 0x1002], dtype=np.uint64)
        tree.expand_node(0, root_moves, root_priors, root_hashes)

        root_children = tree.get_children(0)
        assert len(root_children) == 3

        # Expand first child
        child0 = root_children[0]
        child_moves = np.array([10, 11], dtype=np.uint16)
        child_priors = np.array([0.6, 0.4], dtype=np.float32)
        child_hashes = np.array([0x2000, 0x2001], dtype=np.uint64)
        tree.expand_node(child0, child_moves, child_priors, child_hashes)

        # Verify grandchildren
        assert tree.get_child_count(child0) == 2
        grandchildren = tree.get_children(child0)
        assert len(grandchildren) == 2

        # Verify parent relationships
        for gc_idx in grandchildren:
            gc = tree.get_node(gc_idx)
            assert gc.parent_idx == child0

    def test_child_count_empty(self):
        """Test get_child_count on node with no children"""
        tree = mcts_py.TinyNodeTree(1000)
        tree.init_root(0x12345)

        # Root has no children initially
        assert tree.get_child_count(0) == 0

    def test_validate_with_children(self):
        """Test tree validation with child structure"""
        import numpy as np

        tree = mcts_py.TinyNodeTree(1000)
        tree.init_root(0x12345)

        # Build tree: root -> 3 children -> each child has 2 grandchildren
        for i in range(3):
            child_idx = tree.add_child(0, i, 0.33, 0x1000 + i)

            # Add grandchildren
            for j in range(2):
                tree.add_child(child_idx, j, 0.5, 0x2000 + i * 10 + j)

        # Should validate successfully
        assert tree.validate()

        # Verify structure
        assert tree.get_child_count(0) == 3
        for child_idx in tree.get_children(0):
            assert tree.get_child_count(child_idx) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
