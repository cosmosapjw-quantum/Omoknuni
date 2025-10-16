// tiny_node_tree.hpp - Tree storage using TinyNode AoS layout with bump allocator
// Part of T024f-1: TinyNode Storage Layer

#pragma once

#include "tiny_node.hpp"
#include <cstdint>
#include <atomic>
#include <mutex>
#include <vector>
#include <memory>
#include <cassert>

namespace mcts {

/**
 * @brief MCTS tree using TinyNode array-of-structs layout
 *
 * This class implements zero-copy MCTS tree storage where:
 * - Nodes are 64-byte aligned structs (TinyNode)
 * - O(1) bump allocation for fast node creation
 * - Free list for node reuse
 * - Thread-safe allocation via atomics
 * - No state cloning (stores only moves + statistics)
 *
 * Memory layout:
 * - Single contiguous array of TinyNode structs
 * - Each node: 64 bytes (34 bytes data + 30 bytes padding)
 * - 10M nodes = 640 MB (vs 1.2 GB with state cloning)
 */
class TinyNodeTree {
public:
    /**
     * @brief Initialize tree with specified capacity
     *
     * @param max_nodes Maximum number of nodes to support
     */
    explicit TinyNodeTree(std::size_t max_nodes = 50'000'000);

    /**
     * @brief Destructor - frees aligned memory
     */
    ~TinyNodeTree();

    // Disable copy/move to avoid complexity
    TinyNodeTree(const TinyNodeTree&) = delete;
    TinyNodeTree& operator=(const TinyNodeTree&) = delete;
    TinyNodeTree(TinyNodeTree&&) = delete;
    TinyNodeTree& operator=(TinyNodeTree&&) = delete;

    /**
     * @brief Allocate a single node from the pool (O(1) bump allocation)
     *
     * Thread-safe: Uses atomic increment for next_index_
     * Fast path: Bump allocator (no lock)
     * Slow path: Free list (with lock)
     *
     * @return Index of allocated node, or -1 if pool is full
     */
    int32_t allocate_node();

    /**
     * @brief Deallocate a single node back to the pool
     *
     * Note: Node is added to free list for reuse.
     * Node data is NOT cleared (will be overwritten on reuse).
     *
     * @param index Index of node to deallocate
     */
    void deallocate_node(int32_t index);

    /**
     * @brief Clear all nodes and reset tree to empty state
     *
     * O(1) operation: Just resets allocation index and clears free list.
     * Memory is NOT zeroed (nodes will be initialized on allocation).
     */
    void clear();

    /**
     * @brief Get pointer to node by index
     *
     * WARNING: Pointer may be invalidated by allocate_node() if reallocation occurs.
     * Use node index as the canonical reference, not pointers.
     *
     * @param index Node index (0 to node_count_ - 1)
     * @return Pointer to TinyNode, or nullptr if index invalid
     */
    TinyNode* get_node(int32_t index);

    /**
     * @brief Get const pointer to node by index
     */
    const TinyNode* get_node(int32_t index) const;

    /**
     * @brief Check if node index is valid
     */
    bool is_valid_index(int32_t index) const {
        return index >= 0 && static_cast<std::size_t>(index) < next_index_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Get current number of allocated nodes
     */
    std::size_t get_node_count() const {
        return next_index_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Get maximum capacity
     */
    std::size_t get_max_nodes() const {
        return max_nodes_;
    }

    /**
     * @brief Get memory usage in bytes
     */
    std::size_t get_memory_usage() const {
        return max_nodes_ * sizeof(TinyNode);
    }

    /**
     * @brief Get bytes per node (actual footprint)
     */
    double get_bytes_per_node() const {
        return sizeof(TinyNode);  // Always 64 bytes (aligned)
    }

    /**
     * @brief Get root node index (always 0 if tree has nodes)
     */
    int32_t get_root_index() const {
        return get_node_count() > 0 ? 0 : -1;
    }

    /**
     * @brief Check if tree has space for additional nodes
     *
     * @param count Number of nodes to check for
     * @return true if space available
     */
    bool has_space_for(std::size_t count) const {
        std::size_t current = next_index_.load(std::memory_order_relaxed);
        std::size_t available_bump = (current < max_nodes_) ? (max_nodes_ - current) : 0;

        std::lock_guard<std::mutex> lock(free_list_mutex_);
        std::size_t available_free = free_list_.size();

        return (available_bump + available_free) >= count;
    }

    /**
     * @brief Initialize root node (called once at start of search)
     *
     * @param zobrist_hash Initial zobrist hash for root position
     * @return Index of root node (always 0)
     */
    int32_t init_root(uint64_t zobrist_hash);

    /**
     * @brief Validate tree structure and constraints
     *
     * @return true if tree is valid
     */
    bool validate() const;

private:
    // Maximum capacity
    std::size_t max_nodes_;

    // Next index for bump allocation (atomic for thread-safety)
    std::atomic<std::size_t> next_index_{0};

    // Free list for node reuse
    std::vector<int32_t> free_list_;
    mutable std::mutex free_list_mutex_;

    // Node storage (64-byte aligned array)
    TinyNode* nodes_;

    /**
     * @brief Allocate aligned memory for node array
     */
    void allocate_array();

    /**
     * @brief Free aligned memory
     */
    void deallocate_array();

    /**
     * @brief Initialize a single node to default state
     */
    void init_node(int32_t index);
};

} // namespace mcts
