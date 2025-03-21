// Create file: alphazero/core/mcts/node_pool.h

#pragma once

#include <vector>
#include <mutex>
#include <memory>
#include "mcts_node.h"

namespace alphazero {

/**
 * Memory pool for efficient MCTSNode allocation.
 */
class NodePool {
public:
    NodePool(size_t initial_size = 1024) {
        // Allocate initial block
        allocate_block(initial_size);
    }
    
    /**
     * Allocate a node from the pool.
     */
    MCTSNode* allocate(float prior = 0.0f, MCTSNode* parent = nullptr, int move = -1) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        if (free_nodes_.empty()) {
            // Double the previous block size for exponential growth
            size_t new_block_size = blocks_.empty() ? 1024 : block_sizes_.back() * 2;
            allocate_block(new_block_size);
        }
        
        // Get a node from the free list
        MCTSNode* node = free_nodes_.back();
        free_nodes_.pop_back();
        
        // Initialize the node with placement new
        new (node) MCTSNode(prior, parent, move);
        
        return node;
    }
    
    /**
     * Return a node to the pool.
     */
    void deallocate(MCTSNode* node) {
        if (!node) return;
        
        // Call destructor
        node->~MCTSNode();
        
        // Return to free list
        std::lock_guard<std::mutex> lock(pool_mutex_);
        free_nodes_.push_back(node);
    }
    
    /**
     * Clear the pool and release all memory.
     */
    void clear() {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        blocks_.clear();
        block_sizes_.clear();
        free_nodes_.clear();
    }
    
private:
    // Allocate a new block of nodes
    void allocate_block(size_t size) {
        // Create a new block
        blocks_.emplace_back(new MCTSNode[size]);
        block_sizes_.push_back(size);
        
        // Add nodes to free list in reverse order (LIFO)
        for (size_t i = 0; i < size; ++i) {
            free_nodes_.push_back(&blocks_.back()[i]);
        }
    }
    
    // Storage for node blocks
    std::vector<std::unique_ptr<MCTSNode[]>> blocks_;
    std::vector<size_t> block_sizes_;
    
    // Free nodes list
    std::vector<MCTSNode*> free_nodes_;
    
    // Mutex for thread safety
    std::mutex pool_mutex_;
};

} // namespace alphazero