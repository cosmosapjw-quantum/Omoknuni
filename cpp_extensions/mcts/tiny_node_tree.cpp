// tiny_node_tree.cpp - Implementation of TinyNodeTree
// Part of T024f-1: TinyNode Storage Layer

#include "tiny_node_tree.hpp"
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#endif

namespace mcts {

TinyNodeTree::TinyNodeTree(std::size_t max_nodes)
    : max_nodes_(max_nodes), nodes_(nullptr) {
    if (max_nodes_ == 0) {
        throw std::invalid_argument("TinyNodeTree: max_nodes must be > 0");
    }

    allocate_array();
}

TinyNodeTree::~TinyNodeTree() {
    deallocate_array();
}

void TinyNodeTree::allocate_array() {
    // Allocate 64-byte aligned memory for node array
    std::size_t size = max_nodes_ * sizeof(TinyNode);

#ifdef _WIN32
    nodes_ = static_cast<TinyNode*>(_aligned_malloc(size, 64));
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, size) != 0) {
        ptr = nullptr;
    }
    nodes_ = static_cast<TinyNode*>(ptr);
#endif

    if (!nodes_) {
        throw std::bad_alloc();
    }

    // Zero-initialize the array
    std::memset(nodes_, 0, size);
}

void TinyNodeTree::deallocate_array() {
    if (nodes_) {
#ifdef _WIN32
        _aligned_free(nodes_);
#else
        free(nodes_);
#endif
        nodes_ = nullptr;
    }
}

int32_t TinyNodeTree::allocate_node() {
    // Fast path: Try free list first (with lock)
    {
        std::lock_guard<std::mutex> lock(free_list_mutex_);
        if (!free_list_.empty()) {
            int32_t index = free_list_.back();
            free_list_.pop_back();
            init_node(index);
            return index;
        }
    }

    // Slow path: Bump allocation (lock-free)
    std::size_t index = next_index_.fetch_add(1, std::memory_order_relaxed);

    if (index >= max_nodes_) {
        // Pool exhausted - rollback
        next_index_.fetch_sub(1, std::memory_order_relaxed);
        return -1;
    }

    init_node(static_cast<int32_t>(index));
    return static_cast<int32_t>(index);
}

void TinyNodeTree::deallocate_node(int32_t index) {
    if (!is_valid_index(index)) {
        return;  // Silently ignore invalid indices
    }

    // Add to free list for reuse
    std::lock_guard<std::mutex> lock(free_list_mutex_);
    free_list_.push_back(index);
}

void TinyNodeTree::clear() {
    // O(1) reset: just reset bump allocator and clear free list
    next_index_.store(0, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lock(free_list_mutex_);
    free_list_.clear();

    // Note: We do NOT zero the memory - nodes will be re-initialized on allocation
}

TinyNode* TinyNodeTree::get_node(int32_t index) {
    if (!is_valid_index(index)) {
        return nullptr;
    }
    return &nodes_[index];
}

const TinyNode* TinyNodeTree::get_node(int32_t index) const {
    if (!is_valid_index(index)) {
        return nullptr;
    }
    return &nodes_[index];
}

int32_t TinyNodeTree::init_root(uint64_t zobrist_hash) {
    // Clear tree first
    clear();

    // Allocate root node
    int32_t root_idx = allocate_node();
    if (root_idx != 0) {
        throw std::runtime_error("TinyNodeTree::init_root: root index != 0");
    }

    TinyNode* root = get_node(root_idx);
    root->move = 0;  // Root has no move
    root->parent_idx = 0;  // Root points to self
    root->zobrist_hash = zobrist_hash;
    root->flags |= TinyNode::FLAG_ROOT;

    // Root starts with 1 visit (per WU-UCT)
    root->visit_count.store(1, std::memory_order_relaxed);

    return root_idx;
}

void TinyNodeTree::init_node(int32_t index) {
    TinyNode* node = &nodes_[index];

    // Zero-initialize critical fields
    node->move = 0;
    node->parent_idx = 0;
    node->first_child_idx = 0;
    node->next_sibling_idx = 0;
    node->visit_count.store(0, std::memory_order_relaxed);
    node->total_value_scaled.store(0, std::memory_order_relaxed);
    node->prior_scaled = 0;
    node->virtual_loss.store(0, std::memory_order_relaxed);
    node->flags = 0;
    node->zobrist_hash = 0;
}

bool TinyNodeTree::validate() const {
    std::size_t count = get_node_count();

    // Check all allocated nodes
    for (std::size_t i = 0; i < count; ++i) {
        const TinyNode* node = get_node(static_cast<int32_t>(i));
        if (!node) {
            return false;  // Invalid node pointer
        }

        // Validate parent index
        if (i != 0) {  // Non-root
            if (node->parent_idx == 0 && !(node->flags & TinyNode::FLAG_ROOT)) {
                // Only root should have parent_idx = 0 (self-reference)
                // Allow for now - parent will be set during expansion
            }
            if (node->parent_idx >= static_cast<uint32_t>(count)) {
                return false;  // Parent index out of bounds
            }
        } else {  // Root
            if (!(node->flags & TinyNode::FLAG_ROOT)) {
                return false;  // Root must have FLAG_ROOT set
            }
            if (node->parent_idx != 0) {
                return false;  // Root must point to self
            }
        }

        // Validate child indices
        if (node->first_child_idx != 0) {
            if (node->first_child_idx >= static_cast<uint32_t>(count)) {
                return false;  // First child out of bounds
            }
        }

        if (node->next_sibling_idx != 0) {
            if (node->next_sibling_idx >= static_cast<uint32_t>(count)) {
                return false;  // Next sibling out of bounds
            }
        }
    }

    return true;
}

} // namespace mcts
