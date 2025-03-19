#include "transposition_table.h"
#include "mcts_node.h"

namespace alphazero {

TranspositionTable::TranspositionTable(size_t max_size)
    : max_size_(max_size) {
}

TranspositionTable::~TranspositionTable() {
    // Note: We don't delete the nodes since they are owned by the MCTS tree
}

MCTSNode* TranspositionTable::lookup(uint64_t hash_value) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = table_.find(hash_value);
    if (it != table_.end()) {
        return it->second;
    }
    
    return nullptr;
}

void TranspositionTable::store(uint64_t hash_value, MCTSNode* node) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // If table is full, randomly evict an existing entry
    if (table_.size() >= max_size_) {
        // Find a random entry to evict
        auto it = table_.begin();
        std::advance(it, rand() % table_.size());
        table_.erase(it);
    }
    
    table_[hash_value] = node;
}

void TranspositionTable::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    table_.clear();
}

size_t TranspositionTable::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return table_.size();
}

bool TranspositionTable::contains(uint64_t hash_value) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return table_.find(hash_value) != table_.end();
}

// LRU Transposition Table implementation

LRUTranspositionTable::LRUTranspositionTable(size_t max_size)
    : TranspositionTable(max_size) {
}

MCTSNode* LRUTranspositionTable::lookup(uint64_t hash_value) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = table_.find(hash_value);
    if (it != table_.end()) {
        // Make sure the node still exists and hasn't been deallocated
        if (it->second != nullptr) {
            // Check if hash exists in lru_map before trying to access it
            auto lru_map_it = lru_map_.find(hash_value);
            if (lru_map_it != lru_map_.end()) {
                // Move the hash to the front of the LRU list
                auto lru_it = lru_map_it->second;
                lru_list_.erase(lru_it);
                lru_list_.push_front(hash_value);
                lru_map_[hash_value] = lru_list_.begin();
            } else {
                // If the hash is in table_ but not in lru_map_, add it
                lru_list_.push_front(hash_value);
                lru_map_[hash_value] = lru_list_.begin();
            }
            
            return it->second;
        } else {
            // If the node is null, remove the entry from the table
            auto lru_map_it = lru_map_.find(hash_value);
            if (lru_map_it != lru_map_.end()) {
                lru_list_.erase(lru_map_it->second);
                lru_map_.erase(lru_map_it);
            }
            table_.erase(it);
            return nullptr;
        }
    }
    
    return nullptr;
}

void LRUTranspositionTable::store(uint64_t hash_value, MCTSNode* node) {
    if (node == nullptr) {
        return; // Don't store null nodes
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // If hash already exists, update it and move to front of LRU list
    auto it = table_.find(hash_value);
    if (it != table_.end()) {
        // Only update if the new node has more information (more visits)
        if (node->visit_count.load() > it->second->visit_count.load()) {
            it->second = node;
        }
        
        // Always move to front of LRU list regardless of whether we updated the node
        auto lru_map_it = lru_map_.find(hash_value);
        if (lru_map_it != lru_map_.end()) {
            auto lru_it = lru_map_it->second;
            lru_list_.erase(lru_it);
            lru_list_.push_front(hash_value);
            lru_map_[hash_value] = lru_list_.begin();
        } else {
            // If somehow the hash is in table_ but not in lru_map_, add it
            lru_list_.push_front(hash_value);
            lru_map_[hash_value] = lru_list_.begin();
        }
        
        return;
    }
    
    // If table is full, remove the least recently used entry
    if (table_.size() >= max_size_) {
        // Make sure the list is not empty
        if (!lru_list_.empty()) {
            uint64_t lru_hash = lru_list_.back();
            table_.erase(lru_hash);
            lru_map_.erase(lru_hash);
            lru_list_.pop_back();
        } else {
            // If the list is empty but the table is full, something is wrong
            // Let's clear everything to be safe
            table_.clear();
            lru_map_.clear();
            lru_list_.clear();
        }
    }
    
    // Add the new entry
    table_[hash_value] = node;
    lru_list_.push_front(hash_value);
    lru_map_[hash_value] = lru_list_.begin();
}

void LRUTranspositionTable::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    table_.clear();
    lru_list_.clear();
    lru_map_.clear();
}

} // namespace alphazero