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
    
    // If table is full, don't add new entries
    if (table_.size() >= max_size_) {
        return;
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
        // Move the hash to the front of the LRU list
        auto lru_it = lru_map_[hash_value];
        lru_list_.erase(lru_it);
        lru_list_.push_front(hash_value);
        lru_map_[hash_value] = lru_list_.begin();
        
        return it->second;
    }
    
    return nullptr;
}

void LRUTranspositionTable::store(uint64_t hash_value, MCTSNode* node) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // If hash already exists, update it and move to front of LRU list
    auto it = table_.find(hash_value);
    if (it != table_.end()) {
        it->second = node;
        
        auto lru_it = lru_map_[hash_value];
        lru_list_.erase(lru_it);
        lru_list_.push_front(hash_value);
        lru_map_[hash_value] = lru_list_.begin();
        
        return;
    }
    
    // If table is full, remove the least recently used entry
    if (table_.size() >= max_size_) {
        uint64_t lru_hash = lru_list_.back();
        table_.erase(lru_hash);
        lru_map_.erase(lru_hash);
        lru_list_.pop_back();
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