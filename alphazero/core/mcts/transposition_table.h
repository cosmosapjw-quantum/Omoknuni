#pragma once

#include <unordered_map>
#include <mutex>
#include <memory>
#include <list>
#include <vector>
#include <cstdint>

namespace alphazero {

// Forward declarations
class MCTSNode;

/**
 * Thread-safe transposition table for MCTS.
 * Maps game state hashes to nodes in the search tree.
 */
class TranspositionTable {
public:
    /**
     * Constructor.
     * 
     * @param max_size Maximum number of entries in the table
     */
    TranspositionTable(size_t max_size = 1000000);
    
    /**
     * Virtual destructor for proper cleanup.
     */
    virtual ~TranspositionTable();
    
    /**
     * Look up a node in the table.
     * 
     * @param hash_value Hash value of the game state
     * @return Pointer to the node if found, nullptr otherwise
     */
    virtual MCTSNode* lookup(uint64_t hash_value);
    
    /**
     * Store a node in the table.
     * 
     * @param hash_value Hash value of the game state
     * @param node Pointer to the node to store
     */
    virtual void store(uint64_t hash_value, MCTSNode* node);
    
    /**
     * Clear the table.
     */
    virtual void clear();
    
    /**
     * Get the number of entries in the table.
     * 
     * @return Number of entries
     */
    virtual size_t size() const;
    
    /**
     * Check if a hash value is in the table.
     * 
     * @param hash_value Hash value to check
     * @return true if the hash value is in the table, false otherwise
     */
    virtual bool contains(uint64_t hash_value) const;
    
protected:
    // Maximum size of the table
    size_t max_size_;
    
    // Hash table
    std::unordered_map<uint64_t, MCTSNode*> table_;
    
    // Mutex for thread safety
    mutable std::recursive_mutex mutex_;  // Changed from std::mutex to std::recursive_mutex
};

/**
 * LRU (Least Recently Used) transposition table.
 * Evicts the least recently used entries when the table is full.
 */
class LRUTranspositionTable : public TranspositionTable {
public:
    /**
     * Constructor.
     * 
     * @param max_size Maximum number of entries in the table
     */
    LRUTranspositionTable(size_t max_size = 1000000);
    
    /**
     * Look up a node in the table and update its access time.
     * 
     * @param hash_value Hash value of the game state
     * @return Pointer to the node if found, nullptr otherwise
     */
    MCTSNode* lookup(uint64_t hash_value) override;
    
    /**
     * Store a node in the table, evicting the least recently used entry if necessary.
     * 
     * @param hash_value Hash value of the game state
     * @param node Pointer to the node to store
     */
    void store(uint64_t hash_value, MCTSNode* node) override;
    
    /**
     * Clear the table and LRU list.
     */
    void clear() override;
    
private:
    // LRU list (most recently used at the front)
    std::list<uint64_t> lru_list_;
    
    // Map from hash to position in LRU list
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> lru_map_;
};

} // namespace alphazero