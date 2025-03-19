#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <mutex>
#include <cmath>

namespace alphazero {

// Forward declarations
class MCTSNode;

/**
 * Represents a node in the Monte Carlo Search Tree.
 * Thread-safe implementation for parallel MCTS with virtual loss.
 */
class MCTSNode {
public:
    // Constructor
    MCTSNode(float prior = 0.0f, MCTSNode* parent = nullptr, int move = -1);
    
    // Destructor
    ~MCTSNode();
    
    // Node statistics
    std::atomic<int> visit_count;
    float value_sum;  // Changed from std::atomic<float> to regular float with mutex protection
    mutable std::mutex value_mutex;  // Add a mutex to protect value_sum
    float prior;
    MCTSNode* parent;
    int move;
    
    // Children nodes
    std::unordered_map<int, std::unique_ptr<MCTSNode>> children;
    mutable std::mutex children_mutex;
    
    // Virtual loss for parallelization
    std::atomic<int> virtual_loss;
    
    // Check if node has been expanded
    bool is_expanded() const;
    
    // Get the mean value of this node
    float value() const;
    
    // Add virtual loss to this node
    void add_virtual_loss(int amount = 1);
    
    // Remove virtual loss from this node
    void remove_virtual_loss(int amount = 1);
    
    // Calculate UCB score
    float ucb_score(int parent_visit_count, float c_puct) const;
    
    // Select the best child according to UCB
    std::pair<int, MCTSNode*> select_child(float c_puct);
    
    // Expand the node with new children
    void expand(const std::vector<int>& moves, const std::vector<float>& priors);
    
    // Update node value and visit count (backpropagation)
    void backup(float value);
    
    // Get the child with the given move
    MCTSNode* get_child(int move);
    
    // Get visit counts for all children
    std::unordered_map<int, int> get_visit_counts() const;
};

} // namespace alphazero