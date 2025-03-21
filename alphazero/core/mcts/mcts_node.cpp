#include "mcts_node.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>

namespace alphazero {

MCTSNode::MCTSNode(float prior, MCTSNode* parent, int move)
    : visit_count(0),
      value_sum(0.0f),
      prior(prior),
      parent(parent),
      move(move),
      virtual_loss(0) {
}

MCTSNode::~MCTSNode() {
    // Standard destructor (smart pointers will handle children cleanup)
}

bool MCTSNode::is_expanded() const {
    std::lock_guard<std::recursive_mutex> lock(children_mutex);
    return !children.empty();
}

float MCTSNode::value() const {
    int visits = visit_count.load();
    if (visits == 0) {
        return 0.0f;
    }
    
    std::lock_guard<std::recursive_mutex> lock(value_mutex);
    return value_sum / static_cast<float>(visits);
}

void MCTSNode::add_virtual_loss(int amount) {
    virtual_loss.fetch_add(amount);
}

void MCTSNode::remove_virtual_loss(int amount) {
    // Check to avoid underflow
    int current = virtual_loss.load();
    if (current >= amount) {
        virtual_loss.fetch_sub(amount);
    } else {
        virtual_loss.store(0);
    }
}

float MCTSNode::ucb_score(int parent_visit_count, float c_puct) const {
    // Get the visit count including virtual loss
    int visits = visit_count.load();
    int effective_visits = visits + virtual_loss.load();
    
    // Edge case: if no visits, return infinity to ensure exploration
    if (effective_visits == 0) {
        return std::numeric_limits<float>::max();
    }
    
    // Calculate exploitation term (Q-value)
    float exploitation;
    {
        std::lock_guard<std::recursive_mutex> lock(value_mutex);
        exploitation = value_sum / static_cast<float>(effective_visits);
    }
    
    // Calculate exploration term (U-value)
    float exploration = c_puct * prior * std::sqrt(static_cast<float>(parent_visit_count)) / 
                       (1.0f + static_cast<float>(effective_visits));
    
    // Return the combined score
    return exploitation + exploration;
}

std::pair<int, MCTSNode*> MCTSNode::select_child(float c_puct) {
    std::lock_guard<std::recursive_mutex> lock(children_mutex);
    
    if (children.empty()) {
        return {-1, nullptr};
    }
    
    float best_score = -std::numeric_limits<float>::max();
    std::vector<std::pair<int, MCTSNode*>> best_children;
    
    for (const auto& child_pair : children) {
        int move = child_pair.first;
        MCTSNode* child = child_pair.second.get();
        
        // Skip null children
        if (!child) continue;
        
        float score = child->ucb_score(visit_count.load(), c_puct);
        
        if (score > best_score) {
            best_score = score;
            best_children.clear();
            best_children.emplace_back(move, child);
        } else if (std::abs(score - best_score) < 1e-6) {
            // If scores are very close, consider them tied
            best_children.emplace_back(move, child);
        }
    }
    
    // If there are multiple best children, select randomly
    if (best_children.size() > 1) {
        int index = std::rand() % best_children.size();
        return best_children[index];
    } else if (!best_children.empty()) {
        return best_children[0];
    }
    
    // Should never reach here if children is not empty
    return {-1, nullptr};
}

void MCTSNode::expand(const std::vector<int>& moves, const std::vector<float>& priors) {
    std::lock_guard<std::recursive_mutex> lock(children_mutex);
    
    // If already expanded, don't re-expand
    if (!children.empty()) {
        return;
    }
    
    // Already expanded
    if (!children.empty()) {
        return;
    }
    
    // Create child nodes for each move
    for (size_t i = 0; i < moves.size(); ++i) {
        int move = moves[i];
        float move_prior = (i < priors.size()) ? priors[i] : 0.0f;
        
        auto child = std::make_unique<MCTSNode>(move_prior, this, move);
        children[move] = std::move(child);
    }
}

void MCTSNode::backup(float value) {
<<<<<<< HEAD
    // Use an iterative approach instead of recursion to avoid deadlocks
=======
>>>>>>> 42bb511ab1410a992c3fb9fc8a11235d555aea77
    MCTSNode* current = this;
    float current_value = value;
    
    while (current != nullptr) {
<<<<<<< HEAD
        // Update visit count atomically
=======
>>>>>>> 42bb511ab1410a992c3fb9fc8a11235d555aea77
        current->visit_count.fetch_add(1);
        
        {
<<<<<<< HEAD
            std::lock_guard<std::recursive_mutex> lock(current->value_mutex);
            current->value_sum += current_value;
            
            // Remove any virtual loss that was applied during selection
            if (current->virtual_loss.load() > 0) {
                current->virtual_loss.fetch_sub(1);
            }
=======
            std::lock_guard<std::mutex> lock(current->value_mutex);
            current->value_sum += current_value;
            
            // Reset virtual loss to 0 instead of decrementing
            current->virtual_loss.store(0);
>>>>>>> 42bb511ab1410a992c3fb9fc8a11235d555aea77
        }
        
        // Store parent in a local variable before potentially modifying 'current'
        MCTSNode* parent = current->parent;
        
        // Negate the value for alternating players
        current_value = -current_value;
        
        // Move to parent
        current = parent;
    }
}

MCTSNode* MCTSNode::get_child(int move) {
    std::lock_guard<std::recursive_mutex> lock(children_mutex);
    
    auto it = children.find(move);
    if (it != children.end()) {
        return it->second.get();
    }
    
    return nullptr;
}

std::unordered_map<int, int> MCTSNode::get_visit_counts() const {
    std::lock_guard<std::recursive_mutex> lock(children_mutex);
    
    std::unordered_map<int, int> visit_counts;
    for (const auto& child_pair : children) {
        visit_counts[child_pair.first] = child_pair.second->visit_count.load();
    }
    
    return visit_counts;
}

} // namespace alphazero