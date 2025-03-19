#include "mcts_node.h"
#include <algorithm>
#include <limits>
#include <cmath>

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
    // Children are automatically cleaned up by unique_ptr
}

bool MCTSNode::is_expanded() const {
    std::lock_guard<std::mutex> lock(children_mutex);
    return !children.empty();
}

float MCTSNode::value() const {
    int visits = visit_count.load();
    if (visits == 0) {
        return 0.0f;
    }
    
    std::lock_guard<std::mutex> lock(value_mutex);
    return value_sum / static_cast<float>(visits);
}

void MCTSNode::add_virtual_loss(int amount) {
    virtual_loss.fetch_add(amount);
}

void MCTSNode::remove_virtual_loss(int amount) {
    virtual_loss.fetch_sub(amount);
}

float MCTSNode::ucb_score(int parent_visit_count, float c_puct) const {
    // Effective visit count (including virtual loss)
    int effective_visits = visit_count.load() + virtual_loss.load();
    
    if (effective_visits == 0) {
        // If no visits yet, use a high value to encourage exploration
        return std::numeric_limits<float>::max();
    }
    
    // Exploitation term: Q(s,a)
    float exploitation;
    {
        std::lock_guard<std::mutex> lock(value_mutex);
        exploitation = value_sum / static_cast<float>(effective_visits);
    }
    
    // Exploration term: c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    float exploration = c_puct * prior * std::sqrt(static_cast<float>(parent_visit_count)) / 
                        (1.0f + static_cast<float>(effective_visits));
    
    return exploitation + exploration;
}

std::pair<int, MCTSNode*> MCTSNode::select_child(float c_puct) {
    std::lock_guard<std::mutex> lock(children_mutex);
    
    if (children.empty()) {
        return {-1, nullptr};
    }
    
    // Find moves with the highest UCB score
    float best_score = -std::numeric_limits<float>::max();
    std::vector<std::pair<int, MCTSNode*>> best_children;
    
    for (const auto& child_pair : children) {
        int move = child_pair.first;
        MCTSNode* child = child_pair.second.get();
        
        float score = child->ucb_score(visit_count.load(), c_puct);
        
        if (score > best_score) {
            best_score = score;
            best_children.clear();
            best_children.emplace_back(move, child);
        } else if (std::abs(score - best_score) < 1e-6) {
            // If scores are approximately equal, consider this child as well
            best_children.emplace_back(move, child);
        }
    }
    
    // If multiple children have the same score, select one randomly
    if (best_children.size() > 1) {
        int index = std::rand() % best_children.size();
        return best_children[index];
    } else if (!best_children.empty()) {
        return best_children[0];
    }
    
    // This should never happen if children is not empty
    return {-1, nullptr};
}

void MCTSNode::expand(const std::vector<int>& moves, const std::vector<float>& priors) {
    std::lock_guard<std::mutex> lock(children_mutex);
    
    // Create children for each move
    for (size_t i = 0; i < moves.size(); ++i) {
        int move = moves[i];
        float move_prior = (i < priors.size()) ? priors[i] : 0.0f;
        
        auto child = std::make_unique<MCTSNode>(move_prior, this, move);
        children[move] = std::move(child);
    }
}

void MCTSNode::backup(float value) {
    // Use an iterative approach instead of recursion to avoid deadlocks
    MCTSNode* current = this;
    float current_value = value;
    
    while (current != nullptr) {
        // Update this node
        current->visit_count.fetch_add(1);
        {
            std::lock_guard<std::mutex> lock(current->value_mutex);
            current->value_sum += current_value;
        }
        
        // Remove any virtual loss that was applied during selection
        if (current->virtual_loss.load() > 0) {
            current->remove_virtual_loss();
        }
        
        // Move to parent with negated value (for alternating players)
        MCTSNode* parent = current->parent;
        current = parent;
        current_value = -current_value; // Negate for alternating players
    }
}

MCTSNode* MCTSNode::get_child(int move) {
    std::lock_guard<std::mutex> lock(children_mutex);
    
    auto it = children.find(move);
    if (it != children.end()) {
        return it->second.get();
    }
    
    return nullptr;
}

std::unordered_map<int, int> MCTSNode::get_visit_counts() const {
    std::lock_guard<std::mutex> lock(children_mutex);
    
    std::unordered_map<int, int> visit_counts;
    for (const auto& child_pair : children) {
        visit_counts[child_pair.first] = child_pair.second->visit_count.load();
    }
    
    return visit_counts;
}

} // namespace alphazero