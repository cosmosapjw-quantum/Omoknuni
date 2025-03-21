#include "mcts.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <chrono>
#include <iostream>
#include <unordered_set>
#include <cstring>
#include <atomic>
#include <map>

namespace alphazero {

MCTS::MCTS(int num_simulations,
           float c_puct,
           float dirichlet_alpha,
           float dirichlet_noise_weight,
           float virtual_loss_weight,
           bool use_transposition_table,
           size_t transposition_table_size,
           int num_threads)
    : num_simulations_(num_simulations),
      c_puct_(c_puct),
      dirichlet_alpha_(dirichlet_alpha),
      dirichlet_noise_weight_(dirichlet_noise_weight),
      virtual_loss_weight_(virtual_loss_weight),
      temperature_(1.0f),
      use_transposition_table_(use_transposition_table),
      num_threads_(num_threads) {
    
    // Create root node
    root_ = std::make_unique<MCTSNode>();
    
    // Create transposition table if needed
    if (use_transposition_table_) {
        transposition_table_ = std::make_unique<LRUTranspositionTable>(transposition_table_size);
    }
    
    // Initialize Zobrist hash - assume 15x15 board size for Gomoku as default
    // This will be adjusted based on actual board sizes during search
    zobrist_hash_ = std::make_unique<ZobristHash>(15, 2);
    
    // Create thread pool if using multiple threads
    if (num_threads_ > 1) {
        thread_pool_ = std::make_unique<ThreadPool>(num_threads_);
    }
    
    // Initialize random number generator with a time-based seed
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    rng_.seed(seed);
}

void MCTS::set_num_simulations(int num_simulations) {
    num_simulations_ = num_simulations;
}

void MCTS::set_c_puct(float c_puct) {
    c_puct_ = c_puct;
}

void MCTS::set_dirichlet_noise(float alpha, float weight) {
    dirichlet_alpha_ = alpha;
    dirichlet_noise_weight_ = weight;
}

void MCTS::set_virtual_loss_weight(float weight) {
    virtual_loss_weight_ = weight;
}

void MCTS::set_temperature(float temperature) {
    temperature_ = temperature;
}

std::unordered_map<int, float> MCTS::calculate_raw_probabilities(float temperature) const {
    std::unordered_map<int, float> probabilities;
    std::unordered_map<int, int> visits = root_->get_visit_counts();
    
    if (visits.empty()) {
        return probabilities;
    }
    
    // Special case for very small temperature (almost 0)
    if (temperature < 0.01f) {
        // Find the move with the most visits
        auto max_it = std::max_element(visits.begin(), visits.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        
        // Set probability 1.0 for the most visited move, 0.0 for others
        for (const auto& [move, count] : visits) {
            probabilities[move] = (move == max_it->first) ? 1.0f : 0.0f;
        }
    } else {
        // Apply temperature
        float sum = 0.0f;
        try {
            // Calculate temperature-adjusted visit counts
            for (const auto& [move, count] : visits) {
                float power = std::pow(static_cast<float>(count), 1.0f / temperature);
                probabilities[move] = power;
                sum += power;
            }
            
            // Normalize
            if (sum > 0.0f) {
                for (auto& [move, prob] : probabilities) {
                    prob /= sum;
                }
            }
        } catch (const std::exception& e) {
            // If we encounter numerical issues, fall back to normalizing the raw visit counts
            sum = 0.0f;
            for (const auto& [move, count] : visits) {
                probabilities[move] = static_cast<float>(count);
                sum += count;
            }
            
            if (sum > 0.0f) {
                for (auto& [move, prob] : probabilities) {
                    prob /= sum;
                }
            }
        }
    }
    
    return probabilities;
}

std::unordered_map<int, float> MCTS::search(
    const std::vector<float>& state_tensor,
    const std::vector<int>& legal_moves,
    const std::function<std::pair<std::vector<float>, float>(const std::vector<float>&)>& evaluator,
    bool progressive_widening) {
    
    // Store the legal moves for future reference
    legal_moves_ = legal_moves;  // Add this line to store legal moves
    
    // Make sure Zobrist hash is initialized with the correct board size
    if (zobrist_hash_ && state_tensor.size() > 0) {
        // Check if we need to initialize the Zobrist hash with correct board size
        int board_size = static_cast<int>(std::sqrt(state_tensor.size()));
        // Only consider the actual board elements, not any attached metadata
        size_t board_elements = board_size * board_size;
        if (board_elements <= state_tensor.size() && 
            board_size != zobrist_hash_->get_board_size()) {
            // Reinitialize with the new board size
            zobrist_hash_ = std::make_unique<ZobristHash>(board_size, 2);
        }
    }
    
    // If root is not expanded, evaluate and expand it
    if (!root_->is_expanded()) {
        // Root evaluation - always done sequentially
        auto eval_result = evaluator(state_tensor);
        auto& policy = eval_result.first;
        
        // IMPORTANT CHANGE: Filter policy to only include legal moves
        std::vector<float> filtered_policy(policy.size(), 0.0f);
        for (int move : legal_moves) {
            if (move >= 0 && move < static_cast<int>(policy.size())) {
                filtered_policy[move] = policy[move];
            }
        }
        
        // Normalize filtered policy
        float sum = 0.0f;
        for (float p : filtered_policy) {
            sum += p;
        }
        
        if (sum > 0.0f) {
            for (float& p : filtered_policy) {
                p /= sum;
            }
        } else {
            // If all probabilities are zero, use uniform distribution for legal moves
            for (int move : legal_moves) {
                if (move >= 0 && move < static_cast<int>(filtered_policy.size())) {
                    filtered_policy[move] = 1.0f / legal_moves.size();
                }
            }
        }
        
        // Apply progressive widening if needed
        if (progressive_widening && legal_moves.size() > 50) {
            // Create temporary mapping of moves to priors
            std::vector<std::pair<int, float>> move_priors;
            for (int move : legal_moves) {
                if (move >= 0 && move < static_cast<int>(filtered_policy.size())) {
                    move_priors.emplace_back(move, filtered_policy[move]);
                }
            }
            
            // Sort by prior probability
            std::sort(move_priors.begin(), move_priors.end(), 
                [](const auto& a, const auto& b) { return a.second > b.second; });
            
            // Take only the top moves - ensure we don't go out of bounds
            int width = std::min(50, static_cast<int>(std::sqrt(legal_moves.size())));
            width = std::min(width, static_cast<int>(move_priors.size())); // Safety check
            std::vector<int> top_moves;
            std::vector<float> top_priors;
            
            for (int i = 0; i < width; ++i) {
                top_moves.push_back(move_priors[i].first);
                top_priors.push_back(move_priors[i].second);
            }
            
            // Normalize priors
            float sum_priors = std::accumulate(top_priors.begin(), top_priors.end(), 0.0f);
            if (sum_priors > 0.0f) {
                for (float& prior : top_priors) {
                    prior /= sum_priors;
                }
            }
            
            // Expand root with top moves only
            root_->expand(top_moves, top_priors);
        } else {
            // IMPORTANT CHANGE: Expand root with legal moves only
            root_->expand(legal_moves, filtered_policy);
        }
        
        // Add Dirichlet noise to root children
        add_dirichlet_noise(legal_moves);
    }
    
    if (num_threads_ > 1 && thread_pool_) {
        // Improved parallel simulation approach
        std::atomic<int> simulations_completed{0};
        std::vector<std::future<void>> futures;
        
        // Use thread pool to run simulations in parallel
        for (int i = 0; i < num_threads_; ++i) {
            futures.push_back(thread_pool_->enqueue([this, &simulations_completed, &state_tensor, &evaluator]() {
                std::vector<std::pair<MCTSNode*, int>> path;
                
                // Each thread runs simulations until the total count is reached
                while (simulations_completed.fetch_add(1) < num_simulations_) {
                    this->simulate(state_tensor, evaluator, path);
                    path.clear();
                }
            }));
        }
        
        // Wait for all simulations to complete and get their results
        for (auto& future : futures) {
<<<<<<< HEAD
            future.get();
=======
            try {
                // Get the result to ensure the future completes
                future.get();
            } catch (const std::exception& e) {
                // Log the error and continue
                std::cerr << "Error in simulation: " << e.what() << std::endl;
            }
>>>>>>> 42bb511ab1410a992c3fb9fc8a11235d555aea77
        }
    } else {
        // Sequential simulations
        for (int i = 0; i < num_simulations_; ++i) {
            try {
                std::vector<std::pair<MCTSNode*, int>> path;
                simulate(state_tensor, evaluator, path);
            } catch (const std::exception& e) {
                std::cerr << "Error in simulation: " << e.what() << std::endl;
            }
        }
    }
    
    // Calculate and store probabilities - filter to legal moves only
    auto raw_probs = calculate_raw_probabilities(temperature_);
    std::unordered_map<int, float> filtered_probs;
    
    // Only include legal moves in the final probabilities
    float total = 0.0f;
    for (const auto& [move, prob] : raw_probs) {
        if (std::find(legal_moves_.begin(), legal_moves_.end(), move) != legal_moves_.end()) {
            filtered_probs[move] = prob;
            total += prob;
        }
    }
    
    // Renormalize if needed
    if (total > 0.0f) {
        for (auto& [move, prob] : filtered_probs) {
            prob /= total;
        }
    } else if (!legal_moves_.empty()) {
        // Fallback to uniform over legal moves
        for (int move : legal_moves_) {
            filtered_probs[move] = 1.0f / legal_moves_.size();
        }
    }
    
    current_probabilities_ = filtered_probs;
    return current_probabilities_;
}

int MCTS::select_move(float temperature) {
    // Use provided temperature instead of the object's temperature
    auto probs = calculate_probabilities(temperature);
    
    // IMPORTANT CHANGE: Filter probabilities to only include legal moves
    std::unordered_map<int, float> legal_probs;
    float total_prob = 0.0f;
    
    for (const auto& [move, prob] : probs) {
        if (std::find(legal_moves_.begin(), legal_moves_.end(), move) != legal_moves_.end()) {
            legal_probs[move] = prob;
            total_prob += prob;
        }
    }
    
    // Re-normalize if needed
    if (total_prob > 0.0f) {
        for (auto& [move, prob] : legal_probs) {
            prob /= total_prob;
        }
    } else if (!legal_moves_.empty()) {
        // If no legal move has probability, use uniform
        for (int move : legal_moves_) {
            legal_probs[move] = 1.0f / legal_moves_.size();
        }
    }
    
    // Check if we have any moves
    if (legal_probs.empty()) {
        return -1;
    }
    
    // Extract moves and probabilities for sampling
    std::vector<int> moves;
    std::vector<float> probabilities;
    
    for (const auto& [move, prob] : legal_probs) {
        moves.push_back(move);
        probabilities.push_back(prob);
    }
    
    // Prepare discrete distribution for sampling
    std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());
    
    // Sample a move
    int index = distribution(rng_);
    int move = moves[index];
    
    return move;
}

std::unordered_map<int, float> MCTS::get_probabilities() const {
    return current_probabilities_;
}

void MCTS::update_with_move(int move) {
    // If move exists in tree, make the child the new root
    MCTSNode* child = root_->get_child(move);
    if (child) {
        // Create a new root with the child's data
        auto new_root = std::make_unique<MCTSNode>(child->prior, nullptr, -1);
        
        // Copy the child's children to the new root
        // This is a shallow copy - the children are moved from the old tree
        {
            std::lock_guard<std::recursive_mutex> lock(child->children_mutex);
            new_root->children = std::move(child->children);
        }
        
        // Replace the old root
        root_ = std::move(new_root);
    } else {
        // If the move doesn't exist in the tree, reset the tree
        reset();
    }
    
    // Clear the transposition table to free memory
    if (use_transposition_table_ && transposition_table_) {
        transposition_table_->clear();
    }
}

void MCTS::reset() {
    // Create a new root node
    root_ = std::make_unique<MCTSNode>();
    
    // Clear the probabilities
    current_probabilities_.clear();
    
    // Clear the transposition table if used
    if (use_transposition_table_ && transposition_table_) {
        transposition_table_->clear();
    }
}

MCTSNode* MCTS::get_root() const {
    return root_.get();
}

size_t MCTS::get_tree_size() const {
    return count_nodes(root_.get());
}

float MCTS::simulate(
    const std::vector<float>& state_tensor,
    const std::function<std::pair<std::vector<float>, float>(const std::vector<float>&)>& evaluator,
    std::vector<std::pair<MCTSNode*, int>>& path) {
    
    // Start with the root node
    MCTSNode* node = root_.get();
    if (!node) {
        return 0.0f;
    }
    
    std::vector<float> current_state = state_tensor;
    int current_player = 1; // Assuming 1 is the first player
    
    path.clear();
    path.push_back({node, -1});
    
    // Keep track of nodes where we added virtual loss
    std::vector<MCTSNode*> virtual_loss_nodes;
    
    // Selection: Traverse the tree to a leaf node
    const int MAX_DEPTH = 100; // Add a maximum depth to prevent infinite loops
    int depth = 0;
    
    // Using a loop structure that better avoids holding locks across calls
    while (depth < MAX_DEPTH) {
        depth++;
        
        // First, check if node is expanded without holding long-term locks
        bool is_expanded = false;
        {
            std::lock_guard<std::recursive_mutex> lock(node->children_mutex);
            is_expanded = !node->children.empty();
        }
        
        if (!is_expanded) {
            break; // Not expanded, we'll expand it below
        }
        
        // Select the best child, carefully managing locks
        int best_move = -1;
        MCTSNode* best_child = nullptr;
        
        // Get all children once with a single lock
        std::vector<std::pair<int, MCTSNode*>> children_copy;
        {
            std::lock_guard<std::recursive_mutex> lock(node->children_mutex);
            
            // Copy the relevant child information to reduce lock duration
            for (const auto& [move, child_ptr] : node->children) {
                if (child_ptr) {
                    // IMPORTANT: Only include children corresponding to legal moves
                    // We're at an intermediate state here, so we need domain-specific logic
                    // to determine legal moves at this point - for the root we have legal_moves_
                    children_copy.emplace_back(move, child_ptr.get());
                }
            }
        }
        
        // Find best child without holding the lock
        if (!children_copy.empty()) {
            float best_score = -std::numeric_limits<float>::max();
            
            for (const auto& [move, child] : children_copy) {
                float score = child->ucb_score(node->visit_count.load(), c_puct_);
                
                if (score > best_score) {
                    best_score = score;
                    best_move = move;
                    best_child = child;
                }
            }
        }
        
        if (!best_child) {
            break; // No suitable child found
        }
        
        // Apply virtual loss - only do this after we're sure about the selected child
        if (num_threads_ > 1) {
<<<<<<< HEAD
            best_child->add_virtual_loss(virtual_loss_weight_);
=======
            child->add_virtual_loss(virtual_loss_weight_);
            virtual_loss_nodes.push_back(child);
>>>>>>> 42bb511ab1410a992c3fb9fc8a11235d555aea77
        }
        
        // Apply the move to the current state
        current_state = apply_move_to_tensor(current_state, best_move, current_player);
        current_player = 3 - current_player; // Switch player (1 -> 2, 2 -> 1)
        
        // Add node and move to the path
        path.push_back({best_child, best_move});
        
        // Update node
        node = best_child;
        
        // Check transposition table if enabled - using local variables to reduce lock duration
        if (use_transposition_table_ && transposition_table_) {
            uint64_t hash = compute_state_hash(current_state, current_player);
            
            // Create path hash set to check for cycles efficiently
            std::unordered_set<MCTSNode*> path_nodes;
            for (const auto& [path_node, _] : path) {
                path_nodes.insert(path_node);
            }
            
            // Lookup transposition - get result once, holding lock briefly
            MCTSNode* transposition_node = transposition_table_->lookup(hash);
<<<<<<< HEAD
            
            if (transposition_node && path_nodes.find(transposition_node) == path_nodes.end()) {
                node = transposition_node;
                path.back().first = node; // Replace the last node in path
=======
            if (transposition_node) {
                // Check if this node is already in our path (cycle detection)
                if (path_nodes.find(transposition_node) == path_nodes.end()) {
                    // When using a transposition node, we need to clean up virtual loss
                    // from the original node and apply it to the transposition node
                    if (num_threads_ > 1) {
                        // Remove virtual loss from the original node
                        node->remove_virtual_loss(virtual_loss_weight_);
                        // Remove from our tracking list
                        if (!virtual_loss_nodes.empty()) {
                            virtual_loss_nodes.pop_back();
                        }
                        
                        // Add virtual loss to the transposition node
                        transposition_node->add_virtual_loss(virtual_loss_weight_);
                        virtual_loss_nodes.push_back(transposition_node);
                    }
                    
                    node = transposition_node;
                    path.back().first = node; // Replace the last node in path
                }
>>>>>>> 42bb511ab1410a992c3fb9fc8a11235d555aea77
            }
        }
    }
    
    // Evaluation: If we're not at a terminal state, evaluate and expand the node
    float value = 0.0f;
    
    // Terminal state detection logic would go here
    bool is_terminal = false; // Proper implementation would check game state
    
    if (!is_terminal) {
        // Evaluate the current state - do this outside of any locks
        std::pair<std::vector<float>, float> eval_result;
        try {
            eval_result = evaluator(current_state);
        }
        catch (const std::exception& e) {
            std::cerr << "Error during evaluation: " << e.what() << std::endl;
            return 0.0f;
        }
        
        auto& [policy, state_value] = eval_result;
        
        // IMPORTANT: Determine legal moves at this state
        // Here we filter potential legal moves based on positive policy values
        // In a real implementation, you would track legal moves through state transitions
        std::vector<int> legal_moves_at_state;
        std::vector<float> filtered_policy;
        
        for (size_t i = 0; i < policy.size(); ++i) {
            if (policy[i] > 0.0f) {
                legal_moves_at_state.push_back(i);
                filtered_policy.push_back(policy[i]);
            }
        }
        
        // Normalize the filtered policy
        float sum_policy = std::accumulate(filtered_policy.begin(), filtered_policy.end(), 0.0f);
        if (sum_policy > 0.0f) {
            for (float& p : filtered_policy) {
                p /= sum_policy;
            }
        } else if (!legal_moves_at_state.empty()) {
            // If all zero, use uniform
            float uniform_prob = 1.0f / legal_moves_at_state.size();
            filtered_policy.resize(legal_moves_at_state.size(), uniform_prob);
        }
        
        // Expand the node - with careful lock management
        bool already_expanded = false;
        {
            std::lock_guard<std::recursive_mutex> lock(node->children_mutex);
            already_expanded = !node->children.empty();
            
            if (!already_expanded && !legal_moves_at_state.empty()) {
                // Only expand if not already expanded and we have legal moves
                for (size_t i = 0; i < legal_moves_at_state.size(); ++i) {
                    int move = legal_moves_at_state[i];
                    float move_prior = (i < filtered_policy.size()) ? filtered_policy[i] : 0.0f;
                    
                    node->children[move] = std::make_unique<MCTSNode>(move_prior, node, move);
                }
            }
        }
        
        // Store in transposition table if enabled - do this after expansion
        if (use_transposition_table_ && transposition_table_ && !already_expanded) {
            uint64_t hash = compute_state_hash(current_state, current_player);
            transposition_table_->store(hash, node);
        }
        
        // Use the evaluated value
        value = state_value;
    }
    
    // Backup: Update the values of all nodes in the path
<<<<<<< HEAD
    // This implementation avoids recursive calls by using a loop
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        MCTSNode* path_node = it->first;
        if (path_node) {
            // Update visit count atomically
            path_node->visit_count.fetch_add(1);
            
            // Update value sum with locking
            {
                std::lock_guard<std::recursive_mutex> lock(path_node->value_mutex);
                path_node->value_sum += value;
                
                // Remove virtual loss if applicable
                if (num_threads_ > 1 && path_node->virtual_loss.load() > 0) {
                    path_node->virtual_loss.fetch_sub(1);
                }
            }
            
            // Negate value for alternating players
            value = -value;
        }
    }
=======
    // The backup method will clear the virtual loss in each node
    node->backup(value);
>>>>>>> 42bb511ab1410a992c3fb9fc8a11235d555aea77
    
    return value;
}

<<<<<<< HEAD
uint64_t MCTS::compute_state_hash(const std::vector<float>& state_tensor, int current_player) const {
    if (state_tensor.empty()) {
        return 0;
    }
    
    // Try to extract stored hash from the state tensor first if it exists
    // Detect if the tensor has extra elements for storing the hash
    int board_size = static_cast<int>(std::sqrt(state_tensor.size()));
    if (board_size * board_size < static_cast<int>(state_tensor.size()) && 
        static_cast<int>(state_tensor.size()) >= board_size * board_size + sizeof(uint64_t) / sizeof(float)) {
        
        // Extra space might contain hash value
        uint64_t hash = 0;
        const float* ptr = state_tensor.data() + board_size * board_size;
        memcpy(&hash, ptr, sizeof(hash));
        
        // If the hash is valid (non-zero), return it directly
        if (hash != 0) {
            return hash;
        }
    }
    
    // If no stored hash or invalid hash, compute it using Zobrist if available
    if (zobrist_hash_) {
        return zobrist_hash_->compute_hash(state_tensor, current_player);
    }
    
    // Fallback to FNV-1a algorithm if Zobrist hash is not available
    const uint64_t FNV_PRIME = 1099511628211ULL;
    const uint64_t FNV_OFFSET = 14695981039346656037ULL;
    
    uint64_t hash = FNV_OFFSET;
    
    // Only hash the board part, not any extra metadata
    size_t board_elements = board_size * board_size;
    for (size_t i = 0; i < std::min(state_tensor.size(), board_elements); ++i) {
        // Convert float to byte representation for more stable hashing
        uint8_t byte_val = static_cast<uint8_t>(
            static_cast<int>(state_tensor[i] * 255.0f) & 0xFF);
        hash ^= byte_val;
        hash *= FNV_PRIME;
    }
    
    // Add the current player to the hash to distinguish same board with different players
    hash ^= static_cast<uint64_t>(current_player);
    hash *= FNV_PRIME;
    
    return hash;
=======
int MCTS::simulate_batched(
    const std::vector<float>& state_tensor,
    BatchEvaluator& batch_evaluator,
    std::vector<std::pair<MCTSNode*, int>>& path,
    std::vector<float>& leaf_state) {
    
    // Start with the root node
    MCTSNode* node = root_.get();
    if (!node) {
        std::cerr << "simulate_batched: ERROR - root node is null!" << std::endl;
        return -1;
    }
    
    std::vector<float> current_state = state_tensor;
    int current_player = 1; // Assuming 1 is the first player
    
    path.clear();
    path.push_back({node, -1});
    
    // Keep track of nodes where we added virtual loss
    std::vector<MCTSNode*> virtual_loss_nodes;
    
    // Selection: Traverse the tree to a leaf node
    const int MAX_DEPTH = 100; // Add a maximum depth to prevent infinite loops
    int depth = 0;
    while (node->is_expanded() && depth < MAX_DEPTH) {
        depth++;
        // Select the best child according to UCB
        auto [move, child] = node->select_child(c_puct_);
        if (!child) {
            break;
        }
        
        // Apply virtual loss if using multiple threads
        if (num_threads_ > 1) {
            child->add_virtual_loss(virtual_loss_weight_);
            virtual_loss_nodes.push_back(child);
        }
        
        // Apply the move to the current state
        current_state = apply_move_to_tensor(current_state, move, current_player);
        current_player = 3 - current_player; // Switch player (1 -> 2, 2 -> 1)
        
        // Add node and move to the path
        path.push_back({child, move});
        
        // Update node
        node = child;
        
        // Check transposition table if enabled
        if (use_transposition_table_ && transposition_table_) {
            // Calculate a hash for the current state
            uint64_t hash = compute_state_hash(current_state, current_player);
            
            // Create a path hash set to check for cycles efficiently
            std::unordered_set<MCTSNode*> path_nodes;
            for (const auto& [path_node, _] : path) {
                path_nodes.insert(path_node);
            }
            
            // Look up in transposition table
            MCTSNode* transposition_node = transposition_table_->lookup(hash);
            if (transposition_node) {
                // Check if this node is already in our path (cycle detection)
                if (path_nodes.find(transposition_node) == path_nodes.end()) {
                    // When using a transposition node, we need to clean up virtual loss
                    // from the original node and apply it to the transposition node
                    if (num_threads_ > 1) {
                        // Remove virtual loss from the original node
                        node->remove_virtual_loss(virtual_loss_weight_);
                        // Remove from our tracking list
                        if (!virtual_loss_nodes.empty()) {
                            virtual_loss_nodes.pop_back();
                        }
                        
                        // Add virtual loss to the transposition node
                        transposition_node->add_virtual_loss(virtual_loss_weight_);
                        virtual_loss_nodes.push_back(transposition_node);
                    }
                    
                    node = transposition_node;
                    path.back().first = node; // Replace the last node in path
                }
            }
        }
    }
    
    // Check for terminal state
    bool is_terminal = current_state.empty(); // Example check - replace with actual logic
    
    if (is_terminal) {
        // If we reached a terminal state, determine the winner (simplified example)
        int winner = 0; // 0 for draw, 1 for first player, 2 for second player
        float value;
        
        if (winner == 0) {
            // Draw
            value = 0.0f;
        } else if (winner == current_player) {
            // Current player wins
            value = 1.0f;
        } else {
            // Current player loses
            value = -1.0f;
        }
        
        // Backup the value immediately for terminal states
        node->backup(value);
        
        // Set leaf state for the caller
        leaf_state = current_state;
        
        // Return -1 to indicate immediate evaluation (no batch request)
        return -1;
    }
    
    // For non-terminal states, queue the evaluation request
    leaf_state = current_state;
    
    // Add the state to the batch evaluator and return the request ID
    return batch_evaluator.enqueue_position(current_state);
>>>>>>> 42bb511ab1410a992c3fb9fc8a11235d555aea77
}

std::vector<float> MCTS::apply_move_to_tensor(
    const std::vector<float>& state_tensor,
    int move,
    int player) {
    // A simple implementation for a board game like Gomoku
    // We assume state_tensor represents a flattened board
    
    // Create a copy of the state
    std::vector<float> new_state = state_tensor;
    
    // Ensure move index is valid
    if (move >= 0 && move < static_cast<int>(new_state.size())) {
        // Get the old piece value before applying the move
        float old_piece_value = new_state[move];
        int old_piece = (old_piece_value == 0.0f) ? 0 : 
                        (old_piece_value > 0.0f) ? 1 : 2;
        
        // Apply the move: we'll set it regardless of whether the position is empty
        // to ensure the state is being modified
        new_state[move] = (player == 1) ? 1.0f : -1.0f;
        
        // If we have a valid Zobrist hash, use incremental update for efficiency
        if (zobrist_hash_ && state_tensor.size() > 0) {
            // Check if we need to initialize the Zobrist hash with correct board size
            int board_size = static_cast<int>(std::sqrt(state_tensor.size()));
            
            // If the board size seems to have changed, reinitialize the Zobrist hash
            if (board_size * board_size != static_cast<int>(state_tensor.size()) || 
                board_size != zobrist_hash_->get_board_size()) {
                // Reinitialize with the new board size
                zobrist_hash_ = std::make_unique<ZobristHash>(board_size, 2);
            }
            
            // Check if there's a hash attached to the state tensor (in extra element)
            uint64_t hash = 0;
            if (state_tensor.size() > board_size * board_size) {
                // Try to extract the hash from the state tensor
                const float* ptr = state_tensor.data() + board_size * board_size;
                memcpy(&hash, ptr, sizeof(uint64_t) < sizeof(float) * 
                       (state_tensor.size() - board_size * board_size) ? 
                       sizeof(uint64_t) : sizeof(float) * (state_tensor.size() - board_size * board_size));
            } else {
                // Compute the hash from scratch if there's no stored hash
                hash = zobrist_hash_->compute_hash(state_tensor, 3 - player); // 3-player gives previous player
            }
            
            // Apply the move to the hash
            int new_piece = (player == 1) ? 1 : 2;
            hash = zobrist_hash_->update_hash(hash, move, old_piece, new_piece);
            
            // Toggle player
            hash = zobrist_hash_->toggle_player(hash, player, 3 - player);
            
            // Store the updated hash in the state
            // Make sure we have enough space to store 64-bit hash
            new_state.resize(board_size * board_size + sizeof(uint64_t) / sizeof(float) + 1);
            memcpy(new_state.data() + board_size * board_size, &hash, sizeof(hash));
        } else {
            // For more reliable state differentiation when not using Zobrist hash, 
            // add a small random perturbation to ensure states are different
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_real_distribution<float> dist(0.001f, 0.002f);
            new_state.push_back(dist(gen));
        }
    } else {
        std::cout << "apply_move_to_tensor: Invalid move index " << move << std::endl;
    }
    
    return new_state;
}

std::unordered_map<int, float> MCTS::search_batched(
    const std::vector<float>& state_tensor,
    const std::vector<int>& legal_moves,
    const std::function<std::vector<std::pair<std::vector<float>, float>>(const std::vector<std::vector<float>>&)>& batch_evaluator_func,
    size_t batch_size,
    size_t max_wait_ms,
    bool progressive_widening) {
    
    // Make sure Zobrist hash is initialized with the correct board size
    if (zobrist_hash_ && state_tensor.size() > 0) {
        // Check if we need to initialize the Zobrist hash with correct board size
        int board_size = static_cast<int>(std::sqrt(state_tensor.size()));
        // Only consider the actual board elements, not any attached metadata
        size_t board_elements = board_size * board_size;
        if (board_elements <= state_tensor.size() && 
            board_size != zobrist_hash_->get_board_size()) {
            // Reinitialize with the new board size
            zobrist_hash_ = std::make_unique<ZobristHash>(board_size, 2);
        }
    }
    
    // If root is not expanded, evaluate and expand it
    if (!root_->is_expanded()) {
        // Use the batch evaluator to evaluate the root state
        auto batch_results = batch_evaluator_func({state_tensor});
        
        if (batch_results.empty()) {
            std::cerr << "search_batched: ERROR - batch evaluator returned empty results for root!" << std::endl;
            return {};
        }
        
        auto [policy, value] = batch_results[0];
        
        // Apply progressive widening if needed
        if (progressive_widening && legal_moves.size() > 50) {
            // Create temporary mapping of moves to priors
            std::vector<std::pair<int, float>> move_priors;
            for (size_t i = 0; i < legal_moves.size(); ++i) {
                int move = legal_moves[i];
                float prior = (i < policy.size()) ? policy[i] : 0.0f;
                move_priors.emplace_back(move, prior);
            }
            
            // Sort by prior probability
            std::sort(move_priors.begin(), move_priors.end(), 
                [](const auto& a, const auto& b) { return a.second > b.second; });
            
            // Take only the top moves - ensure we don't go out of bounds
            int width = std::min(50, static_cast<int>(std::sqrt(legal_moves.size())));
            width = std::min(width, static_cast<int>(move_priors.size())); // Safety check
            std::vector<int> top_moves;
            std::vector<float> top_priors;
            
            for (int i = 0; i < width; ++i) {
                top_moves.push_back(move_priors[i].first);
                top_priors.push_back(move_priors[i].second);
            }
            
            // Normalize priors
            float sum_priors = std::accumulate(top_priors.begin(), top_priors.end(), 0.0f);
            if (sum_priors > 0.0f) {
                for (float& prior : top_priors) {
                    prior /= sum_priors;
                }
            }
            
            // Expand root with top moves
            expand_root(top_moves, top_priors);
        } else {
            // Expand root with all legal moves
            expand_root(legal_moves, policy);
        }
        
        // Add Dirichlet noise to root children
        add_dirichlet_noise(legal_moves);
    }
    
    // Create a BatchEvaluator to manage batched evaluation
    BatchEvaluator batch_evaluator(batch_evaluator_func, batch_size, max_wait_ms);
    batch_evaluator.start();
    
    // Track paths and leaf states for each simulation
    std::vector<std::vector<std::pair<MCTSNode*, int>>> paths(num_simulations_);
    std::vector<std::vector<float>> leaf_states(num_simulations_);
    std::vector<int> request_ids(num_simulations_, -1);
    
    // Run simulations
    if (num_threads_ > 1 && thread_pool_) {
        // Parallel simulations
        std::vector<std::future<int>> futures;
        
        // Make a copy of the state tensor to ensure thread safety
        auto state_tensor_copy = state_tensor;
        
        for (int i = 0; i < num_simulations_; ++i) {
            // Use capture by value for thread safety
            std::vector<float> thread_state = state_tensor_copy;
            futures.push_back(thread_pool_->enqueue([this, i, thread_state, &batch_evaluator, &paths, &leaf_states, &request_ids]() {
                request_ids[i] = this->simulate_batched(thread_state, batch_evaluator, paths[i], leaf_states[i]);
                return i;  // Return the simulation index
            }));
        }
        
        // Wait for all simulations to submit their evaluation requests
        for (auto& future : futures) {
            try {
                // Get the result to ensure the future completes
                future.get();
            } catch (const std::exception& e) {
                // Log the error and continue
                std::cerr << "Error in simulation: " << e.what() << std::endl;
            }
        }
        
        // Process batch results and update the tree
        // Create a new set of futures for processing results
        std::vector<std::future<void>> result_futures;
        
        for (int i = 0; i < num_simulations_; ++i) {
            if (request_ids[i] >= 0) {
                result_futures.push_back(thread_pool_->enqueue([this, i, &batch_evaluator, &paths, &leaf_states, &request_ids]() {
                    this->process_batch_result(request_ids[i], batch_evaluator, paths[i], leaf_states[i]);
                }));
            }
        }
        
        // Wait for all result processing to complete
        for (auto& future : result_futures) {
            try {
                future.get();
            } catch (const std::exception& e) {
                std::cerr << "Error in processing batch result: " << e.what() << std::endl;
            }
        }
    } else {
        // Sequential simulations
        for (int i = 0; i < num_simulations_; ++i) {
            // Submit the simulation
            request_ids[i] = simulate_batched(state_tensor, batch_evaluator, paths[i], leaf_states[i]);
            
            // Process the result immediately if it's available (terminal state)
            if (request_ids[i] < 0) {
                continue;  // Terminal state was processed in simulate_batched
            }
            
            // Process the batch result when it becomes available
            process_batch_result(request_ids[i], batch_evaluator, paths[i], leaf_states[i]);
        }
    }
    
    // Ensure all evaluations are complete
    batch_evaluator.stop();
    
    // Calculate and store probabilities
    current_probabilities_ = calculate_probabilities(temperature_);
    
    return current_probabilities_;
}

std::unordered_map<int, float> MCTS::calculate_probabilities(float temperature) const {
    std::unordered_map<int, float> probabilities;
    std::unordered_map<int, int> visits = root_->get_visit_counts();
    
    if (visits.empty()) {
        return probabilities;
    }
    
    // Special case for very small temperature (almost 0)
    if (temperature < 0.01f) {
        // Find the move with the most visits
        auto max_it = std::max_element(visits.begin(), visits.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        
        // Set probability 1.0 for the most visited move, 0.0 for others
        for (const auto& [move, count] : visits) {
            probabilities[move] = (move == max_it->first) ? 1.0f : 0.0f;
        }
    } else {
        // Apply temperature
        float sum = 0.0f;
        try {
            // Calculate temperature-adjusted visit counts
            for (const auto& [move, count] : visits) {
                float power = std::pow(static_cast<float>(count), 1.0f / temperature);
                probabilities[move] = power;
                sum += power;
            }
            
            // Normalize
            if (sum > 0.0f) {
                for (auto& [move, prob] : probabilities) {
                    prob /= sum;
                }
            }
        } catch (const std::exception& e) {
            // If we encounter numerical issues, fall back to normalizing the raw visit counts
            sum = 0.0f;
            for (const auto& [move, count] : visits) {
                probabilities[move] = static_cast<float>(count);
                sum += count;
            }
            
            if (sum > 0.0f) {
                for (auto& [move, prob] : probabilities) {
                    prob /= sum;
                }
            }
        }
    }
    
    return probabilities;
}

void MCTS::add_dirichlet_noise(const std::vector<int>& legal_moves) {
    if (dirichlet_noise_weight_ <= 0.0f || legal_moves.empty()) {
        return;
    }
    
    // Generate Dirichlet noise
    std::gamma_distribution<float> gamma(dirichlet_alpha_, 1.0f);
    std::vector<float> noise;
    
    for (size_t i = 0; i < legal_moves.size(); ++i) {
        noise.push_back(gamma(rng_));
    }
    
    // Normalize noise
    float sum = std::accumulate(noise.begin(), noise.end(), 0.0f);
    if (sum > 0.0f) {
        for (float& n : noise) {
            n /= sum;
        }
    }
    
    // Apply noise to root children
    for (size_t i = 0; i < legal_moves.size(); ++i) {
        int move = legal_moves[i];
        MCTSNode* child = root_->get_child(move);
        
        if (child) {
            // Update the prior with noise
            child->prior = (1.0f - dirichlet_noise_weight_) * child->prior +
                           dirichlet_noise_weight_ * noise[i];
        }
    }
}

void MCTS::expand_root(const std::vector<int>& legal_moves, const std::vector<float>& priors) {
    root_->expand(legal_moves, priors);
}

void MCTS::process_batch_result(
    int request_id,
    BatchEvaluator& batch_evaluator,
    std::vector<std::pair<MCTSNode*, int>>& path,
    const std::vector<float>& leaf_state) {
    
    // Skip processing if request_id is invalid (e.g., from terminal state)
    if (request_id < 0 || path.empty()) {
        return;
    }
    
    // Get the leaf node from the path
    MCTSNode* node = path.back().first;
    if (!node) {
        std::cerr << "process_batch_result: ERROR - leaf node is null!" << std::endl;
        return;
    }
    
    // Get the evaluation result from the batch evaluator
    auto [policy, value] = batch_evaluator.get_result(request_id);
    
    // Only proceed if the policy is valid
    if (policy.empty()) {
        std::cerr << "process_batch_result: ERROR - received empty policy!" << std::endl;
        return;
    }
    
    // Generate legal moves from the policy
    std::vector<int> legal_moves;
    for (size_t i = 0; i < policy.size(); ++i) {
        if (policy[i] > 0.0f) {
            legal_moves.push_back(i);
        }
    }
    
    // Expand the node with legal moves and prior probabilities
    node->expand(legal_moves, policy);
    
    // Store in transposition table if enabled
    if (use_transposition_table_ && transposition_table_) {
        // Determine the current player (assuming the last player to move is 3 - current_player)
        int current_player = path.size() % 2 == 0 ? 1 : 2;  // Simplified player tracking
        
        // Calculate a hash for the current state
        uint64_t hash = compute_state_hash(leaf_state, current_player);
        
        // Store the node in the transposition table
        transposition_table_->store(hash, node);
    }
    
    // Backup: Update the values of all nodes in the path
    node->backup(value);
}

size_t MCTS::count_nodes(const MCTSNode* node) const {
    if (!node) {
        return 0;
    }
    
    size_t count = 1; // Count this node
    
    // Use a non-recursive approach to avoid potential deadlocks with locks
    std::vector<const MCTSNode*> stack;
    
    // First, lock and copy pointers to all children
    {
        std::lock_guard<std::recursive_mutex> lock(node->children_mutex);
        for (const auto& child_pair : node->children) {
            if (child_pair.second) {
                stack.push_back(child_pair.second.get());
                count++; // Count each child
            }
        }
    }
    
    // Process each node in the stack without recursive calls
    while (!stack.empty()) {
        const MCTSNode* current = stack.back();
        stack.pop_back();
        
        if (current) {
            std::lock_guard<std::recursive_mutex> lock(current->children_mutex);
            for (const auto& child_pair : current->children) {
                if (child_pair.second) {
                    stack.push_back(child_pair.second.get());
                    count++; // Count each child
                }
            }
        }
    }
    
    return count;
}

} // namespace alphazero