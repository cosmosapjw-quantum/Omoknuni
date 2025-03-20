#include "mcts.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <chrono>
#include <iostream>
#include <unordered_set>

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

MCTS::~MCTS() {
    // All resources are automatically cleaned up by smart pointers
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

std::unordered_map<int, float> MCTS::search(
    const std::vector<float>& state_tensor,
    const std::vector<int>& legal_moves,
    const std::function<std::pair<std::vector<float>, float>(const std::vector<float>&)>& evaluator,
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
        auto [policy, value] = evaluator(state_tensor);
        
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
    
    // Run simulations
    if (num_threads_ > 1 && thread_pool_) {
        // Parallel simulations
        std::vector<std::future<float>> futures;
        
        // Make a copy of the state tensor to ensure thread safety
        auto state_tensor_copy = state_tensor;
        
        for (int i = 0; i < num_simulations_; ++i) {
            // Use capture by value for thread safety
            // Make a separate copy of the state tensor for each thread to avoid data races
            std::vector<float> thread_state = state_tensor_copy;
            futures.push_back(thread_pool_->enqueue([this, thread_state, evaluator]() {
                std::vector<std::pair<MCTSNode*, int>> path;
                return this->simulate(thread_state, evaluator, path);
            }));
        }
        
        // Wait for all simulations to complete
        for (auto& future : futures) {
            try {
                future.wait();
            } catch (const std::exception& e) {
                // Log the error and continue
                std::cerr << "Error in simulation: " << e.what() << std::endl;
            }
        }
    } else {
        // Sequential simulations
        for (int i = 0; i < num_simulations_; ++i) {
            std::vector<std::pair<MCTSNode*, int>> path;
            simulate(state_tensor, evaluator, path);
        }
    }
    
    // Calculate and store probabilities
    current_probabilities_ = calculate_probabilities(temperature_);
    
    return current_probabilities_;
}

int MCTS::select_move(float temperature) {
    // Use provided temperature instead of the object's temperature
    auto probs = calculate_probabilities(temperature);
    
    // Extract moves and probabilities for sampling
    std::vector<int> moves;
    std::vector<float> probabilities;
    
    for (const auto& [move, prob] : probs) {
        moves.push_back(move);
        probabilities.push_back(prob);
    }
    
    // Check if we have any moves
    if (moves.empty()) {
        return -1;
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
            std::lock_guard<std::mutex> lock(child->children_mutex);
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
        std::cerr << "simulate: ERROR - root node is null!" << std::endl;
        return 0.0f;
    }
    
    std::vector<float> current_state = state_tensor;
    int current_player = 1; // Assuming 1 is the first player
    
    path.clear();
    path.push_back({node, -1});
    
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
                    node = transposition_node;
                    path.back().first = node; // Replace the last node in path
                }
            }
        }
    }
    
    // Evaluation: If we're not at a terminal state, evaluate and expand the node
    float value;
    // This is a placeholder for terminal state detection - initialize with a meaningful value
    // In a real implementation, this would be computed based on the game state
    bool is_terminal = current_state.empty(); // Example check - replace with actual logic
    if (!is_terminal) {
        // Evaluate the current state
        auto [policy, state_value] = evaluator(current_state);
        
        // Get legal moves
        // This is a placeholder - the actual legal move generation would depend on the game state
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
            // Calculate a hash for the current state using our helper function
            uint64_t hash = compute_state_hash(current_state, current_player);
            
            // Store the node in the transposition table
            transposition_table_->store(hash, node);
        }
        
        // Use the evaluated value
        value = state_value;
    } else {
        // If we reached a terminal state, determine the winner
        // This is a placeholder - the actual winner determination would depend on the game state
        int winner = 0; // 0 for draw, 1 for first player, 2 for second player
        
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
    }
    
    // Backup: Update the values of all nodes in the path
    node->backup(value);
    
    return value;
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
        // std::cout << "apply_move_to_tensor: Invalid move index " << move << std::endl;
    }
    
    return new_state;
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

size_t MCTS::count_nodes(const MCTSNode* node) const {
    if (!node) {
        return 0;
    }
    
    size_t count = 1; // Count this node
    
    // Use a non-recursive approach to avoid potential deadlocks with locks
    std::vector<const MCTSNode*> stack;
    
    // First, lock and copy pointers to all children
    {
        std::lock_guard<std::mutex> lock(node->children_mutex);
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
            std::lock_guard<std::mutex> lock(current->children_mutex);
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