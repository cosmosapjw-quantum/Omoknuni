#pragma once

#include <vector>
#include <memory>
#include <random>
#include <functional>
#include <unordered_map>
#include <future>
#include <cstdint>
#include <cstring>

#include "mcts_node.h"
#include "transposition_table.h"
#include "zobrist_hash.h"
#include "batch_evaluator.h"
#include "../utils/thread_pool.h"

namespace alphazero {

/**
 * Monte Carlo Tree Search algorithm with parallelization.
 * This implementation includes virtual loss, transposition table,
 * and other optimizations for a high-performance MCTS.
 */
class MCTS {
public:
    /**
     * Constructor.
     * 
     * @param num_simulations Number of simulations to run
     * @param c_puct Exploration constant for UCB
     * @param dirichlet_alpha Alpha parameter for Dirichlet noise
     * @param dirichlet_noise_weight Weight of Dirichlet noise added to root prior probabilities
     * @param virtual_loss_weight Weight of virtual loss
     * @param use_transposition_table Whether to use a transposition table
     * @param transposition_table_size Maximum size of transposition table
     * @param num_threads Number of worker threads for parallel MCTS
     */
    MCTS(int num_simulations = 800,
         float c_puct = 1.5f,
         float dirichlet_alpha = 0.3f,
         float dirichlet_noise_weight = 0.25f,
         float virtual_loss_weight = 1.0f,
         bool use_transposition_table = true,
         size_t transposition_table_size = 1000000,
         int num_threads = 1);
    
    /**
     * Destructor.
     */
    ~MCTS();
    
    /**
     * Set the number of simulations to run.
     * 
     * @param num_simulations Number of simulations
     */
    void set_num_simulations(int num_simulations);
    
    /**
     * Get the number of simulations to run.
     * 
     * @return Number of simulations
     */
    int get_num_simulations() const { return num_simulations_; }
    
    /**
     * Get the exploration constant for UCB.
     * 
     * @return Exploration constant
     */
    float get_c_puct() const { return c_puct_; }
    
    /**
     * Set the exploration constant for UCB.
     * 
     * @param c_puct Exploration constant
     */
    void set_c_puct(float c_puct);
    
    /**
     * Set the Dirichlet noise parameters.
     * 
     * @param alpha Alpha parameter for Dirichlet distribution
     * @param weight Weight of noise added to root prior probabilities
     */
    void set_dirichlet_noise(float alpha, float weight);
    
    /**
     * Set the virtual loss weight.
     * 
     * @param weight Weight of virtual loss
     */
    void set_virtual_loss_weight(float weight);
    
    /**
     * Set the temperature for move selection.
     * 
     * @param temperature Temperature parameter (lower values make selection more deterministic)
     */
    void set_temperature(float temperature);
    
    /**
     * Run the MCTS search from the current root.
     * 
     * @param state_tensor Tensor representation of the current game state
     * @param legal_moves List of legal moves from the current state
     * @param evaluator Function that takes a state tensor and returns (policy, value)
     * @param progressive_widening Whether to use progressive widening for large branching factors
     * @return Map of moves to visit probabilities
     */
    /**
     * Run the MCTS search from the current root.
     * This is the standard MCTS search method.
     * 
     * @param state_tensor Tensor representation of the current game state
     * @param legal_moves List of legal moves from the current state
     * @param evaluator Function that takes a state tensor and returns (policy, value)
     * @param progressive_widening Whether to use progressive widening for large branching factors
     * @return Map of moves to visit probabilities
     */
    std::unordered_map<int, float> search(
        const std::vector<float>& state_tensor,
        const std::vector<int>& legal_moves,
        const std::function<std::pair<std::vector<float>, float>(const std::vector<float>&)>& evaluator,
        bool progressive_widening = false);
        
    /**
     * Run the MCTS search using batched leaf evaluation.
     * This method is optimized for neural network evaluation by collecting leaf nodes
     * and evaluating them in batches, which can significantly improve performance
     * when using a GPU-accelerated neural network.
     * 
     * @param state_tensor Tensor representation of the current game state
     * @param legal_moves List of legal moves from the current state
     * @param batch_evaluator Function that takes a batch of state tensors and returns (policy, value) pairs
     * @param batch_size Maximum batch size for evaluation
     * @param max_wait_ms Maximum time to wait for a full batch in milliseconds
     * @param progressive_widening Whether to use progressive widening for large branching factors
     * @return Map of moves to visit probabilities
     */
    std::unordered_map<int, float> search_batched(
        const std::vector<float>& state_tensor,
        const std::vector<int>& legal_moves,
        const std::function<std::vector<std::pair<std::vector<float>, float>>(const std::vector<std::vector<float>>&)>& batch_evaluator,
        size_t batch_size = 16,
        size_t max_wait_ms = 10,
        bool progressive_widening = false);
    
    /**
     * Select a move based on the current search probabilities.
     * 
     * @param temperature Temperature parameter for move selection
     * @return Selected move
     */
    int select_move(float temperature = 1.0f);
    
    /**
     * Get the current search probabilities.
     * 
     * @return Map of moves to probabilities
     */
    std::unordered_map<int, float> get_probabilities() const;
    
    /**
     * Update the search tree with a move.
     * If the move exists in the tree, make the corresponding child the new root.
     * Otherwise, reset the tree with a new root.
     * 
     * @param move The move to update with
     */
    void update_with_move(int move);
    
    /**
     * Reset the search tree with a new root.
     */
    void reset();
    
    /**
     * Get the current root node.
     * 
     * @return Pointer to the root node
     */
    MCTSNode* get_root() const;
    
    /**
     * Get the number of nodes in the tree.
     * 
     * @return Number of nodes
     */
    size_t get_tree_size() const;
    
    /**
     * Get the number of threads used for parallel search.
     * 
     * @return Number of threads
     */
    int get_num_threads() const { return num_threads_; }
    
    /**
     * Process the result of a batched simulation and update the MCTS tree.
     * This method is called after a leaf node has been evaluated by the batch evaluator.
     * 
     * @param request_id The request ID returned by simulate_batched
     * @param batch_evaluator The BatchEvaluator that processed the request
     * @param path The path of nodes and moves from the root to the leaf
     * @param leaf_state The state tensor at the leaf node
     */
    void process_batch_result(
        int request_id,
        BatchEvaluator& batch_evaluator,
        std::vector<std::pair<MCTSNode*, int>>& path,
        const std::vector<float>& leaf_state);

private:
    // Root node of the search tree
    std::unique_ptr<MCTSNode> root_;
    
    // MCTS parameters
    int num_simulations_;
    float c_puct_;
    float dirichlet_alpha_;
    float dirichlet_noise_weight_;
    float virtual_loss_weight_;
    float temperature_;
    
    // Transposition table
    std::unique_ptr<TranspositionTable> transposition_table_;
    bool use_transposition_table_;
    
    // Zobrist hash for efficient state hashing
    std::unique_ptr<ZobristHash> zobrist_hash_;
    
    // Thread pool for parallel simulations
    std::unique_ptr<ThreadPool> thread_pool_;
    int num_threads_;
    
    // Current search probabilities
    std::unordered_map<int, float> current_probabilities_;
    
    // Random number generator
    std::mt19937 rng_;
    
    /**
     * Compute a hash value for a state tensor.
     * Uses Zobrist hashing for efficient board state hashing.
     * 
     * @param state_tensor The state tensor to hash
     * @param current_player The current player (to differentiate same board, different player)
     * @return Hash value for the state
     */
    uint64_t compute_state_hash(const std::vector<float>& state_tensor, int current_player) const {
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
    }
    
    /**
     * Perform a single MCTS simulation.
     * 
     * @param state_tensor Tensor representation of the current game state
     * @param evaluator Function that takes a state tensor and returns (policy, value)
     * @param path Path of moves taken during this simulation
     * @return Value of the leaf node
     */
    /**
     * Perform a single MCTS simulation.
     * 
     * @param state_tensor Tensor representation of the current game state
     * @param evaluator Function that takes a state tensor and returns (policy, value)
     * @param path Path of moves taken during this simulation
     * @return Value of the leaf node
     */
    float simulate(
        const std::vector<float>& state_tensor,
        const std::function<std::pair<std::vector<float>, float>(const std::vector<float>&)>& evaluator,
        std::vector<std::pair<MCTSNode*, int>>& path);
        
    /**
     * Perform MCTS simulation with batched leaf evaluation.
     * Instead of evaluating the leaf node directly, this method enqueues it for batch evaluation
     * and returns the evaluation asynchronously.
     * 
     * @param state_tensor Tensor representation of the current game state
     * @param batch_evaluator BatchEvaluator for evaluating leaf nodes in batches
     * @param path Path of moves taken during this simulation
     * @param leaf_state Output parameter to store the leaf state tensor
     * @return Unique request ID for retrieving the evaluation result later
     */
    int simulate_batched(
        const std::vector<float>& state_tensor,
        BatchEvaluator& batch_evaluator,
        std::vector<std::pair<MCTSNode*, int>>& path,
        std::vector<float>& leaf_state);
    
    /**
     * Apply a move to a state tensor to get a new state tensor.
     * This is a placeholder function and should be replaced with actual game logic.
     * 
     * @param state_tensor Current state tensor
     * @param move Move to apply
     * @param player Player making the move (1 or 2)
     * @return New state tensor
     */
    std::vector<float> apply_move_to_tensor(
        const std::vector<float>& state_tensor,
        int move,
        int player);
    
    /**
     * Calculate search probabilities based on visit counts and temperature.
     * 
     * @param temperature Temperature parameter
     * @return Map of moves to probabilities
     */
    std::unordered_map<int, float> calculate_probabilities(float temperature) const;
    
    /**
     * Apply Dirichlet noise to the root node's children.
     * 
     * @param legal_moves List of legal moves
     */
    void add_dirichlet_noise(const std::vector<int>& legal_moves);
    
    /**
     * Expand the root node with the given legal moves and priors.
     * 
     * @param legal_moves List of legal moves
     * @param priors Prior probabilities for the moves
     */
    void expand_root(const std::vector<int>& legal_moves, const std::vector<float>& priors);
    
    /**
     * Count the number of nodes in the tree rooted at the given node.
     * 
     * @param node Root node to count from
     * @return Number of nodes
     */
    size_t count_nodes(const MCTSNode* node) const;
};

} // namespace alphazero