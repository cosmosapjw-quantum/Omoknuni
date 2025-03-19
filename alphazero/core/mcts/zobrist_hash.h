#pragma once

#include <vector>
#include <random>
#include <cstdint>
#include <unordered_map>

namespace alphazero {

/**
 * Implements Zobrist hashing for game states.
 * Zobrist hashing is an efficient way to hash board positions in board games.
 * It uses a table of random numbers to hash different aspects of the board state.
 */
class ZobristHash {
public:
    /**
     * Constructor.
     * 
     * @param board_size Size of the board (e.g., 9 for 9x9 board)
     * @param num_players Number of players in the game
     * @param seed Seed for the random number generator (default: 42)
     */
    ZobristHash(int board_size, int num_players = 2, unsigned seed = 42);
    
    /**
     * Get the board size used for this Zobrist hash.
     * 
     * @return The board size
     */
    int get_board_size() const { return board_size_; }
    
    /**
     * Get the number of players used for this Zobrist hash.
     * 
     * @return The number of players
     */
    int get_num_players() const { return num_players_; }
    
    /**
     * Compute a hash value for a given board state.
     * 
     * @param board The board state as a flat vector (0 for empty, 1 for player 1, 2 for player 2, etc.)
     * @param player The player to move (1 for player 1, 2 for player 2, etc.)
     * @return The Zobrist hash value for the board state
     */
    uint64_t compute_hash(const std::vector<float>& board, int player) const;
    
    /**
     * Compute a hash value for a given board state.
     * 
     * @param board The board state as a flat vector of integers
     * @param player The player to move
     * @return The Zobrist hash value for the board state
     */
    uint64_t compute_hash(const std::vector<int>& board, int player) const;
    
    /**
     * Update a hash value incrementally with a move.
     * 
     * @param hash The current hash value
     * @param position The position on the board where the move is made
     * @param old_piece The piece that was previously at the position (0 for empty)
     * @param new_piece The piece that is placed at the position
     * @return The updated hash value
     */
    uint64_t update_hash(uint64_t hash, int position, int old_piece, int new_piece) const;
    
    /**
     * Toggle the player in the hash value.
     * 
     * @param hash The current hash value
     * @param old_player The current player
     * @param new_player The new player
     * @return The updated hash value
     */
    uint64_t toggle_player(uint64_t hash, int old_player, int new_player) const;
    
private:
    // Board size
    int board_size_;
    
    // Number of players
    int num_players_;
    
    // Zobrist hash table for pieces at positions:
    // hash_table_[position][piece] gives the random number for piece at position
    std::vector<std::vector<uint64_t>> hash_table_;
    
    // Random numbers for the player to move
    std::vector<uint64_t> player_hashes_;
};

} // namespace alphazero