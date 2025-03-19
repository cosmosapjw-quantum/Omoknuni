#include "zobrist_hash.h"

namespace alphazero {

ZobristHash::ZobristHash(int board_size, int num_players, unsigned seed)
    : board_size_(board_size), num_players_(num_players) {
    
    // Initialize random number generator
    std::mt19937_64 rng(seed);
    
    // Initialize hash table for pieces at positions
    int total_positions = board_size * board_size;
    hash_table_.resize(total_positions);
    
    for (int pos = 0; pos < total_positions; ++pos) {
        // Add 1 for each player plus empty (0)
        hash_table_[pos].resize(num_players + 1);
        for (int piece = 0; piece <= num_players; ++piece) {
            hash_table_[pos][piece] = rng();
        }
    }
    
    // Initialize player hashes
    player_hashes_.resize(num_players + 1);
    for (int player = 1; player <= num_players; ++player) {
        player_hashes_[player] = rng();
    }
}

uint64_t ZobristHash::compute_hash(const std::vector<float>& board, int player) const {
    uint64_t hash = 0;
    
    // Hash the board state
    for (size_t pos = 0; pos < board.size() && pos < hash_table_.size(); ++pos) {
        int piece = static_cast<int>(board[pos]);
        if (piece >= 0 && piece <= num_players_) {
            hash ^= hash_table_[pos][piece];
        }
    }
    
    // Hash the player to move
    if (player >= 1 && player <= num_players_) {
        hash ^= player_hashes_[player];
    }
    
    return hash;
}

uint64_t ZobristHash::compute_hash(const std::vector<int>& board, int player) const {
    uint64_t hash = 0;
    
    // Hash the board state
    for (size_t pos = 0; pos < board.size() && pos < hash_table_.size(); ++pos) {
        int piece = board[pos];
        if (piece >= 0 && piece <= num_players_) {
            hash ^= hash_table_[pos][piece];
        }
    }
    
    // Hash the player to move
    if (player >= 1 && player <= num_players_) {
        hash ^= player_hashes_[player];
    }
    
    return hash;
}

uint64_t ZobristHash::update_hash(uint64_t hash, int position, int old_piece, int new_piece) const {
    if (position < 0 || position >= static_cast<int>(hash_table_.size())) {
        return hash;
    }
    
    // XOR out the old piece
    if (old_piece >= 0 && old_piece <= num_players_) {
        hash ^= hash_table_[position][old_piece];
    }
    
    // XOR in the new piece
    if (new_piece >= 0 && new_piece <= num_players_) {
        hash ^= hash_table_[position][new_piece];
    }
    
    return hash;
}

uint64_t ZobristHash::toggle_player(uint64_t hash, int old_player, int new_player) const {
    // XOR out the old player
    if (old_player >= 1 && old_player <= num_players_) {
        hash ^= player_hashes_[old_player];
    }
    
    // XOR in the new player
    if (new_player >= 1 && new_player <= num_players_) {
        hash ^= player_hashes_[new_player];
    }
    
    return hash;
}

} // namespace alphazero