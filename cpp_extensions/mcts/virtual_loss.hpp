/**
 * @file virtual_loss.hpp
 * @brief Thread-safe virtual loss mechanism for MCTS tree search
 *
 * Virtual loss is a technique to prevent multiple search threads from
 * exploring the same path simultaneously. When a thread traverses down
 * the tree, it applies a temporary "virtual loss" to each node along
 * the path. This makes the path appear less attractive to other threads,
 * encouraging them to explore different branches.
 *
 * Key features:
 * - Thread-safe atomic operations on virtual loss values
 * - Configurable virtual loss magnitude (default 1.0)
 * - Path-based application and removal during tree traversal
 * - Integration with PUCT selection formula
 */

#pragma once

#include "tree.hpp"
#include <vector>
#include <atomic>
#include <cstdint>

namespace mcts {

/**
 * @brief Configuration for virtual loss behavior
 */
struct VirtualLossConfig {
    float magnitude = 1.0f;           // Virtual loss value to apply
    bool enable_virtual_loss = true;  // Enable/disable virtual loss

    VirtualLossConfig() = default;

    VirtualLossConfig(float mag, bool enable = true)
        : magnitude(mag), enable_virtual_loss(enable) {}
};

/**
 * @brief Thread-safe virtual loss manager for MCTS tree
 *
 * This class provides atomic operations for applying and removing
 * virtual loss along search paths. Virtual loss helps coordinate
 * multiple search threads by temporarily penalizing nodes being
 * explored by other threads.
 */
class VirtualLossManager {
public:
    /**
     * @brief Initialize virtual loss manager
     *
     * @param tree Reference to MCTS tree to manage
     * @param config Virtual loss configuration
     */
    explicit VirtualLossManager(MCTSTree& tree, const VirtualLossConfig& config = VirtualLossConfig());

    /**
     * @brief Apply virtual loss along a path from leaf to root
     *
     * This function should be called when a thread starts exploring
     * a path. It applies virtual loss to each node in the path to
     * discourage other threads from following the same route.
     *
     * @param path Vector of node indices from leaf to root
     * @return true if virtual loss was successfully applied to all nodes
     */
    bool apply_virtual_loss_to_path(const std::vector<NodeIndex>& path);

    /**
     * @brief Remove virtual loss along a path from leaf to root
     *
     * This function should be called when a thread finishes exploring
     * a path and is ready to backup the results. It removes the virtual
     * loss that was previously applied.
     *
     * @param path Vector of node indices from leaf to root (same as apply)
     * @return true if virtual loss was successfully removed from all nodes
     */
    bool remove_virtual_loss_from_path(const std::vector<NodeIndex>& path);

    /**
     * @brief Apply virtual loss to a single node atomically
     *
     * @param node_index Index of node to apply virtual loss to
     * @param magnitude Virtual loss value to add (default: config magnitude)
     * @return true if virtual loss was successfully applied
     */
    bool apply_virtual_loss(NodeIndex node_index, float magnitude = -1.0f);

    /**
     * @brief Remove virtual loss from a single node atomically
     *
     * @param node_index Index of node to remove virtual loss from
     * @param magnitude Virtual loss value to remove (default: config magnitude)
     * @return true if virtual loss was successfully removed
     */
    bool remove_virtual_loss(NodeIndex node_index, float magnitude = -1.0f);

    /**
     * @brief Get current virtual loss value for a node
     *
     * This is a non-atomic read for debugging purposes.
     * For thread-safe access during selection, use tree methods directly.
     *
     * @param node_index Index of node to query
     * @return Current virtual loss value
     */
    float get_virtual_loss(NodeIndex node_index) const;

    /**
     * @brief Reset all virtual loss values to zero
     *
     * Useful for debugging and testing. Should not be called
     * during active search operations.
     */
    void reset_all_virtual_loss();

    /**
     * @brief Get virtual loss configuration
     */
    const VirtualLossConfig& get_config() const { return config_; }

    /**
     * @brief Update virtual loss configuration
     *
     * @param new_config New configuration to apply
     */
    void set_config(const VirtualLossConfig& new_config) { config_ = new_config; }

    /**
     * @brief Get statistics about virtual loss usage
     *
     * @return Struct containing virtual loss statistics
     */
    struct VirtualLossStats {
        std::size_t total_applications = 0;    // Total times virtual loss was applied
        std::size_t total_removals = 0;        // Total times virtual loss was removed
        std::size_t current_active_paths = 0;  // Current number of active paths with virtual loss
        float max_virtual_loss = 0.0f;         // Maximum virtual loss value currently in tree
        float avg_virtual_loss = 0.0f;         // Average virtual loss value across all nodes
    };

    VirtualLossStats get_statistics() const;

private:
    MCTSTree& tree_;                    // Reference to MCTS tree
    VirtualLossConfig config_;          // Virtual loss configuration

    // Statistics tracking (atomic for thread safety)
    mutable std::atomic<std::size_t> total_applications_{0};
    mutable std::atomic<std::size_t> total_removals_{0};

    /**
     * @brief Validate node index before virtual loss operations
     */
    bool validate_node_index(NodeIndex node_index) const;

    /**
     * @brief Atomic add operation on virtual loss value
     *
     * Uses compare-and-swap loop to ensure thread-safe updates
     * to the virtual loss array.
     */
    bool atomic_add_virtual_loss(NodeIndex node_index, float delta);
};

/**
 * @brief RAII wrapper for automatic virtual loss management
 *
 * This class automatically applies virtual loss when constructed
 * and removes it when destroyed, ensuring proper cleanup even
 * if exceptions occur during search.
 */
class VirtualLossGuard {
public:
    /**
     * @brief Apply virtual loss to path and store for automatic removal
     *
     * @param manager Reference to virtual loss manager
     * @param path Path to apply virtual loss to
     */
    VirtualLossGuard(VirtualLossManager& manager, const std::vector<NodeIndex>& path);

    /**
     * @brief Remove virtual loss from stored path
     */
    ~VirtualLossGuard();

    // Disable copy/move to prevent double-removal
    VirtualLossGuard(const VirtualLossGuard&) = delete;
    VirtualLossGuard& operator=(const VirtualLossGuard&) = delete;
    VirtualLossGuard(VirtualLossGuard&&) = delete;
    VirtualLossGuard& operator=(VirtualLossGuard&&) = delete;

    /**
     * @brief Check if virtual loss was successfully applied
     */
    bool is_valid() const { return valid_; }

    /**
     * @brief Manually remove virtual loss (called automatically by destructor)
     */
    void release();

private:
    VirtualLossManager& manager_;
    std::vector<NodeIndex> path_;
    bool valid_;
    bool released_;
};

} // namespace mcts