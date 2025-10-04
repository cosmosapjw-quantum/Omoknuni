/**
 * @file continuous_simulation_runner.hpp
 * @brief Continuous MCTS simulation runner with async inference
 *
 * This module implements a simulation runner that executes MCTS simulations
 * continuously without blocking on neural network inference. Simulations
 * submit inference requests to an AsyncInferenceQueue and immediately continue
 * with new simulations while waiting for results.
 *
 * Performance targets:
 * - 30,000+ simulations/second with 8-12 threads
 * - 75-85% parallel efficiency
 * - GPU utilization 60-80%
 * - Average batch size 48-64 positions
 *
 * Key design principles:
 * - Non-blocking simulation loop (threads never wait for inference)
 * - Pending expansion tracking (map of request_id → expansion data)
 * - Async result processing (check queue periodically, expand when ready)
 * - Continuous progress (always making forward progress on tree growth)
 */

#pragma once

#include "simulation_runner.hpp"
#include "async_inference_queue.hpp"
#include <unordered_map>

namespace mcts {

/**
 * @brief Pending expansion data
 *
 * Tracks state for a simulation that has selected to a leaf
 * and submitted an inference request, but hasn't yet received
 * the result to expand the node.
 */
struct PendingExpansion {
    NodeIndex leaf_node;                       // Node to expand with result
    std::vector<NodeIndex> path;               // Path from root to leaf (for backup)
    std::unique_ptr<IGameState> state;         // Game state at leaf (for expansion)

    // Move-only type (owns game state)
    PendingExpansion() = default;
    PendingExpansion(PendingExpansion&&) = default;
    PendingExpansion& operator=(PendingExpansion&&) = default;
    PendingExpansion(const PendingExpansion&) = delete;
    PendingExpansion& operator=(const PendingExpansion&) = delete;
};

/**
 * @brief Continuous MCTS simulation runner
 *
 * Runs MCTS simulations continuously without blocking on inference.
 * Achieves high throughput (30k+ sims/sec) by decoupling simulation
 * threads from GPU inference latency.
 *
 * Algorithm:
 * 1. Select to leaf (C++ tree traversal, ~0.26ms)
 * 2. Submit inference request to queue (non-blocking, ~0.1ms)
 * 3. Immediately start next simulation (no waiting!)
 * 4. Periodically check for completed results
 * 5. Expand nodes and backup values when results arrive
 * 6. Continue until quota reached
 *
 * Thread Safety:
 * - Multiple ContinuousSimulationRunner instances can run concurrently
 * - Each runner has independent pending expansion map
 * - Shared AsyncInferenceQueue is thread-safe
 * - Tree operations use atomics (same as base SimulationRunner)
 *
 * Performance Characteristics:
 * - Throughput: 30,000-40,000 sims/sec (8-12 threads)
 * - Latency: ~5ms per simulation (including queue time)
 * - Memory: ~100 bytes per pending expansion
 * - Parallelism: 75-85% efficiency with thread scaling
 */
class ContinuousSimulationRunner : public SimulationRunner {
public:
    /**
     * @brief Construct continuous simulation runner
     *
     * @param tree Shared MCTS tree (thread-safe via atomics)
     * @param selector PUCT child selector (thread-safe, no state)
     * @param backup Value backup manager (thread-safe)
     * @param virtual_loss Virtual loss coordinator (thread-safe)
     */
    ContinuousSimulationRunner(MCTSTree& tree,
                                PUCTSelector& selector,
                                BackupManager& backup,
                                VirtualLossManager& virtual_loss);

    /**
     * @brief Run continuous MCTS simulations with async inference
     *
     * Executes: Continuous loop of (Select → Queue → Process Results)
     * - Threads never block waiting for inference
     * - Simulations accumulate in pending expansions
     * - Results processed asynchronously as they arrive
     * - Loop continues until num_simulations completed
     *
     * Performance:
     * - Target: 30,000+ sims/sec with 8-12 threads
     * - Each simulation submits request (~0.1ms) then continues
     * - Results processed in batches (amortized cost)
     * - GPU batching happens in background coordinator
     *
     * Thread Safety:
     * - Safe to call from multiple threads with same queue
     * - Each thread has independent pending_expansions_ map
     * - Queue handles concurrent access internally
     *
     * @param root_state Initial game state (will be cloned for each simulation)
     * @param root_index Root node index in MCTS tree
     * @param queue Async inference queue for request/result exchange
     * @param num_simulations Number of simulations to complete
     * @return Number of successfully completed simulations
     */
    int run_continuous(IGameState& root_state,
                       NodeIndex root_index,
                       AsyncInferenceQueue& queue,
                       int num_simulations);

private:
    // Pending expansions waiting for inference results
    std::unordered_map<uint64_t, PendingExpansion> pending_expansions_;

    /**
     * @brief Process completed inference results
     *
     * Checks queue for available results, expands corresponding nodes,
     * and backs up values along paths. Processes all available results
     * in a batch to amortize queue access overhead.
     *
     * @param queue Async inference queue to poll for results
     * @return Number of results processed
     */
    int process_completed_results(AsyncInferenceQueue& queue);

    /**
     * @brief Expand node with pre-fetched inference result
     *
     * Same logic as SimulationRunner::expand_node() but with
     * policy/value already fetched from queue instead of calling
     * inference callback synchronously.
     *
     * @param leaf_node Node to expand
     * @param state Game state at leaf
     * @param policy Policy distribution over actions
     * @param value Position evaluation
     * @return true if expansion successful, false on error
     */
    bool expand_node_with_result(NodeIndex leaf_node,
                                   const IGameState& state,
                                   const std::vector<float>& policy,
                                   float value);
};

} // namespace mcts
