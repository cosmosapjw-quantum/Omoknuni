/**
 * @file python_bindings.cpp
 * @brief Python bindings for MCTS components including virtual loss
 *
 * This module exposes the high-performance MCTS tree and virtual loss
 * mechanisms to Python for testing and integration with the AlphaZero
 * training pipeline.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include <vector>

#include "tree.hpp"
#include "virtual_loss.hpp"
#include "selection.hpp"
#include "backup.hpp"
#include "simulation_runner.hpp"
#include "inference_callback.hpp"
#include "async_inference_queue.hpp"
#include "continuous_simulation_runner.hpp"
#include "batch_inference_coordinator.hpp"
#include "instrumentation.hpp"

namespace py = pybind11;

namespace mcts {
namespace python {

using NoGil = py::call_guard<py::gil_scoped_release>;

/**
 * @brief Create a simple MCTS tree for testing purposes
 *
 * This creates a minimal tree with a few nodes for virtual loss testing.
 * In production, trees would be created by the search algorithm.
 */
std::shared_ptr<MCTSTree> create_test_tree(int max_nodes = 1000) {
    auto tree = std::make_shared<MCTSTree>(max_nodes);

    // Add root node for testing
    NodeIndex root = tree->add_root_node(0.5f, 0);

    return tree;
}

/**
 * @brief Factory function to create virtual loss manager with test tree
 */
std::shared_ptr<VirtualLossManager> create_test_virtual_loss_manager(
    std::shared_ptr<MCTSTree> tree,
    const VirtualLossConfig& config = VirtualLossConfig()) {
    return std::make_shared<VirtualLossManager>(*tree, config);
}

/**
 * @brief Factory function to create PUCT selector
 */
std::shared_ptr<PUCTSelector> create_puct_selector(
    const PUCTConfig& config = PUCTConfig()) {
    return std::make_shared<PUCTSelector>(config);
}

/**
 * @brief Factory function to create backup manager
 */
std::shared_ptr<BackupManager> create_backup_manager(
    std::shared_ptr<MCTSTree> tree,
    const BackupConfig& config = BackupConfig()) {
    return std::make_shared<BackupManager>(*tree, config);
}

PYBIND11_MODULE(mcts_py, m) {
    m.doc() = "MCTS Tree and Virtual Loss Python bindings";

    // Node index type
    m.attr("NULL_NODE_INDEX") = NULL_NODE_INDEX;

    // Node Flags
    py::class_<NodeFlags>(m, "NodeFlags")
        .def(py::init<>())
        .def("is_expanded", &NodeFlags::is_expanded)
        .def("is_terminal", &NodeFlags::is_terminal)
        .def("current_player", &NodeFlags::current_player)
        .def("is_expanding", &NodeFlags::is_expanding)
        .def("set_expanded", &NodeFlags::set_expanded)
        .def("set_terminal", &NodeFlags::set_terminal)
        .def("set_current_player", &NodeFlags::set_current_player)
        .def("set_expanding", &NodeFlags::set_expanding)
        .def_readwrite("flags", &NodeFlags::flags);

    // Node Info
    py::class_<NodeInfo>(m, "NodeInfo")
        .def(py::init<>())
        .def_readwrite("index", &NodeInfo::index)
        .def_readwrite("visit_count", &NodeInfo::visit_count)
        .def_readwrite("total_value", &NodeInfo::total_value)
        .def_readwrite("prior_prob", &NodeInfo::prior_prob)
        .def_readwrite("virtual_loss", &NodeInfo::virtual_loss)
        .def_readwrite("parent_index", &NodeInfo::parent_index)
        .def_readwrite("first_child_index", &NodeInfo::first_child_index)
        .def_readwrite("num_children", &NodeInfo::num_children)
        .def_readwrite("flags", &NodeInfo::flags)
        .def("q_value", &NodeInfo::q_value)
        .def("is_root", &NodeInfo::is_root);

    // Virtual Loss Configuration
    py::class_<VirtualLossConfig>(m, "VirtualLossConfig")
        .def(py::init<>())
        .def(py::init<float, bool>(), py::arg("magnitude"), py::arg("enable") = true)
        .def_readwrite("magnitude", &VirtualLossConfig::magnitude)
        .def_readwrite("enable_virtual_loss", &VirtualLossConfig::enable_virtual_loss);

    // MCTS Tree (complete interface for production use)
    py::class_<MCTSTree, std::shared_ptr<MCTSTree>>(m, "MCTSTree")
        .def(py::init<std::size_t>(), py::arg("max_nodes") = 50000000, NoGil())
        .def("allocate_node", &MCTSTree::allocate_node, NoGil())
        .def("allocate_nodes", &MCTSTree::allocate_nodes, NoGil())
        .def("deallocate_node", &MCTSTree::deallocate_node, NoGil())
        .def("deallocate_nodes", &MCTSTree::deallocate_nodes, NoGil())
        .def("get_node_count", &MCTSTree::get_node_count)
        .def("get_max_nodes", &MCTSTree::get_max_nodes)
        .def("add_root_node", &MCTSTree::add_root_node, NoGil())
        .def("get_root_index", &MCTSTree::get_root_index)
        .def("is_valid_index", &MCTSTree::is_valid_index)
        .def("clear", &MCTSTree::clear, NoGil())
        // Node data access (with GIL release for hot path performance)
        .def("get_visit_count", &MCTSTree::get_visit_count, NoGil())
        .def("get_total_value", &MCTSTree::get_total_value, NoGil())
        .def("get_prior_prob", &MCTSTree::get_prior_prob, NoGil())
        .def("get_virtual_loss", &MCTSTree::get_virtual_loss, NoGil())
        .def("get_parent_index", &MCTSTree::get_parent_index, NoGil())
        .def("get_first_child_index", &MCTSTree::get_first_child_index, NoGil())
        .def("get_num_children", &MCTSTree::get_num_children, NoGil())
        .def("get_flags", &MCTSTree::get_flags, NoGil())
        .def("get_node_info", &MCTSTree::get_node_info, NoGil())
        // Node data modification (with GIL release)
        .def("set_visit_count", &MCTSTree::set_visit_count, NoGil())
        .def("set_total_value", &MCTSTree::set_total_value, NoGil())
        .def("set_prior_prob", &MCTSTree::set_prior_prob, NoGil())
        .def("set_virtual_loss", &MCTSTree::set_virtual_loss, NoGil())
        .def("set_parent_index", &MCTSTree::set_parent_index, NoGil())
        .def("set_first_child_index", &MCTSTree::set_first_child_index, NoGil())
        .def("set_num_children", &MCTSTree::set_num_children, NoGil())
        .def("set_flags", &MCTSTree::set_flags, NoGil())
        .def("atomic_try_set_expanded", &MCTSTree::atomic_try_set_expanded, NoGil(),
            "Atomically try to set expanded flag. Returns true if successful (caller owns expansion), "
            "false if already expanded by another thread.")
        .def("atomic_try_mark_expanding", &MCTSTree::atomic_try_mark_expanding, NoGil(),
            "Atomically mark a node as in-flight for expansion. Returns true if caller owns the request.")
        .def("clear_expanding_flag", &MCTSTree::clear_expanding_flag, NoGil(),
            "Clear the expanding flag after inference completes (success or failure).")
        // Memory and performance
        .def("get_memory_usage", &MCTSTree::get_memory_usage)
        .def("get_bytes_per_node", &MCTSTree::get_bytes_per_node)
        .def("get_available_nodes", &MCTSTree::get_available_nodes)
        .def("has_space_for", &MCTSTree::has_space_for)
        .def("validate_tree", &MCTSTree::validate_tree)
        // Move storage methods
        .def("get_move", &MCTSTree::get_move, NoGil())
        .def("set_move", &MCTSTree::set_move, NoGil());

    // Virtual Loss Manager
    py::class_<VirtualLossManager, std::shared_ptr<VirtualLossManager>>(m, "VirtualLossManager")
        .def("get_config", &VirtualLossManager::get_config,
             py::return_value_policy::reference_internal, NoGil())
        .def("set_config", &VirtualLossManager::set_config, NoGil())
        .def("get_virtual_loss", &VirtualLossManager::get_virtual_loss, NoGil())
        .def("reset_all_virtual_loss", &VirtualLossManager::reset_all_virtual_loss, NoGil())
        .def("apply_virtual_loss", &VirtualLossManager::apply_virtual_loss,
             py::arg("node_index"), py::arg("magnitude") = -1.0f, NoGil())
        .def("remove_virtual_loss", &VirtualLossManager::remove_virtual_loss,
             py::arg("node_index"), py::arg("magnitude") = -1.0f, NoGil())
        .def("apply_virtual_loss_to_path", &VirtualLossManager::apply_virtual_loss_to_path, NoGil())
        .def("remove_virtual_loss_from_path", &VirtualLossManager::remove_virtual_loss_from_path, NoGil())
        .def("get_statistics", &VirtualLossManager::get_statistics, NoGil());

    // Virtual Loss Statistics
    py::class_<VirtualLossManager::VirtualLossStats>(m, "VirtualLossStats")
        .def_readonly("total_applications", &VirtualLossManager::VirtualLossStats::total_applications)
        .def_readonly("total_removals", &VirtualLossManager::VirtualLossStats::total_removals)
        .def_readonly("current_active_paths", &VirtualLossManager::VirtualLossStats::current_active_paths)
        .def_readonly("max_virtual_loss", &VirtualLossManager::VirtualLossStats::max_virtual_loss)
        .def_readonly("avg_virtual_loss", &VirtualLossManager::VirtualLossStats::avg_virtual_loss);

    // Virtual Loss Guard (RAII wrapper)
    py::class_<VirtualLossGuard>(m, "VirtualLossGuard")
        .def(py::init<VirtualLossManager&, const std::vector<NodeIndex>&>(), NoGil())
        .def("is_valid", &VirtualLossGuard::is_valid, NoGil())
        .def("release", &VirtualLossGuard::release, NoGil());

    // PUCT Configuration
    py::class_<PUCTConfig>(m, "PUCTConfig")
        .def(py::init<>())
        .def_readwrite("cpuct", &PUCTConfig::cpuct)
        .def_readwrite("fpu_value", &PUCTConfig::fpu_value)
        .def_readwrite("use_fpu", &PUCTConfig::use_fpu)
        .def_readwrite("enable_simd", &PUCTConfig::enable_simd);

    // Selection Result
    py::class_<SelectionResult>(m, "SelectionResult")
        .def(py::init<>())
        .def_readwrite("selected_child", &SelectionResult::selected_child)
        .def_readwrite("best_puct_value", &SelectionResult::best_puct_value)
        .def_readwrite("child_position", &SelectionResult::child_position)
        .def_readwrite("valid", &SelectionResult::valid);

    // PUCT Selector
    py::class_<PUCTSelector, std::shared_ptr<PUCTSelector>>(m, "PUCTSelector")
        .def("select_child", &PUCTSelector::select_child, NoGil())
        .def("set_config", &PUCTSelector::set_config, NoGil())
        .def("get_config", &PUCTSelector::get_config,
             py::return_value_policy::reference_internal, NoGil())
        .def_static("is_avx2_supported", &PUCTSelector::is_avx2_supported);

    // Backup Configuration
    py::class_<BackupConfig>(m, "BackupConfig")
        .def(py::init<>())
        .def(py::init<bool, bool, float, float>(),
             py::arg("enable_value_clipping"), py::arg("enable_statistics") = true,
             py::arg("value_clip_min") = -1.0f, py::arg("value_clip_max") = 1.0f)
        .def_readwrite("enable_value_clipping", &BackupConfig::enable_value_clipping)
        .def_readwrite("enable_statistics", &BackupConfig::enable_statistics)
        .def_readwrite("value_clip_min", &BackupConfig::value_clip_min)
        .def_readwrite("value_clip_max", &BackupConfig::value_clip_max);

    // Backup Result
    py::class_<BackupResult>(m, "BackupResult")
        .def(py::init<>())
        .def_readwrite("success", &BackupResult::success)
        .def_readwrite("nodes_updated", &BackupResult::nodes_updated)
        .def_readwrite("final_root_value", &BackupResult::final_root_value)
        .def_readwrite("original_leaf_value", &BackupResult::original_leaf_value);

    // Backup Manager
    py::class_<BackupManager, std::shared_ptr<BackupManager>>(m, "BackupManager")
        .def("backup_value_along_path", &BackupManager::backup_value_along_path,
             py::arg("path"), py::arg("leaf_value"), py::arg("virtual_loss_manager") = nullptr, NoGil())
        .def("backup_terminal_value", &BackupManager::backup_terminal_value,
             py::arg("path"), py::arg("terminal_value"), py::arg("virtual_loss_manager") = nullptr, NoGil())
        .def("update_node_atomic", &BackupManager::update_node_atomic,
             py::arg("node_index"), py::arg("value_increment"), py::arg("visit_increment") = 1.0f, NoGil())
        .def("get_q_value", &BackupManager::get_q_value, NoGil())
        .def("validate_backup_path", &BackupManager::validate_backup_path, NoGil())
        .def("get_config", &BackupManager::get_config,
             py::return_value_policy::reference_internal, NoGil())
        .def("set_config", &BackupManager::set_config, NoGil())
        .def("get_statistics", &BackupManager::get_statistics, NoGil())
        .def("reset_statistics", &BackupManager::reset_statistics, NoGil());

    // Backup Statistics
    py::class_<BackupManager::BackupStats>(m, "BackupStats")
        .def_readonly("total_backups", &BackupManager::BackupStats::total_backups)
        .def_readonly("successful_backups", &BackupManager::BackupStats::successful_backups)
        .def_readonly("total_nodes_updated", &BackupManager::BackupStats::total_nodes_updated)
        .def_readonly("path_validation_failures", &BackupManager::BackupStats::path_validation_failures)
        .def_readonly("avg_path_length", &BackupManager::BackupStats::avg_path_length)
        .def_readonly("avg_absolute_leaf_value", &BackupManager::BackupStats::avg_absolute_leaf_value);

    // Backup Guard (RAII wrapper)
    py::class_<BackupGuard>(m, "BackupGuard")
        .def(py::init<BackupManager&, VirtualLossManager&, const std::vector<NodeIndex>&, float>())
        .def("was_successful", &BackupGuard::was_successful)
        .def("get_result", &BackupGuard::get_result, py::return_value_policy::reference_internal)
        .def("cleanup", &BackupGuard::cleanup);

    // Factory functions
    m.def("create_test_tree", &create_test_tree, py::arg("max_nodes") = 1000,
          "Create a test MCTS tree with basic nodes");

    m.def("create_test_virtual_loss_manager", &create_test_virtual_loss_manager,
          py::arg("tree"), py::arg("config") = VirtualLossConfig(),
          "Create a virtual loss manager for the given tree");

    m.def("create_puct_selector", &create_puct_selector,
          py::arg("config") = PUCTConfig(),
          "Create a PUCT selector with given configuration");

    m.def("create_backup_manager", &create_backup_manager,
          py::arg("tree"), py::arg("config") = BackupConfig(),
          "Create a backup manager for the given tree");

    // InferenceCallback - Abstract base class
    py::class_<InferenceCallback>(m, "InferenceCallback",
        "Abstract base class for neural network inference callbacks")
        .def("request_inference", &InferenceCallback::request_inference,
             py::arg("state"),
             "Request neural network inference for a game state");

    // PyInferenceCallback - Python callable wrapper
    py::class_<PyInferenceCallback, InferenceCallback>(m, "PyInferenceCallback",
        "Python inference callback wrapper for MCTS simulation runner.\n\n"
        "Wraps a Python callable to make it usable as an inference callback in C++.\n"
        "The callable should have signature: (state: IGameState) -> tuple[list[float], float]\n\n"
        "Example:\n"
        "    def my_inference(state):\n"
        "        policy = [0.1, 0.2, ...]  # Probability distribution\n"
        "        value = 0.5                # Position evaluation\n"
        "        return (policy, value)\n\n"
        "    callback = mcts_py.PyInferenceCallback(my_inference)")
        .def(py::init<py::object>(),
             py::arg("python_fn"),
             "Construct callback with a Python callable");

    // SimulationRunner - Phase 2 implementation complete
    py::class_<SimulationRunner>(m, "SimulationRunner",
        "High-performance MCTS simulation runner (C++ implementation).\n\n"
        "Executes complete MCTS simulations with GIL released, enabling true parallel search.\n"
        "Performance: 30k-40k simulations/second with 8 threads.")
        .def(py::init<MCTSTree&, PUCTSelector&, BackupManager&, VirtualLossManager&>(),
             py::arg("tree"), py::arg("selector"), py::arg("backup"), py::arg("virtual_loss"),
             "Construct simulation runner with required MCTS components")
        .def("run_simulation",
             &SimulationRunner::run_simulation,
             py::arg("root_state"), py::arg("root_index"), py::arg("inference_fn"),
             "Run a single MCTS simulation (select → expand → backup) with GIL released.\n\n"
             "Args:\n"
             "    root_state: Game state at root position\n"
             "    root_index: Root node index in tree\n"
             "    inference_fn: InferenceCallback for neural network evaluation\n\n"
             "Returns:\n"
             "    bool: True if simulation completed successfully");

    // BatchInferenceCallback - for batched async inference
    py::class_<BatchInferenceCallback>(m, "BatchInferenceCallback",
        "Abstract base class for batched neural network inference")
        .def("batch_inference", &BatchInferenceCallback::batch_inference,
             py::arg("states"),
             "Evaluate multiple game states in a single batch.\n\n"
             "Args:\n"
             "    states: List of IGameState pointers\n\n"
             "Returns:\n"
             "    List of (policy, value) tuples");

    // PyBatchInferenceCallback - Python wrapper
    py::class_<PyBatchInferenceCallback, BatchInferenceCallback>(m, "PyBatchInferenceCallback",
        "Python wrapper for batch inference callbacks")
        .def(py::init<py::object>(),
             py::arg("python_fn"),
             "Construct batch callback with Python callable");

    // AsyncInferenceQueue - Non-blocking inference queue for async MCTS
    py::class_<InferenceRequest>(m, "InferenceRequest")
        .def(py::init<>())
        .def_readwrite("request_id", &InferenceRequest::request_id)
        .def_readwrite("node_index", &InferenceRequest::node_index)
        .def_property("state",
            [](InferenceRequest& req) -> IGameState* {
                return req.state.get();
            },
            [](InferenceRequest& req, std::unique_ptr<IGameState> state) {
                req.state = std::move(state);
            })
        .def("__repr__", [](const InferenceRequest& req) {
            return "<InferenceRequest id=" + std::to_string(req.request_id) +
                   " node=" + std::to_string(req.node_index) + ">";
        });

    py::class_<InferenceResult>(m, "InferenceResult")
        .def(py::init<>())
        .def_readwrite("request_id", &InferenceResult::request_id)
        .def_readwrite("policy", &InferenceResult::policy)
        .def_readwrite("value", &InferenceResult::value)
        .def("__repr__", [](const InferenceResult& res) {
            return "<InferenceResult id=" + std::to_string(res.request_id) +
                   " value=" + std::to_string(res.value) + ">";
        });

    py::class_<AsyncInferenceQueue>(m, "AsyncInferenceQueue",
             "Thread-safe async inference queue for decoupling simulation from GPU\n\n"
             "Allows simulation threads to submit inference requests without blocking,\n"
             "while a coordinator thread batches requests for efficient GPU inference.")
        .def(py::init<>(),
             "Create empty async inference queue")
        .def("submit_request",
             [](AsyncInferenceQueue& queue, IGameState* state,
                NodeIndex node_index, std::vector<NodeIndex> path) -> uint64_t {
                 // Clone the state to transfer ownership to queue
                 auto cloned_state = state->clone();
                 return queue.submit_request(std::move(cloned_state), node_index, std::move(path));
             },
             py::arg("state"), py::arg("node_index"), py::arg("path"),
             "Submit inference request (non-blocking)\n\n"
             "Args:\n"
             "    state: Game state to evaluate (ownership transferred)\n"
             "    node_index: Tree node to expand\n"
             "    path: Path from root to node\n\n"
             "Returns:\n"
             "    int: Unique request ID for retrieving result")
        .def("collect_batch",
             &AsyncInferenceQueue::collect_batch,
             py::arg("min_batch_size"), py::arg("timeout_ms"),
             NoGil(),
             "Collect batch of pending requests\n\n"
             "Returns when either min_batch_size requests available OR timeout elapsed.\n\n"
             "Args:\n"
             "    min_batch_size: Minimum batch size to wait for\n"
             "    timeout_ms: Maximum wait time in milliseconds\n\n"
             "Returns:\n"
             "    list[InferenceRequest]: Batch of requests to process")
        .def("submit_results",
             &AsyncInferenceQueue::submit_results,
             py::arg("results"),
             "Submit batch of inference results\n\n"
             "Args:\n"
             "    results: List of InferenceResult matching collected requests")
        .def("try_get_result",
             &AsyncInferenceQueue::try_get_result,
             py::arg("request_id"),
             "Try to retrieve result for a request (non-blocking)\n\n"
             "Args:\n"
             "    request_id: Request ID from submit_request()\n\n"
             "Returns:\n"
             "    InferenceResult or None if not available")
        .def("has_results",
             &AsyncInferenceQueue::has_results,
             "Check if any results are available\n\n"
             "Returns:\n"
             "    bool: True if results available for retrieval")
        .def("pending_count",
             &AsyncInferenceQueue::pending_count,
             "Get number of pending requests\n\n"
             "Returns:\n"
             "    int: Number of requests waiting for inference")
        .def("results_count",
             &AsyncInferenceQueue::results_count,
             "Get number of completed results\n\n"
             "Returns:\n"
             "    int: Number of results available for retrieval")
        .def("get_memory_usage",
             &AsyncInferenceQueue::get_memory_usage,
             "Get memory usage estimate in bytes\n\n"
             "Returns:\n"
             "    int: Estimated memory usage");

    // Instrumentation controls
    m.def("set_instrumentation_enabled",
          [](bool enabled) {
              Instrumentation::instance().set_enabled(enabled);
          },
          py::arg("enabled"),
          "Enable or disable instrumentation metrics collection");

    m.def("reset_instrumentation_metrics",
          []() {
              Instrumentation::instance().reset();
          },
          "Reset instrumentation counters and timers");

    m.def("get_instrumentation_snapshot",
          []() {
              py::dict result;
              const auto snapshot = Instrumentation::instance().snapshot();
              for (const auto& entry : snapshot) {
                  const auto metric = entry.first;
                  const auto& data = entry.second;
                  py::dict payload;
                  payload["calls"] = data.call_count;
                  payload["total_ns"] = py::int_(data.total_elapsed_ns);
                  const double avg_ns = (data.call_count > 0)
                      ? static_cast<double>(data.total_elapsed_ns) / static_cast<double>(data.call_count)
                      : 0.0;
                  payload["avg_ns"] = avg_ns;
                  result[py::str(metric_to_string(metric))] = payload;
              }
              return result;
          },
          "Get snapshot of instrumentation metrics as a dictionary");

    // ContinuousSimulationRunner - Async MCTS runner for 30k+ sims/sec
    py::class_<ContinuousSimulationRunner, SimulationRunner>(m, "ContinuousSimulationRunner",
             "Continuous MCTS simulation runner with async inference\n\n"
             "Runs simulations continuously without blocking on neural network inference.\n"
             "Achieves 30,000+ sims/sec by decoupling simulation threads from GPU latency.")
        .def(py::init<MCTSTree&, PUCTSelector&, BackupManager&, VirtualLossManager&>(),
             py::arg("tree"), py::arg("selector"), py::arg("backup"), py::arg("virtual_loss"),
             "Create continuous simulation runner\n\n"
             "Args:\n"
             "    tree: Shared MCTS tree\n"
             "    selector: PUCT selector\n"
             "    backup: Backup manager\n"
             "    virtual_loss: Virtual loss manager")
        .def("run_continuous",
             &ContinuousSimulationRunner::run_continuous,
             py::arg("root_state"), py::arg("root_index"), py::arg("queue"), py::arg("num_simulations"),
             NoGil(),
             "Run continuous MCTS simulations with async inference\n\n"
             "Simulations run in a continuous loop without blocking on inference:\n"
             "1. Select to leaf (fast C++ tree traversal)\n"
             "2. Submit inference request to queue (non-blocking)\n"
             "3. Immediately start next simulation\n"
             "4. Process completed results asynchronously\n"
             "5. Continue until quota reached\n\n"
             "Performance: 30,000+ sims/sec with 8-12 threads\n\n"
             "Args:\n"
             "    root_state: Game state at root\n"
             "    root_index: Root node index\n"
             "    queue: AsyncInferenceQueue for request/result exchange\n"
             "    num_simulations: Number of simulations to complete\n\n"
             "Returns:\n"
             "    int: Number of successfully completed simulations");

    // BatchInferenceCoordinator - Background batching coordinator
    py::class_<BatchInferenceCoordinator>(m, "BatchInferenceCoordinator",
             "Background coordinator for batched GPU inference\n\n"
             "Spawns a background thread that continuously collects inference requests\n"
             "from AsyncInferenceQueue, batches them, calls Python for GPU inference\n"
             "(single GIL crossing per batch), and distributes results back.\n\n"
             "This reduces GIL time from >50% to <30% by batching all GPU calls.")
        .def(py::init<>(),
             "Create batch inference coordinator (thread not started)")
        .def("start",
             &BatchInferenceCoordinator::start,
             py::arg("queue"), py::arg("callback"), py::arg("batch_size"), py::arg("timeout_ms"),
             "Start background coordinator thread\n\n"
             "Args:\n"
             "    queue: AsyncInferenceQueue for request/result exchange\n"
             "    callback: BatchInferenceCallback for GPU inference\n"
             "    batch_size: Minimum batch size (e.g., 32)\n"
             "    timeout_ms: Maximum wait time for batch (e.g., 2.0ms)")
        .def("stop",
             &BatchInferenceCoordinator::stop,
             "Stop background thread and wait for completion")
        .def("is_running",
             &BatchInferenceCoordinator::is_running,
             "Check if coordinator is running\n\n"
             "Returns:\n"
             "    bool: True if coordinator thread is active");
}

} // namespace python
} // namespace mcts
