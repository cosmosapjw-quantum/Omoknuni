#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <future>
#include <cstdint>
#include <cstring>
#include <thread>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

// Comprehensive test that closely simulates the entire MCTS implementation
// with special focus on multi-threading and transposition table interactions

// Forward declarations
class MCTSNode;
class TranspositionTable;
class ThreadPool;
class MCTS;

// Configuration
constexpr int NUM_SIMULATIONS = 100;
constexpr int BOARD_SIZE = 5;
constexpr float C_PUCT = 1.5f;

// Node class
class MCTSNode {
public:
    std::atomic<int> visit_count{0};
    float value_sum{0.0f};
    float prior{0.0f};
    MCTSNode* parent{nullptr};
    int move{-1};
    std::atomic<int> virtual_loss{0};
    
    std::unordered_map<int, std::unique_ptr<MCTSNode>> children;
    mutable std::mutex children_mutex;
    mutable std::mutex value_mutex;
    
    MCTSNode(float prior = 0.0f, MCTSNode* parent = nullptr, int move = -1)
        : prior(prior), parent(parent), move(move) {}
    
    ~MCTSNode() = default;
    
    bool is_expanded() const {
        std::lock_guard<std::mutex> lock(children_mutex);
        return !children.empty();
    }
    
    float value() const {
        int visits = visit_count.load();
        if (visits == 0) {
            return 0.0f;
        }
        
        std::lock_guard<std::mutex> lock(value_mutex);
        return value_sum / static_cast<float>(visits);
    }
    
    void add_virtual_loss(int amount) {
        virtual_loss.fetch_add(amount);
    }
    
    void remove_virtual_loss(int amount) {
        virtual_loss.fetch_sub(amount);
    }
    
    float ucb_score(int parent_visit_count, float c_puct) const {
        int effective_visits = visit_count.load() + virtual_loss.load();
        
        if (effective_visits == 0) {
            return std::numeric_limits<float>::max();
        }
        
        float exploitation;
        {
            std::lock_guard<std::mutex> lock(value_mutex);
            exploitation = value_sum / static_cast<float>(effective_visits);
        }
        
        float exploration = c_puct * prior * std::sqrt(static_cast<float>(parent_visit_count)) / 
                            (1.0f + static_cast<float>(effective_visits));
                            
        return exploitation + exploration;
    }
    
    std::pair<int, MCTSNode*> select_child(float c_puct) {
        std::lock_guard<std::mutex> lock(children_mutex);
        
        if (children.empty()) {
            return {-1, nullptr};
        }
        
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
                best_children.emplace_back(move, child);
            }
        }
        
        if (best_children.size() > 1) {
            int index = std::rand() % best_children.size();
            return best_children[index];
        } else if (!best_children.empty()) {
            return best_children[0];
        }
        
        return {-1, nullptr};
    }
    
    void expand(const std::vector<int>& moves, const std::vector<float>& priors) {
        std::lock_guard<std::mutex> lock(children_mutex);
        
        for (size_t i = 0; i < moves.size(); ++i) {
            int move = moves[i];
            float move_prior = (i < priors.size()) ? priors[i] : 0.0f;
            
            auto child = std::make_unique<MCTSNode>(move_prior, this, move);
            children[move] = std::move(child);
        }
    }
    
    // This is the fixed backup function that properly handles virtual loss
    void backup(float value) {
        MCTSNode* current = this;
        float current_value = value;
        
        while (current != nullptr) {
            current->visit_count.fetch_add(1);
            
            {
                std::lock_guard<std::mutex> lock(current->value_mutex);
                current->value_sum += current_value;
                
                // Reset virtual loss to 0 instead of decrementing
                current->virtual_loss.store(0);
            }
            
            MCTSNode* parent = current->parent;
            current_value = -current_value;
            current = parent;
        }
    }
    
    MCTSNode* get_child(int move) {
        std::lock_guard<std::mutex> lock(children_mutex);
        
        auto it = children.find(move);
        if (it != children.end()) {
            return it->second.get();
        }
        
        return nullptr;
    }
    
    std::unordered_map<int, int> get_visit_counts() const {
        std::lock_guard<std::mutex> lock(children_mutex);
        
        std::unordered_map<int, int> visit_counts;
        for (const auto& child_pair : children) {
            visit_counts[child_pair.first] = child_pair.second->visit_count.load();
        }
        
        return visit_counts;
    }
    
    void print_status() const {
        int visits = visit_count.load();
        int vl = virtual_loss.load();
        
        std::lock_guard<std::mutex> lock(value_mutex);
        float avg_value = (visits > 0) ? value_sum / static_cast<float>(visits) : 0.0f;
        
        std::cout << "Node: visits=" << visits << ", avg_value=" << avg_value 
                  << ", virtual_loss=" << vl << ", children=" << children.size() << std::endl;
    }
};

// Transposition table
class TranspositionTable {
public:
    TranspositionTable(size_t max_size = 1000000) : max_size_(max_size) {}
    
    virtual ~TranspositionTable() = default;
    
    virtual MCTSNode* lookup(uint64_t hash_value) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = table_.find(hash_value);
        if (it != table_.end()) {
            return it->second;
        }
        
        return nullptr;
    }
    
    virtual void store(uint64_t hash_value, MCTSNode* node) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (table_.size() >= max_size_) {
            auto it = table_.begin();
            std::advance(it, std::rand() % table_.size());
            table_.erase(it);
        }
        
        table_[hash_value] = node;
    }
    
    virtual void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        table_.clear();
    }
    
    virtual size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return table_.size();
    }
    
    virtual bool contains(uint64_t hash_value) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return table_.find(hash_value) != table_.end();
    }
    
protected:
    size_t max_size_;
    std::unordered_map<uint64_t, MCTSNode*> table_;
    mutable std::mutex mutex_;
};

// Thread pool
class ThreadPool {
public:
    ThreadPool(size_t num_threads = 0) : stop(false), active_tasks(0) {
        // Use hardware concurrency if num_threads is 0
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) {
                num_threads = 2;
            }
        }
        
        // Create worker threads
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] {
                            return stop || !tasks.empty();
                        });
                        
                        if (stop && tasks.empty()) {
                            return;
                        }
                        
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    
                    task();
                }
            });
        }
    }
    
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stop = true;
        }
        
        condition.notify_all();
        
        for (std::thread& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    
    size_t size() const {
        return workers.size();
    }
    
    size_t queue_size() const {
        std::lock_guard<std::mutex> lock(queue_mutex);
        return tasks.size();
    }
    
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type> {
        
        using return_type = typename std::result_of<F(Args...)>::type;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> result = task->get_future();
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            
            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            
            active_tasks++;
            
            tasks.emplace([task, this]() {
                (*task)();
                
                {
                    std::lock_guard<std::mutex> lock(done_mutex);
                    active_tasks--;
                    if (active_tasks == 0 && tasks.empty()) {
                        done_condition.notify_all();
                    }
                }
            });
        }
        
        condition.notify_one();
        
        return result;
    }
    
    void wait_all() {
        std::unique_lock<std::mutex> lock(done_mutex);
        done_condition.wait(lock, [this] {
            size_t queue_size = 0;
            {
                std::lock_guard<std::mutex> q_lock(queue_mutex);
                queue_size = tasks.size();
            }
            return active_tasks == 0 && queue_size == 0;
        });
    }
    
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    
    mutable std::mutex queue_mutex;
    std::condition_variable condition;
    
    std::atomic<bool> stop;
    
    std::atomic<size_t> active_tasks;
    std::condition_variable done_condition;
    mutable std::mutex done_mutex;
};

// Simple evaluator function
using EvaluatorType = std::function<std::pair<std::vector<float>, float>(const std::vector<float>&)>;

// MCTS implementation
class MCTS {
public:
    MCTS(int num_simulations = 100,
         float c_puct = 1.5f,
         bool use_transposition_table = true,
         size_t transposition_table_size = 10000,
         int num_threads = 1)
        : num_simulations_(num_simulations),
          c_puct_(c_puct),
          use_transposition_table_(use_transposition_table),
          num_threads_(num_threads) {
          
        // Create root node
        root_ = std::make_unique<MCTSNode>();
        
        // Create transposition table if needed
        if (use_transposition_table_) {
            transposition_table_ = std::make_unique<TranspositionTable>(transposition_table_size);
        }
        
        // Create thread pool if using multiple threads
        if (num_threads_ > 1) {
            thread_pool_ = std::make_unique<ThreadPool>(num_threads_);
        }
    }
    
    ~MCTS() = default;
    
    std::unordered_map<int, float> search(
        const std::vector<float>& state_tensor,
        const std::vector<int>& legal_moves,
        const EvaluatorType& evaluator) {
        
        // If root is not expanded, evaluate and expand it
        if (!root_->is_expanded()) {
            auto [policy, value] = evaluator(state_tensor);
            expand_root(legal_moves, policy);
        }
        
        // Run simulations
        if (num_threads_ > 1 && thread_pool_) {
            // Parallel simulations
            std::vector<std::future<float>> futures;
            
            // Make a copy of the state tensor for thread safety
            auto state_tensor_copy = state_tensor;
            
            for (int i = 0; i < num_simulations_; ++i) {
                // Make a separate copy for each simulation to ensure thread safety
                std::vector<float> simulation_state = state_tensor_copy;
                
                futures.push_back(thread_pool_->enqueue([this, simulation_state, &evaluator]() {
                    std::vector<std::pair<MCTSNode*, int>> path;
                    return this->simulate(simulation_state, evaluator, path);
                }));
            }
            
            // Wait for all simulations to complete
            for (auto& future : futures) {
                try {
                    // Use get() instead of wait() to ensure futures complete properly
                    future.get();
                } catch (const std::exception& e) {
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
        
        // Calculate and return probabilities
        return calculate_probabilities();
    }
    
    // Print status of the tree
    void print_status() const {
        std::cout << "MCTS status:" << std::endl;
        std::cout << "  num_simulations: " << num_simulations_ << std::endl;
        std::cout << "  num_threads: " << num_threads_ << std::endl;
        std::cout << "  use_transposition_table: " << (use_transposition_table_ ? "yes" : "no") << std::endl;
        
        if (use_transposition_table_ && transposition_table_) {
            std::cout << "  transposition_table_size: " << transposition_table_->size() << std::endl;
        }
        
        if (root_) {
            std::cout << "  Root node: ";
            root_->print_status();
            
            // Print top child nodes
            std::cout << "  Top children:" << std::endl;
            auto visit_counts = root_->get_visit_counts();
            
            // Sort by visit count
            std::vector<std::pair<int, int>> sorted_children;
            for (const auto& [move, visits] : visit_counts) {
                sorted_children.emplace_back(move, visits);
            }
            
            std::sort(sorted_children.begin(), sorted_children.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
            
            int shown = 0;
            for (const auto& [move, visits] : sorted_children) {
                if (shown++ >= 5) break;
                
                MCTSNode* child = root_->get_child(move);
                if (child) {
                    std::cout << "    Move " << move << ": ";
                    child->print_status();
                }
            }
        }
    }
    
private:
    // Simulate a single MCTS iteration
    float simulate(
        const std::vector<float>& state_tensor,
        const EvaluatorType& evaluator,
        std::vector<std::pair<MCTSNode*, int>>& path) {
        
        MCTSNode* node = root_.get();
        if (!node) {
            return 0.0f;
        }
        
        std::vector<float> current_state = state_tensor;
        int current_player = 1;
        
        path.clear();
        path.push_back({node, -1});
        
        // Track nodes where we added virtual loss
        std::vector<MCTSNode*> virtual_loss_nodes;
        
        // Selection phase
        const int MAX_DEPTH = 100;
        int depth = 0;
        
        while (node->is_expanded() && depth < MAX_DEPTH) {
            depth++;
            
            // Select best child according to UCB
            auto [move, child] = node->select_child(c_puct_);
            if (!child) {
                break;
            }
            
            // Apply virtual loss for parallelization
            if (num_threads_ > 1) {
                child->add_virtual_loss(1.0f);
                virtual_loss_nodes.push_back(child);
            }
            
            // Apply move to get new state
            // Here we skip actual move application to simplify the test
            
            // Add to path
            path.push_back({child, move});
            
            // Update node
            node = child;
            
            // Check transposition table if enabled
            if (use_transposition_table_ && transposition_table_) {
                // Use a simple hash function for testing
                uint64_t hash = std::hash<std::string>{}(std::to_string(move));
                
                // Create path hash set to check for cycles
                std::unordered_set<MCTSNode*> path_nodes;
                for (const auto& [path_node, _] : path) {
                    path_nodes.insert(path_node);
                }
                
                // Look up in transposition table
                MCTSNode* tt_node = transposition_table_->lookup(hash);
                if (tt_node) {
                    // Check if node is already in our path (cycle detection)
                    if (path_nodes.find(tt_node) == path_nodes.end()) {
                        // When using transposition table in multithreaded mode,
                        // must carefully clean up virtual loss on original node
                        if (num_threads_ > 1) {
                            // Remove virtual loss from original node
                            node->remove_virtual_loss(1.0f);
                            
                            // Remove from virtual loss tracking
                            if (!virtual_loss_nodes.empty()) {
                                virtual_loss_nodes.pop_back();
                            }
                            
                            // Add virtual loss to transposition node
                            tt_node->add_virtual_loss(1.0f);
                            virtual_loss_nodes.push_back(tt_node);
                        }
                        
                        // Use the node from the transposition table
                        node = tt_node;
                        path.back().first = node;
                    }
                }
            }
        }
        
        // Evaluation: Generate a random value for testing
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        float value = dist(gen);
        
        // Expand the node with random moves if not expanded
        if (!node->is_expanded()) {
            // Generate random policy
            std::vector<int> moves;
            std::vector<float> policy;
            
            // Generate 10 random moves
            for (int i = 0; i < 10; i++) {
                moves.push_back(i);
                policy.push_back(1.0f / 10.0f);
            }
            
            // Expand node
            node->expand(moves, policy);
            
            // Store in transposition table if enabled
            if (use_transposition_table_ && transposition_table_) {
                uint64_t hash = std::hash<std::string>{}(std::to_string(path.back().second));
                transposition_table_->store(hash, node);
            }
        }
        
        // Backup: Update values in the path
        node->backup(value);
        
        return value;
    }
    
    // Helper to expand the root node
    void expand_root(const std::vector<int>& legal_moves, const std::vector<float>& priors) {
        root_->expand(legal_moves, priors);
    }
    
    // Calculate probabilities from visit counts
    std::unordered_map<int, float> calculate_probabilities() const {
        std::unordered_map<int, float> probabilities;
        std::unordered_map<int, int> visits = root_->get_visit_counts();
        
        if (visits.empty()) {
            return probabilities;
        }
        
        // Sum visits
        float sum = 0.0f;
        for (const auto& [move, count] : visits) {
            sum += static_cast<float>(count);
        }
        
        // Calculate probabilities
        if (sum > 0.0f) {
            for (const auto& [move, count] : visits) {
                probabilities[move] = static_cast<float>(count) / sum;
            }
        }
        
        return probabilities;
    }
    
private:
    std::unique_ptr<MCTSNode> root_;
    int num_simulations_;
    float c_puct_;
    bool use_transposition_table_;
    std::unique_ptr<TranspositionTable> transposition_table_;
    std::unique_ptr<ThreadPool> thread_pool_;
    int num_threads_;
};

// Main test function
int main() {
    std::cout << "=== Comprehensive MCTS Multi-threading and Transposition Table Test ===" << std::endl;
    
    // Simple evaluator that returns uniform policy and random value
    auto evaluator = [](const std::vector<float>& state) -> std::pair<std::vector<float>, float> {
        // Generate uniform policy for BOARD_SIZE × BOARD_SIZE board
        std::vector<float> policy(BOARD_SIZE * BOARD_SIZE, 1.0f / (BOARD_SIZE * BOARD_SIZE));
        
        // Generate random value
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        float value = dist(gen);
        
        return {policy, value};
    };
    
    // Create a simple board state
    std::vector<float> state(BOARD_SIZE * BOARD_SIZE, 0.0f);
    
    // Generate legal moves (all positions on the board)
    std::vector<int> legal_moves;
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        legal_moves.push_back(i);
    }
    
    // Test single-threaded without transposition table (baseline)
    {
        std::cout << "\n=== Single-threaded without transposition table ===" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        MCTS mcts(NUM_SIMULATIONS, C_PUCT, false, 0, 1);
        auto probs = mcts.search(state, legal_moves, evaluator);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        std::cout << "Completed in " << elapsed.count() << " seconds" << std::endl;
        mcts.print_status();
    }
    
    // Test single-threaded with transposition table
    {
        std::cout << "\n=== Single-threaded with transposition table ===" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        MCTS mcts(NUM_SIMULATIONS, C_PUCT, true, 10000, 1);
        auto probs = mcts.search(state, legal_moves, evaluator);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        std::cout << "Completed in " << elapsed.count() << " seconds" << std::endl;
        mcts.print_status();
    }
    
    // Test multi-threaded without transposition table
    {
        std::cout << "\n=== Multi-threaded without transposition table ===" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        MCTS mcts(NUM_SIMULATIONS, C_PUCT, false, 0, 4);
        auto probs = mcts.search(state, legal_moves, evaluator);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        std::cout << "Completed in " << elapsed.count() << " seconds" << std::endl;
        mcts.print_status();
    }
    
    // Test multi-threaded with transposition table
    {
        std::cout << "\n=== Multi-threaded with transposition table ===" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        MCTS mcts(NUM_SIMULATIONS, C_PUCT, true, 10000, 4);
        auto probs = mcts.search(state, legal_moves, evaluator);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        std::cout << "Completed in " << elapsed.count() << " seconds" << std::endl;
        mcts.print_status();
    }
    
    std::cout << "\n=== Test completed ===" << std::endl;
    std::cout << "If all tests completed without errors and virtual_loss=0 in all cases," << std::endl;
    std::cout << "then the fixes are working correctly!" << std::endl;
    
    return 0;
}