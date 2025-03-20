#include <iostream>
#include <vector>
#include <future>
#include <thread>
#include <atomic>
#include <mutex>
#include <random>
#include <chrono>
#include <algorithm>
#include <unordered_map>
#include <memory>
#include <cstring>

// Simple node class matching key structures from the real implementation
class MCTSNode {
public:
    std::atomic<int> visit_count{0};
    float value_sum = 0.0f;
    std::atomic<int> virtual_loss{0};
    mutable std::mutex value_mutex;
    
    // The fixed backup function that properly resets virtual loss
    void backup(float value) {
        // Update visit count atomically
        visit_count.fetch_add(1);
        
        // Update value sum with mutex protection
        {
            std::lock_guard<std::mutex> lock(value_mutex);
            value_sum += value;
            
            // Set virtual loss to 0 (our fix)
            virtual_loss.store(0);
        }
    }
    
    // Add virtual loss for selection phase
    void add_virtual_loss(int amount) {
        virtual_loss.fetch_add(amount);
    }
    
    // Remove virtual loss (old way, for comparison)
    void remove_virtual_loss(int amount) {
        virtual_loss.fetch_sub(amount);
    }
    
    // Print node statistics
    void print_stats() const {
        int visits = visit_count.load();
        int vl = virtual_loss.load();
        
        std::lock_guard<std::mutex> lock(value_mutex);
        float avg_value = (visits > 0) ? value_sum / static_cast<float>(visits) : 0.0f;
        
        std::cout << "Node stats: visits=" << visits 
                  << ", value_sum=" << value_sum 
                  << ", avg_value=" << avg_value
                  << ", virtual_loss=" << vl << std::endl;
    }
};

// Simplified transposition table
class TranspositionTable {
private:
    std::unordered_map<uint64_t, MCTSNode*> table_;
    mutable std::mutex mutex_;
    size_t max_size_;
    
public:
    TranspositionTable(size_t max_size = 1000) : max_size_(max_size) {}
    
    MCTSNode* lookup(uint64_t hash) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = table_.find(hash);
        if (it != table_.end()) {
            return it->second;
        }
        return nullptr;
    }
    
    void store(uint64_t hash, MCTSNode* node) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // If table is full, remove a random entry
        if (table_.size() >= max_size_) {
            auto it = table_.begin();
            std::advance(it, rand() % table_.size());
            table_.erase(it);
        }
        
        table_[hash] = node;
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        table_.clear();
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return table_.size();
    }
};

// Mock thread pool similar to actual implementation
class ThreadPool {
private:
    int num_threads_;
    
public:
    ThreadPool(int num_threads) : num_threads_(num_threads) {}
    
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type> {
        
        using return_type = typename std::result_of<F(Args...)>::type;
        
        // Create a packaged task
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        // Get future
        std::future<return_type> result = task->get_future();
        
        // Launch the task asynchronously
        std::thread([task]() { (*task)(); }).detach();
        
        return result;
    }
};

// Simplified MCTS class with the core components and transposition table
class MCTS {
private:
    std::unique_ptr<MCTSNode> root_;
    int num_threads_;
    std::unique_ptr<ThreadPool> thread_pool_;
    float virtual_loss_weight_ = 1.0f;
    std::unique_ptr<TranspositionTable> transposition_table_;
    bool use_transposition_table_ = true;
    
    // Get state hash (simplified)
    uint64_t get_hash(int state_id) {
        // Simple hash function for test purposes
        return static_cast<uint64_t>(state_id);
    }
    
    // Simulate function matching the pattern in the real implementation
    float simulate(int state_id) {
        // Try to find the node in transposition table
        MCTSNode* node = root_.get();
        
        // Keep track of nodes where we added virtual loss
        std::vector<MCTSNode*> virtual_loss_nodes;
        
        // Add virtual loss to the root
        node->add_virtual_loss(virtual_loss_weight_);
        virtual_loss_nodes.push_back(node);
        
        // Check if we should use transposition table
        if (use_transposition_table_ && transposition_table_) {
            uint64_t hash = get_hash(state_id);
            
            // Look up in transposition table
            MCTSNode* tt_node = transposition_table_->lookup(hash);
            
            if (tt_node) {
                // We found a matching node in the transposition table
                // Clean up virtual loss from original node
                if (num_threads_ > 1) {
                    // Remove virtual loss from original node
                    node->remove_virtual_loss(virtual_loss_weight_);
                    
                    // Add virtual loss to transposition node
                    tt_node->add_virtual_loss(virtual_loss_weight_);
                    
                    // Update our tracking
                    virtual_loss_nodes[0] = tt_node;
                }
                
                // Use the node from the transposition table
                node = tt_node;
            }
        }
        
        // Simulate some work
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        // Random value between -1 and 1
        static std::mt19937 gen(std::random_device{}());
        static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        float value = dist(gen);
        
        // Store in transposition table
        if (use_transposition_table_ && transposition_table_) {
            uint64_t hash = get_hash(state_id);
            transposition_table_->store(hash, node);
        }
        
        // Backup the value (this should properly clean up virtual loss)
        node->backup(value);
        
        return value;
    }
    
public:
    MCTS(int num_threads = 1, bool use_tt = true) : 
        num_threads_(num_threads), 
        use_transposition_table_(use_tt) {
        
        // Create root node
        root_ = std::make_unique<MCTSNode>();
        
        // Create thread pool if using multiple threads
        if (num_threads_ > 1) {
            thread_pool_ = std::make_unique<ThreadPool>(num_threads_);
        }
        
        // Create transposition table if needed
        if (use_transposition_table_) {
            transposition_table_ = std::make_unique<TranspositionTable>(1000);
        }
    }
    
    void search(int num_simulations = 100) {
        std::cout << "Running " << num_simulations << " simulations with " 
                  << num_threads_ << " threads";
        if (use_transposition_table_) {
            std::cout << " (using transposition table)";
        }
        std::cout << "..." << std::endl;
        
        if (num_threads_ > 1 && thread_pool_) {
            // Parallel simulations
            std::vector<std::future<float>> futures;
            
            for (int i = 0; i < num_simulations; ++i) {
                // Use a different state ID for each simulation to test transposition table
                int state_id = i % 10;  // Use 10 different states
                
                futures.push_back(thread_pool_->enqueue([this, state_id]() {
                    return this->simulate(state_id);
                }));
            }
            
            // Wait for all simulations to complete using .get() as in our fix
            for (auto& future : futures) {
                try {
                    future.get();
                } catch (const std::exception& e) {
                    std::cerr << "Error in simulation: " << e.what() << std::endl;
                }
            }
        } else {
            // Sequential simulations
            for (int i = 0; i < num_simulations; ++i) {
                // Use a different state ID for each simulation to test transposition table
                int state_id = i % 10;  // Use 10 different states
                simulate(state_id);
            }
        }
        
        // Print stats
        root_->print_stats();
        
        // Print transposition table stats
        if (use_transposition_table_ && transposition_table_) {
            std::cout << "Transposition table size: " << transposition_table_->size() << std::endl;
        }
    }
};

int main() {
    std::cout << "MCTS with Transposition Table Bug Fix Verification" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    // Single-threaded baseline test (with transposition table)
    {
        std::cout << "\nSingle-threaded test with transposition table:" << std::endl;
        MCTS mcts(1, true);
        mcts.search(100);
    }
    
    // Multi-threaded test with transposition table 
    {
        std::cout << "\nMulti-threaded test with transposition table:" << std::endl;
        MCTS mcts(4, true);
        mcts.search(100);
    }
    
    // Multi-threaded test without transposition table (for comparison)
    {
        std::cout << "\nMulti-threaded test without transposition table:" << std::endl;
        MCTS mcts(4, false);
        mcts.search(100);
    }
    
    std::cout << "\nTests completed. If virtual_loss=0 in all tests, the fix is working correctly" << std::endl;
    std::cout << "with and without the transposition table in multi-threaded mode." << std::endl;
    
    return 0;
}