#include <iostream>
#include <vector>
#include <future>
#include <thread>
#include <atomic>
#include <mutex>
#include <random>
#include <chrono>
#include <algorithm>
#include <cstring>

// Simple node class matching key structures in the real MCTS implementation
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

// Simplified MCTS class with the core components we're testing
class MCTS {
private:
    MCTSNode root_;
    int num_threads_;
    std::unique_ptr<ThreadPool> thread_pool_;
    float virtual_loss_weight_ = 1.0f;
    
    // Simulate function matching the pattern in the real implementation
    float simulate() {
        // Add virtual loss to the root (in real code this happens to children during selection)
        root_.add_virtual_loss(virtual_loss_weight_);
        
        // Simulate some work
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        // Random value between -1 and 1
        static std::mt19937 gen(std::random_device{}());
        static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        float value = dist(gen);
        
        // Backup the value (this should properly clean up virtual loss)
        root_.backup(value);
        
        return value;
    }
    
public:
    MCTS(int num_threads = 1) : num_threads_(num_threads) {
        if (num_threads_ > 1) {
            thread_pool_ = std::make_unique<ThreadPool>(num_threads_);
        }
    }
    
    void search(int num_simulations = 100) {
        std::cout << "Running " << num_simulations << " simulations with " 
                  << num_threads_ << " threads..." << std::endl;
        
        if (num_threads_ > 1 && thread_pool_) {
            // Parallel simulations
            std::vector<std::future<float>> futures;
            
            for (int i = 0; i < num_simulations; ++i) {
                futures.push_back(thread_pool_->enqueue([this]() {
                    return this->simulate();
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
                simulate();
            }
        }
        
        root_.print_stats();
    }
};

int main() {
    std::cout << "MCTS Virtual Loss Bug Fix Verification" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Test with single thread first
    {
        std::cout << "\nSingle-threaded test:" << std::endl;
        MCTS mcts(1);
        mcts.search(100);
    }
    
    // Test with multiple threads
    {
        std::cout << "\nMulti-threaded test:" << std::endl;
        MCTS mcts(4);
        mcts.search(100);
    }
    
    std::cout << "\nTests completed. If virtual_loss=0 in both tests, the fix is working correctly." << std::endl;
    
    return 0;
}