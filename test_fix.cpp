#include <iostream>
#include <thread>
#include <vector>
#include <future>
#include <functional>
#include <cmath>
#include <random>
#include <cstring>

// This is a minimal test case that demonstrates the fix for the MCTS threading bug
// It simulates the core of the MCTS algorithm focusing on the threading and virtual loss aspects

class MockNode {
public:
    std::atomic<int> visit_count;
    float value_sum;
    std::atomic<int> virtual_loss;
    std::mutex value_mutex;
    
    MockNode() : visit_count(0), value_sum(0.0f), virtual_loss(0) {}
    
    void add_virtual_loss(int amount) {
        virtual_loss.fetch_add(amount);
    }
    
    void remove_virtual_loss(int amount) {
        virtual_loss.fetch_sub(amount);
    }
    
    void backup(float value) {
        // Update visit count atomically
        visit_count.fetch_add(1);
        
        // Update value sum with mutex protection
        {
            std::lock_guard<std::mutex> lock(value_mutex);
            value_sum += value;
            
            // Reset virtual loss to 0 (our fix)
            virtual_loss.store(0);
        }
    }
    
    float get_value() const {
        int visits = visit_count.load();
        if (visits == 0) return 0.0f;
        
        // Need a non-const reference to mutex for lock_guard
        // Using const_cast because the method is const but we need to lock
        std::lock_guard<std::mutex> lock(*const_cast<std::mutex*>(&value_mutex));
        return value_sum / static_cast<float>(visits);
    }
};

class MockMCTS {
private:
    int num_threads_;
    std::vector<MockNode> nodes_;
    std::unique_ptr<std::thread[]> threads_;
    
    float simulate(int node_idx) {
        // Add virtual loss to the node
        nodes_[node_idx].add_virtual_loss(1);
        
        // Simulate some work
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        
        // Generate a random value between -1 and 1
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        float value = dist(gen);
        
        // Backup the value (this should also handle virtual loss removal)
        nodes_[node_idx].backup(value);
        
        return value;
    }
    
public:
    MockMCTS(int num_threads, int num_nodes) 
        : num_threads_(num_threads), nodes_(num_nodes) {}
    
    void run_parallel_simulations(int num_simulations) {
        if (num_threads_ > 1) {
            // Create futures to hold results
            std::vector<std::future<float>> futures;
            
            for (int i = 0; i < num_simulations; ++i) {
                // Choose a random node to simulate
                int node_idx = i % nodes_.size();
                
                // Use a thread pool or similar to enqueue the task
                auto future = std::async(std::launch::async, 
                    [this, node_idx]() { return this->simulate(node_idx); });
                
                futures.push_back(std::move(future));
            }
            
            // Wait for all futures to complete
            for (auto& future : futures) {
                try {
                    // Using get() instead of wait() to ensure completion
                    future.get();
                } catch (const std::exception& e) {
                    std::cerr << "Error in simulation: " << e.what() << std::endl;
                }
            }
        } else {
            // Sequential simulation
            for (int i = 0; i < num_simulations; ++i) {
                int node_idx = i % nodes_.size();
                simulate(node_idx);
            }
        }
    }
    
    void print_node_stats() {
        for (size_t i = 0; i < nodes_.size(); ++i) {
            const auto& node = nodes_[i];
            std::cout << "Node " << i 
                      << ": visits=" << node.visit_count.load()
                      << ", value=" << node.get_value()
                      << ", virtual_loss=" << node.virtual_loss.load() 
                      << std::endl;
        }
    }
};

int main() {
    // Test with a single thread first
    {
        std::cout << "Testing with 1 thread..." << std::endl;
        MockMCTS mcts(1, 5);  // 1 thread, 5 nodes
        mcts.run_parallel_simulations(100);
        mcts.print_node_stats();
    }
    
    // Test with multiple threads
    {
        std::cout << "\nTesting with 4 threads..." << std::endl;
        MockMCTS mcts(4, 5);  // 4 threads, 5 nodes
        mcts.run_parallel_simulations(100);
        mcts.print_node_stats();
    }
    
    return 0;
}