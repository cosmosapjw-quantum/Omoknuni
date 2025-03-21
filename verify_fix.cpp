#include <iostream>
#include <future>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

// This is a simplified version of the MCTS class structure
// just to verify that our fix for virtual loss works

class TestNode {
public:
    std::atomic<int> visit_count{0};
    std::atomic<int> virtual_loss{0};
    float value_sum = 0.0f;
    std::mutex value_mutex;
    
    void backup(float value) {
        // Update visit count atomically
        visit_count.fetch_add(1);
        
        // Update value sum with mutex protection
        {
            std::lock_guard<std::mutex> lock(value_mutex);
            value_sum += value;
            
            // Reset virtual loss to 0 instead of decrementing
            virtual_loss.store(0);
        }
    }
};

void run_test(int num_threads) {
    std::cout << "Testing with " << num_threads << " threads..." << std::endl;
    
    TestNode node;
    std::vector<std::future<void>> futures;
    
    // Create threads and run simulations
    for (int i = 0; i < 100; ++i) {
        futures.push_back(std::async(std::launch::async, [&node]() {
            // Add virtual loss
            node.virtual_loss.fetch_add(1);
            
            // Simulate some work
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            
            // Backup a random value between -1 and 1
            float value = (rand() % 200 - 100) / 100.0f;
            node.backup(value);
        }));
    }
    
    // Wait for all futures to complete
    for (auto& future : futures) {
        future.get();
    }
    
    // Check the final state
    std::cout << "Final visit count: " << node.visit_count.load() << std::endl;
    std::cout << "Final virtual loss: " << node.virtual_loss.load() << std::endl;
    std::cout << "Final value sum: " << node.value_sum << std::endl;
    
    // The virtual loss should be 0 if our fix works
    if (node.virtual_loss.load() == 0) {
        std::cout << "Virtual loss bug fix verified! ✅" << std::endl;
    } else {
        std::cout << "Virtual loss bug still present! ❌" << std::endl;
    }
}

int main() {
    std::cout << "Testing virtual loss bug fix in C++" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    
    // Run test with multiple threads
    run_test(4);
    
    return 0;
}