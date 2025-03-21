// Create file: alphazero/core/mcts/batch_evaluator.h

#pragma once

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>
#include <future>
#include <atomic>
#include <iostream>

namespace alphazero {

/**
 * Batched evaluator for efficient neural network inference.
 * Collects positions to evaluate and processes them in batches.
 */
class BatchEvaluator {
public:
    /**
     * Constructor.
     * 
     * @param batch_size Maximum batch size for evaluation
     * @param max_wait_ms Maximum time to wait for a full batch (milliseconds)
     */
    BatchEvaluator(int batch_size = 16, int max_wait_ms = 5) 
        : batch_size_(batch_size), max_wait_ms_(max_wait_ms), running_(true) {
        
        // Start worker thread
        worker_thread_ = std::thread(&BatchEvaluator::process_queue, this);
    }
    
    /**
     * Destructor.
     */
    virtual ~BatchEvaluator() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            running_ = false;
            queue_cv_.notify_all();
        }
        
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }
    
    /**
     * Submit a position for evaluation.
     * 
     * @param state Position state representation
     * @return Future with evaluation result (policy, value)
     */
    std::future<std::pair<std::vector<float>, float>> submit(const std::vector<float>& state) {
        // Create a promise and future pair
        auto promise = std::make_shared<std::promise<std::pair<std::vector<float>, float>>>();
        std::future<std::pair<std::vector<float>, float>> future = promise->get_future();
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            eval_queue_.push({state, promise});
            
            if (eval_queue_.size() >= batch_size_) {
                queue_cv_.notify_one();
            }
        }
        
        return future;
    }
    
    /**
     * Evaluate a batch of states. To be implemented by derived classes.
     * 
     * @param states Vector of state tensors
     * @return Vector of (policy, value) pairs
     */
    virtual std::vector<std::pair<std::vector<float>, float>> evaluate_batch(
        const std::vector<std::vector<float>>& states) = 0;
    
private:
    // Evaluation queue item
    struct QueueItem {
        std::vector<float> state;
        std::shared_ptr<std::promise<std::pair<std::vector<float>, float>>> promise;
    };
    
    // Evaluation parameters
    int batch_size_;
    int max_wait_ms_;
    std::atomic<bool> running_;
    
    // Evaluation queue
    std::queue<QueueItem> eval_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Worker thread
    std::thread worker_thread_;
    
    /**
     * Process the evaluation queue in batches.
     */
    void process_queue() {
        while (running_) {
            std::vector<QueueItem> batch;
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                
                // Wait for items or until timeout
                queue_cv_.wait_for(lock, std::chrono::milliseconds(max_wait_ms_), 
                    [this]() { return !eval_queue_.empty() || !running_; });
                
                if (!running_) break;
                
                // Check if we have items to process
                if (eval_queue_.empty()) continue;
                
                // Take items from the queue
                int count = std::min(static_cast<int>(eval_queue_.size()), batch_size_);
                for (int i = 0; i < count; ++i) {
                    batch.push_back(eval_queue_.front());
                    eval_queue_.pop();
                }
            }
            
            if (!batch.empty()) {
                // Extract states for batch evaluation
                std::vector<std::vector<float>> states;
                for (const auto& item : batch) {
                    states.push_back(item.state);
                }
                
                // Evaluate batch
                std::vector<std::pair<std::vector<float>, float>> results;
                try {
                    results = evaluate_batch(states);
                } catch (const std::exception& e) {
                    // On error, set empty results
                    results.clear();
                }
                
                // Set promises with results
                for (size_t i = 0; i < batch.size(); ++i) {
                    try {
                        if (i < results.size()) {
                            batch[i].promise->set_value(results[i]);
                        } else {
                            // Error case - set empty result
                            batch[i].promise->set_value({{}, 0.0f});
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "Error: " << e.what() << std::endl;
                    }
                }
            }
        }
    }
};

} // namespace alphazero