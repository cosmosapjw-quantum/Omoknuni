<<<<<<< HEAD
// Create file: alphazero/core/mcts/batch_evaluator.h

=======
>>>>>>> 42bb511ab1410a992c3fb9fc8a11235d555aea77
#pragma once

#include <vector>
#include <queue>
<<<<<<< HEAD
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>
#include <future>
=======
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
>>>>>>> 42bb511ab1410a992c3fb9fc8a11235d555aea77
#include <atomic>
#include <iostream>

namespace alphazero {

/**
<<<<<<< HEAD
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
=======
 * @brief BatchEvaluator collects leaf positions and evaluates them in batches.
 * 
 * This class is designed to work with leaf parallelization in MCTS, where multiple
 * threads reach leaf nodes and need neural network evaluation. Instead of evaluating
 * each position individually, which would be inefficient with Python's GIL, this
 * class collects positions in batches and evaluates them all at once.
 */
class BatchEvaluator {
public:
    // Type definition for evaluation function
    using EvaluationFunction = std::function<std::vector<std::pair<std::vector<float>, float>>(const std::vector<std::vector<float>>&)>;

    /**
     * @brief Construct a new BatchEvaluator
     * 
     * @param evaluation_function The function to call for evaluating batches (neural network)
     * @param batch_size Maximum batch size
     * @param max_wait_ms Maximum time to wait before processing a non-full batch (milliseconds)
     */
    BatchEvaluator(
        EvaluationFunction evaluation_function,
        size_t batch_size = 16,
        size_t max_wait_ms = 10
    ) : evaluation_function_(evaluation_function),
        batch_size_(batch_size),
        max_wait_ms_(max_wait_ms),
        next_request_id_(0),
        running_(false),
        stop_requested_(false),
        positions_evaluated_(0),
        batches_processed_(0) {}
    
    /**
     * @brief Destroy the BatchEvaluator
     */
    ~BatchEvaluator() {
        stop();
    }

    /**
     * @brief Start the batch evaluation worker thread
     */
    void start() {
        std::lock_guard<std::mutex> lock(thread_mutex_);
        if (!running_) {
            stop_requested_ = false;
            running_ = true;
            worker_thread_ = std::thread(&BatchEvaluator::worker_loop, this);
        }
    }

    /**
     * @brief Stop the batch evaluation worker thread
     */
    void stop() {
        {
            std::lock_guard<std::mutex> lock(thread_mutex_);
            if (!running_) return;
            
            stop_requested_ = true;
>>>>>>> 42bb511ab1410a992c3fb9fc8a11235d555aea77
            queue_cv_.notify_all();
        }
        
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
<<<<<<< HEAD
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
=======
        
        {
            std::lock_guard<std::mutex> lock(thread_mutex_);
            running_ = false;
        }
    }

    /**
     * @brief Enqueue a position for evaluation
     * 
     * @param position The position to evaluate
     * @return int A unique request ID used to retrieve the result
     */
    int enqueue_position(const std::vector<float>& position) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        int request_id = next_request_id_++;
        queue_.push(QueueItem{request_id, position});
        queue_cv_.notify_one();
        
        return request_id;
    }

    /**
     * @brief Get the result of a previously enqueued position
     * 
     * @param request_id The request ID returned by enqueue_position
     * @return std::pair<std::vector<float>, float> Policy and value for the position
     */
    std::pair<std::vector<float>, float> get_result(int request_id) {
        std::unique_lock<std::mutex> lock(results_mutex_);
        
        // Wait until the result is available
        results_cv_.wait(lock, [this, request_id]() {
            return results_.find(request_id) != results_.end() || stop_requested_;
        });
        
        if (stop_requested_ && results_.find(request_id) == results_.end()) {
            // Return default values if shutting down
            return {{}, 0.0f};
        }
        
        // Get and remove the result
        auto result = std::move(results_[request_id]);
        results_.erase(request_id);
        
        return result;
    }

    /**
     * @brief Get statistics about the batch evaluator
     * 
     * @return std::pair<size_t, size_t> Positions evaluated and batches processed
     */
    std::pair<size_t, size_t> get_stats() const {
        return {positions_evaluated_.load(), batches_processed_.load()};
    }

    /**
     * @brief Reset the statistics counters
     */
    void reset_stats() {
        positions_evaluated_ = 0;
        batches_processed_ = 0;
    }

private:
    /**
     * @brief Structure representing a position in the evaluation queue
     */
    struct QueueItem {
        int request_id;
        std::vector<float> position;
    };

    /**
     * @brief The main worker loop that processes batches
     */
    void worker_loop() {
        while (!stop_requested_) {
            process_next_batch();
        }
    }

    /**
     * @brief Process the next batch of positions
     */
    void process_next_batch() {
        // Collect a batch of positions
        std::vector<int> batch_ids;
        std::vector<std::vector<float>> batch_positions;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            // Wait until we have items to process or need to stop
            if (queue_.empty() && !stop_requested_) {
                queue_cv_.wait_for(lock, std::chrono::milliseconds(max_wait_ms_), [this]() {
                    return !queue_.empty() || stop_requested_;
                });
            }
            
            // Check if we need to stop
            if (stop_requested_) {
                return;
            }
            
            // Check if the queue is still empty after waiting
            if (queue_.empty()) {
                return;
            }
            
            // Collect items up to batch_size
            while (!queue_.empty() && batch_positions.size() < batch_size_) {
                auto item = std::move(queue_.front());
                queue_.pop();
                
                batch_ids.push_back(item.request_id);
                batch_positions.push_back(std::move(item.position));
            }
        }
        
        // Process the batch if we have positions
        if (!batch_positions.empty()) {
            try {
                // Call the evaluation function with the batch
                auto batch_results = evaluation_function_(batch_positions);
                
                // Make sure we have results for all positions
                if (batch_results.size() != batch_positions.size()) {
                    std::cerr << "Warning: Batch evaluation returned " << batch_results.size()
                              << " results for " << batch_positions.size() << " positions" << std::endl;
                }
                
                // Store the results
                std::lock_guard<std::mutex> lock(results_mutex_);
                for (size_t i = 0; i < batch_results.size() && i < batch_ids.size(); ++i) {
                    results_[batch_ids[i]] = batch_results[i];
                }
                
                // Notify waiting threads
                results_cv_.notify_all();
                
                // Update statistics
                positions_evaluated_ += batch_positions.size();
                batches_processed_++;
            }
            catch (const std::exception& e) {
                std::cerr << "Error in batch evaluation: " << e.what() << std::endl;
                
                // Create default results for all items in the batch
                std::lock_guard<std::mutex> lock(results_mutex_);
                for (int id : batch_ids) {
                    // Default policy is uniform random
                    size_t policy_size = batch_positions[0].size();
                    std::vector<float> default_policy(policy_size, 1.0f / policy_size);
                    results_[id] = {default_policy, 0.0f};
                }
                
                // Notify waiting threads
                results_cv_.notify_all();
            }
        }
    }

    // Evaluation function to call (neural network)
    EvaluationFunction evaluation_function_;
    
    // Configuration
    size_t batch_size_;
    size_t max_wait_ms_;
    
    // Request ID counter
    std::atomic<int> next_request_id_;
    
    // Queue of positions to evaluate
    std::queue<QueueItem> queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Map of results
    std::unordered_map<int, std::pair<std::vector<float>, float>> results_;
    std::mutex results_mutex_;
    std::condition_variable results_cv_;
    
    // Worker thread management
    std::thread worker_thread_;
    std::mutex thread_mutex_;
    std::atomic<bool> running_;
    std::atomic<bool> stop_requested_;
    
    // Statistics
    std::atomic<size_t> positions_evaluated_;
    std::atomic<size_t> batches_processed_;
>>>>>>> 42bb511ab1410a992c3fb9fc8a11235d555aea77
};

} // namespace alphazero