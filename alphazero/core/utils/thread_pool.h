#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <atomic>

namespace alphazero {

/**
 * Thread pool for parallel execution of tasks.
 * This implementation uses a fixed number of worker threads and a task queue.
 */
class ThreadPool {
public:
    /**
     * Constructor.
     * 
     * @param num_threads Number of worker threads to create. If 0, use the number of hardware threads.
     */
    ThreadPool(size_t num_threads = 0);
    
    /**
     * Destructor. Joins all threads.
     */
    ~ThreadPool();
    
    /**
     * Get the number of worker threads.
     * 
     * @return Number of worker threads
     */
    size_t size() const;
    
    /**
     * Get the number of tasks currently in the queue.
     * 
     * @return Number of queued tasks
     */
    size_t queue_size() const;
    
    /**
     * Enqueue a task for execution by the thread pool.
     * 
     * @param f Function to execute
     * @param args Arguments to pass to the function
     * @return Future for the result of the function
     */
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    /**
     * Wait for all tasks to complete.
     */
    void wait_all();
    
private:
    // Worker threads
    std::vector<std::thread> workers;
    
    // Task queue
    std::queue<std::function<void()>> tasks;
    
    // Synchronization
    mutable std::mutex queue_mutex;
    std::condition_variable condition;
    
    // Shutdown flag
    std::atomic<bool> stop;
    
    // Number of active tasks
    std::atomic<size_t> active_tasks;
    std::condition_variable done_condition;
    mutable std::mutex done_mutex;
};

// Implementation of the enqueue template method
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    
    using return_type = typename std::result_of<F(Args...)>::type;
    
    // Create a packaged task that wraps the function and arguments
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    // Get the future for the task's result
    std::future<return_type> result = task->get_future();
    
    // Add the task to the queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        
        // Don't allow enqueueing after stopping the pool
        if (stop) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        
        // Increment the active task count
        active_tasks++;
        
        // Add a wrapper that decrements the active task count when done
        tasks.emplace([task, this]() {
            (*task)();
            
            // Decrement the active task count and notify if all tasks are done
            std::lock_guard<std::mutex> lock(done_mutex);
            active_tasks--;
            if (active_tasks == 0) {
                done_condition.notify_all();
            }
        });
    }
    
    // Notify a worker thread
    condition.notify_one();
    
    return result;
}

} // namespace alphazero