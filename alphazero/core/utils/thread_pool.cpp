#include <iostream>
#include "thread_pool.h"

namespace alphazero {

    ThreadPool::ThreadPool(size_t num_threads)
    : stop(false), active_tasks(0) {
    
    // Use hardware concurrency if num_threads is 0
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        // Fallback to 2 threads if hardware_concurrency returns 0
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
                    // Wait for a task or stop signal
                    std::unique_lock<std::recursive_mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] {
                        return this->stop || !this->tasks.empty();
                    });
                    
                    // Exit if stopping and no tasks
                    if (this->stop && this->tasks.empty()) {
                        return;
                    }
                    
                    // Get a task
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                
                // Execute the task with exception handling
                try {
                    task();
                } catch (const std::exception& e) {
                    std::cerr << "Error in thread pool task: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "Unknown error in thread pool task" << std::endl;
                }
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::recursive_mutex> lock(queue_mutex);
        stop = true;
    }
    
    // Notify all threads to check the stop flag
    condition.notify_all();
    
    // Join all worker threads
    for (std::thread& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

size_t ThreadPool::size() const {
    return workers.size();
}

size_t ThreadPool::queue_size() const {
    std::lock_guard<std::recursive_mutex> lock(queue_mutex);
    return tasks.size();
}

void ThreadPool::wait_all() {
    std::unique_lock<std::recursive_mutex> lock(done_mutex);
    done_condition.wait(lock, [this] {
        size_t queue_size = 0;
        {
            std::lock_guard<std::recursive_mutex> q_lock(queue_mutex);
            queue_size = tasks.size();
        }
        return active_tasks == 0 && queue_size == 0;
    });
}

} // namespace alphazero