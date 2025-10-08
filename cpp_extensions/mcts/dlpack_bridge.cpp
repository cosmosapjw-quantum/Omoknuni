/**
 * @file dlpack_bridge.cpp
 * @brief Implementation of DLPack tensor bridge components
 */

#include "dlpack_bridge.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <optional>

// CUDA headers (with availability detection)
#ifdef __has_include
#  if __has_include(<cuda_runtime.h>)
#    define HAS_CUDA 1
#    include <cuda_runtime.h>
#  else
#    define HAS_CUDA 0
#  endif
#else
#  define HAS_CUDA 0
#endif

namespace mcts {

// ============================================================================
// CUDA Availability Detection
// ============================================================================

bool is_cuda_available() {
#if HAS_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

// ============================================================================
// PinnedBuffer Implementation
// ============================================================================

PinnedBuffer::PinnedBuffer(size_t size_bytes, bool use_cuda)
    : size_bytes_(size_bytes) {

    if (size_bytes == 0) {
        throw std::invalid_argument("PinnedBuffer: size_bytes must be > 0");
    }

    // Try CUDA pinned memory if requested and available
    bool allocated = false;
    if (use_cuda && is_cuda_available()) {
#if HAS_CUDA
        cudaError_t err = cudaMallocHost(&data_, size_bytes);
        if (err == cudaSuccess) {
            is_cuda_pinned_ = true;
            allocated = true;
        }
        // Fall through to malloc on failure
#endif
    }

    // Fallback to regular malloc
    if (!allocated) {
        data_ = std::malloc(size_bytes);
        if (!data_) {
            throw std::bad_alloc();
        }
        is_cuda_pinned_ = false;
    }
}

PinnedBuffer::~PinnedBuffer() {
    free_memory();
}

void PinnedBuffer::free_memory() {
    if (!data_) {
        return;
    }

    if (is_cuda_pinned_) {
#if HAS_CUDA
        cudaFreeHost(data_);
#endif
    } else {
        std::free(data_);
    }
    data_ = nullptr;
}

// ============================================================================
// BufferPool Implementation
// ============================================================================

BufferPool& BufferPool::instance() {
    static BufferPool pool;
    return pool;
}

std::optional<BufferPool::SizeClass> BufferPool::get_size_class(size_t size) const {
    for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
        if (size <= SIZE_CLASS_BYTES[i]) {
            return static_cast<SizeClass>(i);
        }
    }
    return std::nullopt;  // Too large for pooling
}

size_t BufferPool::get_buffer_size(SizeClass sc) const {
    return SIZE_CLASS_BYTES[static_cast<size_t>(sc)];
}

std::shared_ptr<PinnedBuffer> BufferPool::acquire(size_t min_size, bool use_cuda) {
    auto size_class_opt = get_size_class(min_size);

    // If size is within poolable range, try to reuse from pool
    if (size_class_opt.has_value()) {
        SizeClass sc = size_class_opt.value();
        size_t class_idx = static_cast<size_t>(sc);

        std::lock_guard<std::mutex> lock(mutex_);

        // Check if we have a cached buffer
        if (!pools_[class_idx].empty()) {
            auto buffer = pools_[class_idx].back();
            pools_[class_idx].pop_back();

            total_reused_.fetch_add(1, std::memory_order_relaxed);
            return buffer;
        }
    }

    // Cache miss or too large - allocate new buffer
    size_t alloc_size = size_class_opt.has_value()
        ? get_buffer_size(size_class_opt.value())
        : min_size;

    total_allocated_.fetch_add(1, std::memory_order_relaxed);

    // Simple allocation - pool reuse is manual via release()
    return std::make_shared<PinnedBuffer>(alloc_size, use_cuda);
}

void BufferPool::release(std::shared_ptr<PinnedBuffer> buffer) {
    if (!buffer) {
        return;
    }

    // Check if buffer is poolable size
    auto size_class_opt = get_size_class(buffer->size());
    if (!size_class_opt.has_value()) {
        // Too large for pooling, just let it be deleted
        return;
    }

    SizeClass sc = size_class_opt.value();
    size_t class_idx = static_cast<size_t>(sc);

    std::lock_guard<std::mutex> lock(mutex_);

    // Only cache if pool has space
    if (pools_[class_idx].size() < max_buffers_per_class_) {
        pools_[class_idx].push_back(buffer);
    }
    // Otherwise let buffer be deleted when shared_ptr goes out of scope
}

void BufferPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
        pools_[i].clear();
    }
}

BufferPool::Stats BufferPool::get_stats() const {
    Stats stats;
    stats.total_allocated = total_allocated_.load(std::memory_order_relaxed);
    stats.total_reused = total_reused_.load(std::memory_order_relaxed);

    std::lock_guard<std::mutex> lock(mutex_);

    for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
        stats.current_pooled += pools_[i].size();
        for (const auto& buffer : pools_[i]) {
            stats.current_bytes += buffer->size();
        }
    }

    return stats;
}

void BufferPool::set_max_buffers_per_class(size_t max_buffers) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_buffers_per_class_ = max_buffers;

    // Trim pools if they exceed new limit
    for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
        if (pools_[i].size() > max_buffers) {
            pools_[i].resize(max_buffers);
        }
    }
}

} // namespace mcts
