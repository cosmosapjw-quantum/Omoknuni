#include "thread_local_arena.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <new>

namespace alphazero {
namespace core {

// Thread-local storage for arena
thread_local ThreadLocalArena* g_thread_arena = nullptr;

ThreadLocalArena::ThreadLocalArena(
    size_t initial_chunks,
    size_t chunk_size,
    size_t max_chunks
)
    : current_chunk_(nullptr),
      current_offset_(0),
      chunk_size_(chunk_size),
      max_chunks_(max_chunks),
      num_chunks_(0),
      next_chunk_id_(0),
      stats_()
{
    // Pre-allocate initial chunks
    for (size_t i = 0; i < initial_chunks; ++i) {
        Chunk* chunk = allocate_chunk(chunk_size_);
        if (!chunk) {
            // OOM during initialization - clean up and throw
            while (current_chunk_) {
                Chunk* next = current_chunk_->next;
                free_chunk(current_chunk_);
                current_chunk_ = next;
            }
            throw std::bad_alloc();
        }

        // Link chunk into list
        chunk->next = current_chunk_;
        current_chunk_ = chunk;
        num_chunks_++;
    }

    // Reset offset to start of first chunk
    current_offset_ = 0;
}

ThreadLocalArena::~ThreadLocalArena() {
    // Free all chunks in linked list
    Chunk* chunk = current_chunk_;
    while (chunk) {
        Chunk* next = chunk->next;
        free_chunk(chunk);
        chunk = next;
    }

    current_chunk_ = nullptr;
    num_chunks_ = 0;
}

void* ThreadLocalArena::allocate(size_t size) {
    if (size == 0) {
        return nullptr;
    }

    // Round up to alignment boundary (64 bytes)
    size_t aligned_size = align_up(size, CACHE_LINE_SIZE);

    // FAST PATH: Try bump pointer allocation in current chunk
    if (current_chunk_) {
        size_t new_offset = current_offset_ + aligned_size;
        size_t available = chunk_size_ - current_offset_;

        if (aligned_size <= available) {
            // Allocation fits in current chunk
            void* ptr = current_chunk_->data() + current_offset_;
            current_offset_ = new_offset;
            current_chunk_->used_bytes = new_offset;
            stats_.allocations_from_bump++;
            stats_.bytes_allocated += aligned_size;
            return ptr;
        }
    }

    // SLOW PATH: Need new chunk
    return allocate_from_new_chunk(aligned_size);
}

void ThreadLocalArena::deallocate(void* ptr, size_t size) {
    // No-op for now - free list management will be added in T009d
    // For now, memory is only reclaimed via reset() or destructor
    if (ptr) {
        stats_.deallocations++;
    }
}

void ThreadLocalArena::reset() {
    // Reset all chunks to empty state (O(1) operation)
    Chunk* chunk = current_chunk_;
    while (chunk) {
        chunk->used_bytes = 0;
        chunk = chunk->next;
    }

    // Reset to first chunk
    current_offset_ = 0;

    // Reset statistics (keep chunks_allocated, but reset allocations)
    stats_.allocations_from_bump = 0;
    stats_.allocations_from_freelist = 0;
    stats_.deallocations = 0;
    stats_.bytes_allocated = 0;
    stats_.bytes_in_freelists = 0;
    stats_.fallback_to_malloc = 0;
}

ThreadLocalArena::Chunk* ThreadLocalArena::allocate_chunk(size_t size) {
    // Allocate chunk + header in one allocation
    // We use posix_memalign for 64-byte alignment
    size_t total_size = sizeof(Chunk) + size;
    void* memory = nullptr;

#ifdef _WIN32
    // Windows: Use _aligned_malloc
    memory = _aligned_malloc(total_size, CACHE_LINE_SIZE);
#else
    // POSIX: Use posix_memalign
    int ret = posix_memalign(&memory, CACHE_LINE_SIZE, total_size);
    if (ret != 0) {
        memory = nullptr;
    }
#endif

    if (!memory) {
        return nullptr;
    }

    // Initialize chunk header using placement new
    Chunk* chunk = new (memory) Chunk();
    chunk->next = nullptr;
    chunk->chunk_size = size;
    chunk->used_bytes = 0;
    chunk->chunk_id = next_chunk_id_++;

    stats_.chunks_allocated++;

    return chunk;
}

void ThreadLocalArena::free_chunk(Chunk* chunk) {
    if (!chunk) {
        return;
    }

#ifdef _WIN32
    _aligned_free(chunk);
#else
    std::free(chunk);
#endif
}

void* ThreadLocalArena::allocate_from_new_chunk(size_t aligned_size) {
    // Check if we've hit the chunk limit
    if (num_chunks_ >= max_chunks_) {
        // Fallback to malloc
        void* ptr = nullptr;
#ifdef _WIN32
        ptr = _aligned_malloc(aligned_size, CACHE_LINE_SIZE);
#else
        int ret = posix_memalign(&ptr, CACHE_LINE_SIZE, aligned_size);
        if (ret != 0) {
            ptr = nullptr;
        }
#endif

        if (ptr) {
            stats_.fallback_to_malloc++;
            stats_.bytes_allocated += aligned_size;
        }
        return ptr;
    }

    // Allocate new chunk (at least as large as requested size)
    size_t new_chunk_size = (aligned_size > chunk_size_) ? aligned_size : chunk_size_;
    Chunk* new_chunk = allocate_chunk(new_chunk_size);

    if (!new_chunk) {
        // OOM - cannot allocate new chunk
        return nullptr;
    }

    // Link new chunk at head of list
    new_chunk->next = current_chunk_;
    current_chunk_ = new_chunk;
    num_chunks_++;

    // Allocate from new chunk
    void* ptr = new_chunk->data();
    current_offset_ = aligned_size;
    new_chunk->used_bytes = aligned_size;
    stats_.allocations_from_bump++;
    stats_.bytes_allocated += aligned_size;

    return ptr;
}

// Global thread-local arena accessors

ThreadLocalArena* get_thread_arena() {
    if (!g_thread_arena) {
        // Lazy initialization
        g_thread_arena = new ThreadLocalArena(
            /*initial_chunks=*/2,
            /*chunk_size=*/64 * 1024,
            /*max_chunks=*/128
        );
    }
    return g_thread_arena;
}

void destroy_thread_arena() {
    if (g_thread_arena) {
        delete g_thread_arena;
        g_thread_arena = nullptr;
    }
}

}} // namespace alphazero::core
