/**
 * @file enhanced_metrics.hpp
 * @brief Comprehensive metric definitions for MCTS profiling
 * @author MCTS Performance Team
 * @date 2024
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <array>
#include <unordered_map>

namespace mcts {
namespace profiling {

/**
 * @brief Types of profiling metrics
 */
enum class MetricType : uint8_t {
    Timer,          // Duration measurement
    Counter,        // Event counting
    Gauge,          // Current value measurement
    Histogram,      // Distribution measurement
    HardwareCounter // CPU performance counter
};

/**
 * @brief Categories for organizing metrics
 */
enum class MetricCategory : uint8_t {
    Selection,
    Expansion,
    Backup,
    VirtualLoss,
    Queue,
    Memory,
    Synchronization,
    Hardware,
    Thread,
    Pipeline,
    GPU,
    Network
};

/**
 * @brief Comprehensive metric enumeration
 */
enum class ProfileMetric : uint16_t {
    // Selection Phase (0-19)
    SelectionTotal = 0,
    SelectionPUCT,
    SelectionTreeTraversal,
    SelectionChildSelection,
    SelectionVirtualLossWait,
    SelectionBusyEdgeSkip,
    SelectionDepth,
    SelectionRetries,
    SelectionCacheHits,
    SelectionCacheMisses,
    SelectionAVX2Operations,
    SelectionBranchMispredictions,

    // Expansion Phase (20-39)
    ExpansionTotal = 20,
    ExpansionStateClone,
    ExpansionLegalMoveGen,
    ExpansionNeuralNetRequest,
    ExpansionNeuralNetWait,
    ExpansionPolicyMasking,
    ExpansionNodeAllocation,
    ExpansionChildInit,
    ExpansionConflicts,
    ExpansionMemoryAlloc,
    ExpansionDLPackConversion,
    ExpansionTensorCreation,

    // Backup Phase (40-54)
    BackupTotal = 40,
    BackupPathTraversal,
    BackupValueUpdate,
    BackupVirtualLossRemoval,
    BackupAtomicOperations,
    BackupSignFlipping,
    BackupCASRetries,

    // Virtual Loss (55-64)
    VirtualLossApply = 55,
    VirtualLossRemove,
    VirtualLossContention,
    VirtualLossCASRetries,
    VirtualLossSpinWait,

    // Queue Operations (65-84)
    QueueSubmitTotal = 65,
    QueueSubmitEnqueue,
    QueueSubmitRetries,
    QueueCollectTotal,
    QueueCollectWait,
    QueueCollectBatchSize,
    QueueProcessResults,
    QueueTryGetResult,
    QueueConditionWait,
    QueueSpuriousWakeups,
    QueueRingBufferFull,
    QueueDLPackSubmit,

    // Memory Operations (85-104)
    MemoryNodeAllocation = 85,
    MemoryNodeDeallocation,
    MemoryArenaAllocation,
    MemoryArenaContention,
    MemoryFragmentation,
    MemoryPageFaults,
    MemoryCacheLineLoads,
    MemoryCacheLineEvictions,
    MemoryFalseSharing,
    MemoryBandwidthUsed,
    MemoryTLBMisses,
    MemoryPrefetchHits,

    // Synchronization (105-119)
    SyncMutexLockWait = 105,
    SyncMutexLockAcquired,
    SyncAtomicCAS,
    SyncAtomicCASRetries,
    SyncSpinlockSpins,
    SyncCondvarWait,
    SyncCondvarSignal,
    SyncBarrierWait,
    SyncFutexWait,
    SyncRWLockRead,
    SyncRWLockWrite,

    // Hardware Counters (120-144)
    HWCPUCycles = 120,
    HWInstructions,
    HWIPC,
    HWCacheMissesL1D,
    HWCacheMissesL1I,
    HWCacheMissesL2,
    HWCacheMissesLLC,
    HWBranchMisses,
    HWBranchInstructions,
    HWPageFaults,
    HWContextSwitches,
    HWCPUMigrations,
    HWTLBMissesData,
    HWTLBMissesInstruction,
    HWStalledCyclesFrontend,
    HWStalledCyclesBackend,
    HWBusTransactions,
    HWRemoteAccesses,

    // Thread Metrics (145-164)
    ThreadActiveTime = 145,
    ThreadIdleTime,
    ThreadWaitTime,
    ThreadUserTime,
    ThreadSystemTime,
    ThreadVoluntaryCtxSwitch,
    ThreadInvoluntaryCtxSwitch,
    ThreadCPUMigrations,
    ThreadWorkStolen,
    ThreadTasksCompleted,
    ThreadQueueDepth,
    ThreadAffinity,

    // Pipeline Metrics (165-184)
    PipelineE2ELatency = 165,
    PipelineStageSelection,
    PipelineStageExpansion,
    PipelineStageBackup,
    PipelineStageInference,
    PipelineQueueDepth,
    PipelineThroughput,
    PipelineUtilization,
    PipelineStalls,
    PipelineBottleneck,

    // GPU Metrics (185-204)
    GPUInferenceTime = 185,
    GPUKernelTime,
    GPUH2DTransfer,
    GPUD2HTransfer,
    GPUMemoryUsed,
    GPUUtilization,
    GPUSMOccupancy,
    GPUTensorCoreUsage,
    GPUBatchSize,
    GPUQueueDepth,
    GPUPowerUsage,
    GPUTemperature,
    GPUThrottling,

    // Network/Communication (205-219)
    NetInterThreadComm = 205,
    NetCacheSyncs,
    NetMemoryBarriers,
    NetAtomicBroadcasts,
    NetDMATransfers,
    NetPCIeBandwidth,

    // Aggregate/Derived Metrics (220-239)
    TotalSimulations = 220,
    SimulationsPerSecond,
    EffectiveBranchingFactor,
    TreeEfficiency,
    ParallelEfficiency,
    ContentionFactor,
    CacheEfficiency,
    MemoryEfficiency,

    // Sentinel
    MetricCount = 240
};

/**
 * @brief Metadata for each metric
 */
struct MetricMetadata {
    const char* name;
    const char* description;
    MetricType type;
    MetricCategory category;
    const char* unit;
    bool is_derived;  // Computed from other metrics
};

/**
 * @brief Get metadata for a metric
 */
constexpr MetricMetadata get_metric_metadata(ProfileMetric metric) {
    switch (metric) {
        // Selection metrics
        case ProfileMetric::SelectionTotal:
            return {"selection_total", "Total selection time", MetricType::Timer, MetricCategory::Selection, "ns", false};
        case ProfileMetric::SelectionPUCT:
            return {"selection_puct", "PUCT computation time", MetricType::Timer, MetricCategory::Selection, "ns", false};
        case ProfileMetric::SelectionTreeTraversal:
            return {"selection_traversal", "Tree traversal time", MetricType::Timer, MetricCategory::Selection, "ns", false};
        case ProfileMetric::SelectionBusyEdgeSkip:
            return {"selection_busy_skip", "Nodes skipped due to busy edges", MetricType::Counter, MetricCategory::Selection, "count", false};
        case ProfileMetric::SelectionDepth:
            return {"selection_depth", "Selection depth reached", MetricType::Gauge, MetricCategory::Selection, "levels", false};

        // Expansion metrics
        case ProfileMetric::ExpansionTotal:
            return {"expansion_total", "Total expansion time", MetricType::Timer, MetricCategory::Expansion, "ns", false};
        case ProfileMetric::ExpansionNeuralNetWait:
            return {"expansion_nn_wait", "Neural network wait time", MetricType::Timer, MetricCategory::Expansion, "ns", false};
        case ProfileMetric::ExpansionConflicts:
            return {"expansion_conflicts", "Expansion conflicts detected", MetricType::Counter, MetricCategory::Expansion, "count", false};

        // Backup metrics
        case ProfileMetric::BackupTotal:
            return {"backup_total", "Total backup time", MetricType::Timer, MetricCategory::Backup, "ns", false};
        case ProfileMetric::BackupCASRetries:
            return {"backup_cas_retries", "CAS retry count during backup", MetricType::Counter, MetricCategory::Backup, "count", false};

        // Virtual Loss metrics
        case ProfileMetric::VirtualLossContention:
            return {"vl_contention", "Virtual loss contention events", MetricType::Counter, MetricCategory::VirtualLoss, "count", false};

        // Queue metrics
        case ProfileMetric::QueueCollectBatchSize:
            return {"queue_batch_size", "Collected batch size", MetricType::Histogram, MetricCategory::Queue, "count", false};
        case ProfileMetric::QueueConditionWait:
            return {"queue_cv_wait", "Condition variable wait time", MetricType::Timer, MetricCategory::Queue, "ns", false};

        // Memory metrics
        case ProfileMetric::MemoryBandwidthUsed:
            return {"memory_bandwidth", "Memory bandwidth utilization", MetricType::Gauge, MetricCategory::Memory, "GB/s", false};
        case ProfileMetric::MemoryFalseSharing:
            return {"memory_false_sharing", "False sharing events detected", MetricType::Counter, MetricCategory::Memory, "count", false};

        // Hardware counter metrics
        case ProfileMetric::HWIPC:
            return {"hw_ipc", "Instructions per cycle", MetricType::Gauge, MetricCategory::Hardware, "IPC", true};
        case ProfileMetric::HWCacheMissesL1D:
            return {"hw_l1d_misses", "L1 data cache misses", MetricType::Counter, MetricCategory::Hardware, "count", false};

        // Thread metrics
        case ProfileMetric::ThreadActiveTime:
            return {"thread_active", "Thread active time", MetricType::Timer, MetricCategory::Thread, "ns", false};
        case ProfileMetric::ThreadWorkStolen:
            return {"thread_work_stolen", "Work stealing events", MetricType::Counter, MetricCategory::Thread, "count", false};

        // GPU metrics
        case ProfileMetric::GPUInferenceTime:
            return {"gpu_inference", "GPU inference time", MetricType::Timer, MetricCategory::GPU, "ns", false};
        case ProfileMetric::GPUUtilization:
            return {"gpu_utilization", "GPU SM utilization", MetricType::Gauge, MetricCategory::GPU, "%", false};

        // Derived metrics
        case ProfileMetric::SimulationsPerSecond:
            return {"sims_per_sec", "Simulations per second", MetricType::Gauge, MetricCategory::Pipeline, "sims/s", true};
        case ProfileMetric::ParallelEfficiency:
            return {"parallel_efficiency", "Parallel execution efficiency", MetricType::Gauge, MetricCategory::Thread, "%", true};

        default:
            return {"unknown", "Unknown metric", MetricType::Counter, MetricCategory::Selection, "?", false};
    }
}

/**
 * @brief Convert metric to string name
 */
constexpr const char* metric_to_string(ProfileMetric metric) {
    return get_metric_metadata(metric).name;
}

/**
 * @brief Convert category to string
 */
constexpr const char* category_to_string(MetricCategory category) {
    switch (category) {
        case MetricCategory::Selection: return "Selection";
        case MetricCategory::Expansion: return "Expansion";
        case MetricCategory::Backup: return "Backup";
        case MetricCategory::VirtualLoss: return "VirtualLoss";
        case MetricCategory::Queue: return "Queue";
        case MetricCategory::Memory: return "Memory";
        case MetricCategory::Synchronization: return "Synchronization";
        case MetricCategory::Hardware: return "Hardware";
        case MetricCategory::Thread: return "Thread";
        case MetricCategory::Pipeline: return "Pipeline";
        case MetricCategory::GPU: return "GPU";
        case MetricCategory::Network: return "Network";
        default: return "Unknown";
    }
}

/**
 * @brief Profiling levels for compile-time configuration
 */
enum class ProfileLevel : uint8_t {
    None = 0,      // No profiling (0% overhead)
    Basic = 1,     // Timers only (<0.1% overhead)
    Detailed = 2,  // Timers + hardware counters (<0.5% overhead)
    Full = 3       // Everything including memory tracking (<1% overhead)
};

// Compile-time profiling level (use integer constants for preprocessor)
#ifndef PROFILE_LEVEL_VALUE
    #ifdef NDEBUG
        #define PROFILE_LEVEL_VALUE 0  // None
    #else
        #define PROFILE_LEVEL_VALUE 1  // Basic
    #endif
#endif

} // namespace profiling
} // namespace mcts