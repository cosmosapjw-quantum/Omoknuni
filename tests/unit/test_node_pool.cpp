/**
 * @file test_node_pool.cpp
 * @brief Unit tests for node pool pre-allocation functionality
 */

#include <gtest/gtest.h>
#include "../../cpp_extensions/mcts/tree.hpp"
#include <vector>
#include <algorithm>

using namespace mcts;

class NodePoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a small tree for testing
        tree = std::make_unique<MCTSTree>(1000);
    }

    std::unique_ptr<MCTSTree> tree;
};

TEST_F(NodePoolTest, AllocateSingleNode) {
    // Initially no nodes allocated
    EXPECT_EQ(tree->get_node_count(), 0);
    EXPECT_EQ(tree->get_available_nodes(), 1000);

    // Allocate a single node
    NodeIndex node1 = tree->allocate_node();
    EXPECT_NE(node1, NULL_NODE_INDEX);
    EXPECT_EQ(tree->get_node_count(), 1);
    EXPECT_EQ(tree->get_available_nodes(), 999);

    // Allocate another node
    NodeIndex node2 = tree->allocate_node();
    EXPECT_NE(node2, NULL_NODE_INDEX);
    EXPECT_NE(node1, node2);
    EXPECT_EQ(tree->get_node_count(), 2);
    EXPECT_EQ(tree->get_available_nodes(), 998);
}

TEST_F(NodePoolTest, AllocateMultipleContiguousNodes) {
    // Allocate 5 contiguous nodes
    NodeIndex first_node = tree->allocate_nodes(5);
    EXPECT_NE(first_node, NULL_NODE_INDEX);
    EXPECT_EQ(tree->get_node_count(), 5);
    EXPECT_EQ(tree->get_available_nodes(), 995);

    // Verify the nodes are contiguous
    for (int i = 1; i < 5; ++i) {
        EXPECT_EQ(first_node + i, first_node + i);  // Verify arithmetic works
    }

    // Allocate more contiguous nodes
    NodeIndex second_batch = tree->allocate_nodes(3);
    EXPECT_NE(second_batch, NULL_NODE_INDEX);
    EXPECT_EQ(second_batch, first_node + 5);  // Should be immediately after first batch
    EXPECT_EQ(tree->get_node_count(), 8);
    EXPECT_EQ(tree->get_available_nodes(), 992);
}

TEST_F(NodePoolTest, DeallocateAndReuse) {
    // Allocate some nodes
    NodeIndex node1 = tree->allocate_node();
    NodeIndex node2 = tree->allocate_node();
    NodeIndex node3 = tree->allocate_node();

    EXPECT_EQ(tree->get_node_count(), 3);

    // Deallocate middle node
    tree->deallocate_node(node2);
    EXPECT_EQ(tree->get_node_count(), 2);
    EXPECT_EQ(tree->get_available_nodes(), 998);

    // Allocate a new node - should reuse the deallocated one
    NodeIndex node4 = tree->allocate_node();
    EXPECT_EQ(node4, node2);  // Should reuse the freed node
    EXPECT_EQ(tree->get_node_count(), 3);
}

TEST_F(NodePoolTest, DeallocateMultipleNodes) {
    // Allocate a batch of nodes
    NodeIndex first_node = tree->allocate_nodes(10);
    EXPECT_NE(first_node, NULL_NODE_INDEX);
    EXPECT_EQ(tree->get_node_count(), 10);

    // Deallocate half of them
    tree->deallocate_nodes(first_node + 5, 5);
    EXPECT_EQ(tree->get_node_count(), 5);
    EXPECT_EQ(tree->get_available_nodes(), 995);

    // Deallocate the rest
    tree->deallocate_nodes(first_node, 5);
    EXPECT_EQ(tree->get_node_count(), 0);
    EXPECT_EQ(tree->get_available_nodes(), 1000);
}

TEST_F(NodePoolTest, ExhaustPool) {
    // Create a small tree to easily exhaust
    auto small_tree = std::make_unique<MCTSTree>(3);

    // Allocate all available nodes
    NodeIndex node1 = small_tree->allocate_node();
    NodeIndex node2 = small_tree->allocate_node();
    NodeIndex node3 = small_tree->allocate_node();

    EXPECT_NE(node1, NULL_NODE_INDEX);
    EXPECT_NE(node2, NULL_NODE_INDEX);
    EXPECT_NE(node3, NULL_NODE_INDEX);
    EXPECT_EQ(small_tree->get_node_count(), 3);
    EXPECT_EQ(small_tree->get_available_nodes(), 0);

    // Try to allocate one more - should fail
    NodeIndex node4 = small_tree->allocate_node();
    EXPECT_EQ(node4, NULL_NODE_INDEX);
    EXPECT_EQ(small_tree->get_node_count(), 3);  // Should remain unchanged
}

TEST_F(NodePoolTest, ExhaustPoolMultipleAllocation) {
    // Create a small tree
    auto small_tree = std::make_unique<MCTSTree>(5);

    // Try to allocate more nodes than available
    NodeIndex nodes = small_tree->allocate_nodes(10);
    EXPECT_EQ(nodes, NULL_NODE_INDEX);
    EXPECT_EQ(small_tree->get_node_count(), 0);  // Should remain unchanged

    // Allocate exactly the right amount
    nodes = small_tree->allocate_nodes(5);
    EXPECT_NE(nodes, NULL_NODE_INDEX);
    EXPECT_EQ(small_tree->get_node_count(), 5);
    EXPECT_EQ(small_tree->get_available_nodes(), 0);
}

TEST_F(NodePoolTest, HasSpaceForCheck) {
    EXPECT_TRUE(tree->has_space_for(1));
    EXPECT_TRUE(tree->has_space_for(100));
    EXPECT_TRUE(tree->has_space_for(1000));
    EXPECT_FALSE(tree->has_space_for(1001));

    // Allocate some nodes and check again
    tree->allocate_nodes(500);
    EXPECT_TRUE(tree->has_space_for(1));
    EXPECT_TRUE(tree->has_space_for(500));
    EXPECT_FALSE(tree->has_space_for(501));
}

TEST_F(NodePoolTest, ClearResetsPool) {
    // Allocate some nodes
    tree->allocate_nodes(100);
    EXPECT_EQ(tree->get_node_count(), 100);
    EXPECT_EQ(tree->get_available_nodes(), 900);

    // Clear the tree
    tree->clear();
    EXPECT_EQ(tree->get_node_count(), 0);
    EXPECT_EQ(tree->get_available_nodes(), 1000);

    // Should be able to allocate again
    NodeIndex node = tree->allocate_node();
    EXPECT_NE(node, NULL_NODE_INDEX);
    EXPECT_EQ(tree->get_node_count(), 1);
}

TEST_F(NodePoolTest, AddRootNodeUsesPool) {
    // Add root node
    NodeIndex root = tree->add_root_node(0.5f, 0);
    EXPECT_NE(root, NULL_NODE_INDEX);
    EXPECT_EQ(tree->get_node_count(), 1);
    EXPECT_EQ(tree->get_available_nodes(), 999);

    // Verify root is valid
    EXPECT_TRUE(tree->is_valid_index(root));
    EXPECT_EQ(tree->get_prior_prob(root), 0.5f);
    EXPECT_EQ(tree->get_flags(root).current_player(), 0);
}

TEST_F(NodePoolTest, ZeroAllocationHandling) {
    // Test edge cases
    NodeIndex nodes = tree->allocate_nodes(0);
    EXPECT_EQ(nodes, NULL_NODE_INDEX);
    EXPECT_EQ(tree->get_node_count(), 0);

    // Deallocate zero nodes should be safe
    tree->deallocate_nodes(0, 0);
    EXPECT_EQ(tree->get_node_count(), 0);
}

TEST_F(NodePoolTest, MemoryEfficiency) {
    // Test that we achieve the target memory efficiency
    const std::size_t target_nodes = 10'000'000;  // 10M nodes
    const std::size_t max_memory_gb = 1;  // 1GB limit
    const std::size_t max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024;

    // Create tree with target capacity
    auto large_tree = std::make_unique<MCTSTree>(target_nodes);

    // Check memory usage
    std::size_t memory_usage = large_tree->get_memory_usage();
    EXPECT_LE(memory_usage, max_memory_bytes)
        << "Memory usage (" << memory_usage << " bytes) exceeds 1GB limit";

    // Check bytes per node
    double bytes_per_node = static_cast<double>(memory_usage) / target_nodes;
    EXPECT_LE(bytes_per_node, 64.0)
        << "Bytes per node (" << bytes_per_node << ") exceeds 64 byte target";

    // Should be much better than 64 bytes actually
    EXPECT_LE(bytes_per_node, 40.0)
        << "Bytes per node (" << bytes_per_node << ") should be closer to 32-40 bytes";
}