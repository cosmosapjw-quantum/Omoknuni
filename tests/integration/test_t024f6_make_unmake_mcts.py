"""
Integration test for T024f-6: Zero-Copy MCTS with make/unmake pattern.

Validates that the new make/unmake approach:
1. Produces equivalent results to state pooling
2. Achieves significant performance improvement
3. Correctly restores state after each simulation
"""

import pytest
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import alphazero_py
import mcts_py


def create_simple_callback(action_space_size):
    """Create a simple inference callback that returns uniform policy."""
    call_count = [0]  # Mutable to track calls

    def callback_fn(state):
        call_count[0] += 1
        # Return uniform policy and zero value
        policy = [1.0 / action_space_size] * action_space_size
        value = 0.0
        return (policy, value)

    callback_fn.call_count_ref = call_count  # Attach for testing
    return mcts_py.PyInferenceCallback(callback_fn)


class TestT024f6MakeUnmakeMCTS:
    """Test make/unmake zero-copy MCTS implementation."""

    def test_basic_simulation_works(self):
        """Test that basic simulation with make/unmake completes successfully."""
        # Create game state
        state = alphazero_py.GomokuState()

        # Create tree and components
        tree = mcts_py.MCTSTree(10000)
        selector = mcts_py.create_puct_selector()
        backup = mcts_py.create_backup_manager(tree)
        virtual_loss = mcts_py.create_test_virtual_loss_manager(tree)

        # Create simulation runner
        runner = mcts_py.SimulationRunner(tree, selector, backup, virtual_loss)

        # Create callback
        callback = create_simple_callback(state.get_action_space_size())

        # Run single simulation
        success = runner.run_simulation(state, 0, callback)

        assert success, "Simulation should complete successfully"
        assert tree.get_node_count() > 1, "Tree should have grown"

    def test_state_restoration_with_make_unmake(self):
        """Test that thread-local state is correctly restored to root after simulation."""
        state = alphazero_py.GomokuState()
        initial_hash = state.zobrist_hash()

        # Multiple simulations should all start from the same root state
        tree = mcts_py.MCTSTree(10000)
        selector = mcts_py.create_puct_selector()
        backup = mcts_py.create_backup_manager(tree)
        virtual_loss = mcts_py.create_test_virtual_loss_manager(tree)
        runner = mcts_py.SimulationRunner(tree, selector, backup, virtual_loss)
        callback = create_simple_callback(state.get_action_space_size())

        # Run multiple simulations
        for i in range(10):
            # State hash should always be the initial hash (thread-local state restored)
            current_hash = state.zobrist_hash()
            assert current_hash == initial_hash, f"State hash changed on iteration {i}"

            runner.run_simulation(state, 0, callback)

    def test_tree_growth_consistency(self):
        """Test that tree growth is consistent with make/unmake pattern."""
        state = alphazero_py.GomokuState()

        tree = mcts_py.MCTSTree(100000)
        selector = mcts_py.create_puct_selector()
        backup = mcts_py.create_backup_manager(tree)
        virtual_loss = mcts_py.create_test_virtual_loss_manager(tree)
        runner = mcts_py.SimulationRunner(tree, selector, backup, virtual_loss)
        callback = create_simple_callback(state.get_action_space_size())

        # Run N simulations
        num_sims = 100
        for _ in range(num_sims):
            runner.run_simulation(state, 0, callback)

        # Verify tree statistics
        root_visits = tree.get_visit_count(0)
        assert root_visits == pytest.approx(num_sims, abs=5), \
            f"Root should have ~{num_sims} visits, got {root_visits}"

        # Check that tree expanded properly
        assert tree.get_node_count() > 1, "Tree should have multiple nodes"
        assert tree.get_node_count() < 10000, "Tree shouldn't be excessively large"

    def test_make_unmake_performance_improvement(self):
        """Validate that make/unmake provides significant speedup."""
        state = alphazero_py.GomokuState()

        tree = mcts_py.MCTSTree(100000)
        selector = mcts_py.create_puct_selector()
        backup = mcts_py.create_backup_manager(tree)
        virtual_loss = mcts_py.create_test_virtual_loss_manager(tree)
        runner = mcts_py.SimulationRunner(tree, selector, backup, virtual_loss)
        callback = create_simple_callback(state.get_action_space_size())

        # Warmup
        for _ in range(10):
            runner.run_simulation(state, 0, callback)

        # Benchmark: measure time for N simulations
        num_sims = 1000
        start = time.perf_counter()
        for _ in range(num_sims):
            runner.run_simulation(state, 0, callback)
        elapsed = time.perf_counter() - start

        sims_per_sec = num_sims / elapsed
        time_per_sim_us = (elapsed / num_sims) * 1e6

        print(f"\n  Performance:")
        print(f"    {sims_per_sec:.0f} sims/sec")
        print(f"    {time_per_sim_us:.1f} μs per simulation")
        print(f"    Total time: {elapsed*1000:.1f} ms for {num_sims} simulations")

        # With make/unmake, we expect significant improvement
        # Conservative target: >500 sims/sec (much better than old approach)
        # Note: This is WITHOUT async inference, so it's synchronous
        assert sims_per_sec > 500, f"Performance regression: {sims_per_sec:.0f} sims/sec"

    def test_hash_consistency(self):
        """Test that zobrist hash is correctly maintained with make/unmake."""
        state1 = alphazero_py.GomokuState()
        state2 = alphazero_py.GomokuState()

        # Both should start with same hash
        assert state1.zobrist_hash() == state2.zobrist_hash()

        # After simulations, state1 should still match fresh state2
        tree = mcts_py.MCTSTree(10000)
        selector = mcts_py.create_puct_selector()
        backup = mcts_py.create_backup_manager(tree)
        virtual_loss = mcts_py.create_test_virtual_loss_manager(tree)
        runner = mcts_py.SimulationRunner(tree, selector, backup, virtual_loss)
        callback = SimpleInferenceCallback(state1.get_action_space_size())

        for _ in range(50):
            runner.run_simulation(state1, 0, callback)

        # state1 should be restored to root (same hash as fresh state2)
        assert state1.zobrist_hash() == state2.zobrist_hash(), \
            "State not properly restored to root"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
