"""
High-Performance MCTS Engine with C++ Backend
=============================================

Fully-optimized Monte Carlo Tree Search implementation leveraging:
- C++ Structure-of-Arrays tree storage (32-64 bytes/node)
- AVX2-vectorized PUCT selection (4-8x speedup)
- Thread-safe atomic operations for concurrent search
- Memory-efficient index-based node references
- Real-time neural network integration

Performance targets:
- 30,000+ simulations/second including NN inference
- 80-92% GPU utilization sustained
- <1GB memory footprint for 10M node trees
- Thread-safe operation with 12 parallel workers
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
import logging
import time
from concurrent.futures import Future
from threading import Lock

try:
    import mcts_py
except ImportError:
    raise ImportError("C++ MCTS module not available. Run 'pip install -e .' to build extensions.")

try:
    from ..games.game_state import IGameState
except ImportError:
    # Fallback for when imported directly (e.g., in tests)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from games.game_state import IGameState


class MCTSEngine(ABC):
    """Interface for Monte Carlo Tree Search engine."""

    @abstractmethod
    def search(self, root_state: IGameState, simulations: int) -> Dict[int, float]:
        """Run MCTS search from root state."""
        pass

    @abstractmethod
    def get_policy(self, root_state: IGameState, temperature: float = 1.0) -> np.ndarray:
        """Get move probabilities from search results."""
        pass

    def get_value(self, root_state: IGameState) -> float:
        """Get position value estimate."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset search tree and internal state."""
        pass

    @property
    @abstractmethod
    def tree_size(self) -> int:
        """Get current number of nodes in search tree."""
        pass


class AlphaZeroMCTS(MCTSEngine):
    """High-performance AlphaZero MCTS with C++ backend.

    Features:
    - C++ Structure-of-Arrays tree storage
    - AVX2-vectorized PUCT selection
    - Atomic operations for thread safety
    - Memory-efficient index-based references
    - Integrated virtual loss coordination
    - Real-time neural network inference
    - Dirichlet noise for exploration
    """

    def __init__(self,
                 inference_fn: Callable[[IGameState], Future[Tuple[np.ndarray, float]]],
                 c_puct: float = 1.25,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 max_tree_size: int = 10_000_000,  # 10M nodes (~270MB)
                 virtual_loss_magnitude: float = 1.0,
                 enable_virtual_loss: bool = True,
                 enable_value_clipping: bool = True):
        """Initialize high-performance MCTS engine.

        Args:
            inference_fn: Function that returns Future[policy, value] for a game state
            c_puct: PUCT exploration constant (1.25 optimal for most games)
            dirichlet_alpha: Dirichlet noise alpha parameter
            dirichlet_epsilon: Dirichlet noise mixing ratio
            max_tree_size: Maximum nodes to prevent OOM (10M = ~270MB)
            virtual_loss_magnitude: Virtual loss penalty for thread coordination
            enable_virtual_loss: Enable virtual loss for thread safety
            enable_value_clipping: Clip values to [-1, 1] range
        """
        self.inference_fn = inference_fn
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.logger = logging.getLogger('AlphaZeroMCTS')

        # Initialize C++ MCTS tree with memory-efficient storage
        self.tree = mcts_py.MCTSTree(max_tree_size)

        # Initialize C++ PUCT selector with AVX2 optimizations
        puct_config = mcts_py.PUCTConfig()
        puct_config.cpuct = c_puct
        puct_config.fpu_value = 0.0  # First Play Urgency
        puct_config.use_fpu = True
        puct_config.enable_simd = True  # Enable AVX2 vectorization
        self.selector = mcts_py.create_puct_selector(puct_config)

        # Initialize C++ virtual loss manager for thread coordination
        vl_config = mcts_py.VirtualLossConfig(virtual_loss_magnitude, enable_virtual_loss)
        self.virtual_loss_manager = mcts_py.create_test_virtual_loss_manager(self.tree, vl_config)

        # Initialize C++ backup manager for value propagation
        backup_config = mcts_py.BackupConfig(enable_value_clipping, True, -1.0, 1.0)
        self.backup_manager = mcts_py.create_backup_manager(self.tree, backup_config)

        # State management
        self.root_state = None
        self.root_index = mcts_py.NULL_NODE_INDEX
        self._lock = Lock()

        # Critical: Move mapping for child nodes (child_index -> move)
        self._move_mapping = {}

        # Performance statistics
        self._simulations_completed = 0
        self._total_search_time = 0.0

        self.logger.info(f"AlphaZero MCTS initialized with C++ backend")
        self.logger.info(f"  Tree capacity: {max_tree_size:,} nodes (~{max_tree_size * 32 // 1024 // 1024}MB)")
        self.logger.info(f"  AVX2 support: {mcts_py.PUCTSelector.is_avx2_supported()}")
        self.logger.info(f"  PUCT constant: {c_puct}")
        self.logger.info(f"  Virtual loss: {virtual_loss_magnitude} (enabled: {enable_virtual_loss})")

    def search(self, root_state: IGameState, simulations: int, add_noise: bool = False) -> Dict[int, float]:
        """Run high-performance MCTS search with C++ backend.

        Args:
            root_state: Game state to search from
            simulations: Number of MCTS simulations to run
            add_noise: Whether to add Dirichlet noise to root for exploration

        Returns:
            Dictionary mapping moves to visit counts
        """
        start_time = time.perf_counter()

        # Clear and initialize C++ tree
        with self._lock:
            self.tree.clear()
            # CRITICAL: Clear move mapping for new search
            self._move_mapping.clear()
            self.root_state = root_state
            self.root_index = self.tree.add_root_node(0.5, root_state.get_current_player() - 1)

        # Expand root node with neural network evaluation
        self._expand_node(self.root_index, root_state)

        # Validate root expansion
        if self.tree.get_num_children(self.root_index) == 0:
            self.logger.warning("Root node has no children after expansion")
            return {}

        # Add Dirichlet noise to root for exploration
        if add_noise and self.tree.get_num_children(self.root_index) > 0:
            self._add_dirichlet_noise(self.root_index)

        # Run MCTS simulations with C++ backend
        successful_simulations = 0
        for sim in range(simulations):
            if self.tree.get_node_count() >= self.tree.get_max_nodes():
                self.logger.warning(f"Tree size limit reached: {self.tree.get_max_nodes():,}")
                break

            if self._run_simulation():
                successful_simulations += 1

        # Collect visit counts for policy
        visit_counts = {}
        first_child = self.tree.get_first_child_index(self.root_index)
        num_children = self.tree.get_num_children(self.root_index)

        if first_child != mcts_py.NULL_NODE_INDEX:
            for i in range(num_children):
                child_index = first_child + i
                if self.tree.is_valid_index(child_index):
                    move = self._get_move_for_child(child_index)
                    if move is not None:  # CRITICAL: Check for valid move
                        visit_counts[move] = int(self.tree.get_visit_count(child_index))

        # Update performance statistics
        search_time = time.perf_counter() - start_time
        self._simulations_completed += successful_simulations
        self._total_search_time += search_time

        avg_time_per_sim = search_time / max(successful_simulations, 1)
        sims_per_second = successful_simulations / search_time if search_time > 0 else 0

        self.logger.debug(f"Search completed: {successful_simulations}/{simulations} sims, "
                         f"{sims_per_second:.1f} sims/sec, {avg_time_per_sim*1000:.2f}ms/sim")

        return visit_counts

    def get_policy(self, root_state: IGameState, temperature: float = 1.0) -> np.ndarray:
        """Extract move probabilities from C++ tree.

        Args:
            root_state: Game state (for action space size)
            temperature: Temperature for softmax (0 = greedy, 1 = proportional)

        Returns:
            Probability distribution over actions
        """
        action_space_size = root_state.action_space_size
        policy = np.zeros(action_space_size, dtype=np.float32)

        legal_moves = root_state.get_legal_moves()
        legal_moves_array = np.array([move for move in legal_moves if 0 <= move < action_space_size], dtype=np.int64)

        if self.root_index == mcts_py.NULL_NODE_INDEX or self.tree.get_num_children(self.root_index) == 0:
            if legal_moves_array.size > 0:
                uniform_prob = 1.0 / float(legal_moves_array.size)
                policy[legal_moves_array] = uniform_prob
            return policy

        visits = np.zeros(action_space_size, dtype=np.float64)
        first_child = self.tree.get_first_child_index(self.root_index)
        num_children = self.tree.get_num_children(self.root_index)

        for i in range(num_children):
            child_index = first_child + i
            if self.tree.is_valid_index(child_index):
                move = self._get_move_for_child(child_index)
                if move is not None and 0 <= move < action_space_size:  # CRITICAL: Check for valid move
                    visits[move] = self.tree.get_visit_count(child_index)

        if temperature > 1.0 and legal_moves_array.size > 0:
            zero_mask = visits[legal_moves_array] <= 0.0
            if np.any(zero_mask):
                visits[legal_moves_array[zero_mask]] = 1e-6

        if temperature == 0:
            if np.all(visits == 0) and legal_moves_array.size > 0:
                policy[legal_moves_array[0]] = 1.0
            else:
                best_move = int(np.argmax(visits))
                policy[best_move] = 1.0
        else:
            total_visits = np.sum(visits)
            if total_visits > 0:
                if temperature == 1.0:
                    policy = visits / total_visits
                else:
                    visits_temp = np.power(visits, 1.0 / temperature)
                    sum_temp = np.sum(visits_temp)
                    if sum_temp > 0:
                        policy = visits_temp / sum_temp
            elif legal_moves_array.size > 0:
                uniform_prob = 1.0 / float(legal_moves_array.size)
                policy[legal_moves_array] = uniform_prob

        return policy.astype(np.float32)

    def get_value(self, root_state: IGameState) -> float:
        """Get position value estimate from C++ tree.

        Returns:
            Value estimate from current player's perspective [-1, 1]
        """
        if self.root_index == mcts_py.NULL_NODE_INDEX:
            # No search performed - use neural network evaluation
            future = self.inference_fn(root_state)
            try:
                _, value = future.result(timeout=5.0)
                return float(value)
            except Exception as e:
                self.logger.warning(f"Neural network evaluation failed: {e}")
                return 0.0

        # Get Q-value from C++ tree
        return self.backup_manager.get_q_value(self.root_index)

    def reset(self) -> None:
        """Reset search tree and internal state."""
        with self._lock:
            self.tree.clear()
            self.virtual_loss_manager.reset_all_virtual_loss()
            self.backup_manager.reset_statistics()
            self.root_state = None
            self.root_index = mcts_py.NULL_NODE_INDEX
            # CRITICAL: Clear move mapping to prevent stale references
            self._move_mapping.clear()

        self.logger.debug("MCTS tree reset")

    @property
    def tree_size(self) -> int:
        """Get current number of nodes in C++ tree."""
        return self.tree.get_node_count()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive MCTS performance statistics."""
        stats = {
            'tree_size': self.tree.get_node_count(),
            'max_tree_size': self.tree.get_max_nodes(),
            'memory_usage_mb': self.tree.get_memory_usage() / (1024 * 1024),
            'bytes_per_node': self.tree.get_bytes_per_node(),
            'simulations_completed': self._simulations_completed,
            'total_search_time': self._total_search_time,
            'avg_simulations_per_second': (
                self._simulations_completed / self._total_search_time
                if self._total_search_time > 0 else 0
            ),
            'virtual_loss_stats': self.virtual_loss_manager.get_statistics(),
            'backup_stats': self.backup_manager.get_statistics(),
            'selector_config': {
                'cpuct': self.selector.get_config().cpuct,
                'fpu_value': self.selector.get_config().fpu_value,
                'use_fpu': self.selector.get_config().use_fpu,
                'enable_simd': self.selector.get_config().enable_simd,
                'avx2_supported': mcts_py.PUCTSelector.is_avx2_supported()
            }
        }
        return stats

    def _run_simulation(self) -> bool:
        """Run a single MCTS simulation with C++ backend.

        Returns:
            True if simulation completed successfully
        """
        try:
            self.logger.debug("Starting simulation")
            # Selection phase: traverse to leaf using C++ PUCT selection
            path = []
            current_index = self.root_index
            current_state = self.root_state

            while True:
                path.append(current_index)
                self.logger.debug(f"Visiting node {current_index}, path length: {len(path)}")

                # Check if node is terminal
                flags = self.tree.get_flags(current_index)
                if flags.is_terminal():
                    # Terminal node - get game result
                    self.logger.debug(f"Node {current_index} is terminal")
                    value = self._get_terminal_value(current_state)
                    break

                # Check if node is expanded
                if not flags.is_expanded():
                    # Leaf node - expand and evaluate
                    self.logger.debug(f"Expanding leaf node {current_index}")
                    value = self._expand_node(current_index, current_state)
                    break

                # Internal node - select child using C++ PUCT
                if self.tree.get_num_children(current_index) == 0:
                    # No children available
                    value = 0.0
                    break

                selection_result = self.selector.select_child(self.tree, current_index)
                if not selection_result.valid:
                    # Selection failed
                    value = 0.0
                    break

                # Move to selected child
                current_index = selection_result.selected_child
                move = self._get_move_for_child(current_index)

                # CRITICAL: Check for valid move and properly clone state
                if move is None:
                    self.logger.error(f"Invalid move for child {current_index}")
                    value = 0.0
                    break

                # Clone state and apply move (make_move modifies in-place!)
                try:
                    new_state = current_state.clone()
                    result = new_state.make_move(move)
                    # make_move returns None but modifies new_state in-place
                    current_state = new_state
                except Exception as e:
                    self.logger.error(f"Failed to apply move {move}: {e}")
                    value = 0.0
                    break

            # CRITICAL: C++ expects leaf-to-root path order for consistency
            reversed_path = list(reversed(path))

            # Apply virtual loss along path for thread coordination
            self.logger.debug(f"Applying virtual loss to path of length {len(reversed_path)} (leaf-to-root order)")
            vl_guard = mcts_py.VirtualLossGuard(self.virtual_loss_manager, reversed_path)
            if not vl_guard.is_valid():
                self.logger.debug("Virtual loss guard invalid")
                return False

            # Backup phase: propagate value up tree with C++ backend
            self.logger.debug(f"Backing up value {value} along path (leaf-to-root order)")
            result = self.backup_manager.backup_value_along_path(reversed_path, value, self.virtual_loss_manager)

            self.logger.debug(f"Backup result: success={result.success}, nodes_updated={result.nodes_updated}")
            return result.success

        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            return False

    def _expand_node(self, node_index: int, game_state: IGameState) -> float:
        """Expand node using neural network evaluation.

        Args:
            node_index: C++ tree node index to expand
            game_state: Game state at this node

        Returns:
            Neural network value estimate
        """
        # Check if already terminal
        if game_state.is_terminal():
            flags = mcts_py.NodeFlags()
            flags.set_terminal(True)
            flags.set_current_player(game_state.get_current_player() - 1)
            self.tree.set_flags(node_index, flags)
            return self._get_terminal_value(game_state)

        # Get neural network evaluation
        try:
            future = self.inference_fn(game_state)
            policy, value = future.result(timeout=5.0)

            # Extract policy for single game state from batch
            if policy.ndim > 1:
                policy = policy[0]  # Extract the policy for the single game state
        except Exception as e:
            self.logger.error(f"Neural network inference failed: {e}")
            # Fallback to uniform policy and neutral value
            legal_moves = game_state.get_legal_moves()

            # Debug: Check legal moves in fallback case
            if len(legal_moves) > 0:
                self.logger.debug(f"Fallback - Legal moves type: {type(legal_moves)}, first move type: {type(legal_moves[0])}, content: {legal_moves[:5]}")
            else:
                self.logger.debug("Fallback - No legal moves available")

            policy = np.zeros(game_state.get_action_space_size())
            if len(legal_moves) > 0:
                for move in legal_moves:
                    policy[move] = 1.0 / len(legal_moves)
            value = 0.0

        # Mask illegal moves and normalize
        legal_moves = game_state.get_legal_moves()

        # Debug: Check legal moves type and content
        if len(legal_moves) > 0:
            self.logger.debug(f"Legal moves type: {type(legal_moves)}, first move type: {type(legal_moves[0])}, content: {legal_moves[:5]}")
        else:
            self.logger.debug("No legal moves available")

        legal_moves_set = set(legal_moves)

        for move in range(len(policy)):
            if move not in legal_moves_set:
                policy[move] = 0.0

        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy = policy / policy_sum

        # Add children to C++ tree for legal moves
        if len(legal_moves) > 0:
            # Allocate contiguous children for cache efficiency
            first_child = self.tree.allocate_nodes(len(legal_moves))
            if first_child != mcts_py.NULL_NODE_INDEX:
                for i, move in enumerate(legal_moves):
                    child_index = first_child + i
                    if move < len(policy):
                        prob_value = policy[move]
                        # Safely extract scalar value
                        try:
                            prob_array = np.asarray(prob_value).flatten()
                            if len(prob_array) > 0:
                                prior_prob = float(prob_array[0])
                            else:
                                self.logger.warning(f"Empty probability array for move {move}")
                                prior_prob = 0.0
                        except Exception as e:
                            self.logger.warning(f"Failed to extract probability for move {move}: {e}")
                            prior_prob = 0.0
                    else:
                        prior_prob = 0.0

                    # Initialize child node
                    self.tree.set_prior_prob(child_index, prior_prob)
                    self.tree.set_parent_index(child_index, node_index)
                    self.tree.set_visit_count(child_index, 0.0)
                    self.tree.set_total_value(child_index, 0.0)

                    # CRITICAL: Store move mapping for correct traversal
                    self._move_mapping[child_index] = move

                # Update parent node
                self.tree.set_first_child_index(node_index, first_child)
                self.tree.set_num_children(node_index, len(legal_moves))

        # Mark node as expanded
        flags = mcts_py.NodeFlags()
        flags.set_expanded(True)
        flags.set_current_player(game_state.get_current_player())
        self.tree.set_flags(node_index, flags)

        return float(value)

    def _add_dirichlet_noise(self, node_index: int) -> None:
        """Add Dirichlet noise to root node for exploration."""
        num_children = self.tree.get_num_children(node_index)
        if num_children == 0:
            return

        first_child = self.tree.get_first_child_index(node_index)
        if first_child == mcts_py.NULL_NODE_INDEX:
            return

        # Generate Dirichlet noise
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_children)

        # Apply noise to prior probabilities
        for i in range(num_children):
            child_index = first_child + i
            if self.tree.is_valid_index(child_index):
                current_prior = self.tree.get_prior_prob(child_index)
                noisy_prior = ((1 - self.dirichlet_epsilon) * current_prior +
                              self.dirichlet_epsilon * noise[i])
                self.tree.set_prior_prob(child_index, noisy_prior)

    def _get_terminal_value(self, game_state: IGameState) -> float:
        """Get terminal value from game result."""
        result = game_state.get_result()
        return float(result) if result is not None else 0.0

    def _get_move_for_child(self, child_index: int) -> Optional[int]:
        """Get move that led to child node using explicit move mapping.

        Returns:
            Move index, or None if not found
        """
        if child_index in self._move_mapping:
            return self._move_mapping[child_index]

        # Fallback for debugging
        self.logger.warning(f"Move mapping not found for child {child_index}")
        return None


# Backward compatibility
class MockMCTSEngine(MCTSEngine):
    """Mock implementation of MCTSEngine for testing."""

    def __init__(self, action_space_size: int = 225):
        self.action_space_size = action_space_size
        self._tree_size = 0
        self.search_results = {}

    def search(self, root_state: IGameState, simulations: int) -> Dict[int, float]:
        legal_moves = root_state.get_legal_moves()
        visit_counts = {}
        for move in legal_moves[:min(10, len(legal_moves))]:
            visit_counts[move] = np.random.randint(1, simulations // 10 + 1)
        self.search_results = visit_counts
        self._tree_size = simulations
        return visit_counts

    def get_policy(self, root_state: IGameState, temperature: float = 1.0) -> np.ndarray:
        policy = np.zeros(self.action_space_size)
        if self.search_results:
            total_visits = sum(self.search_results.values())
            for move, visits in self.search_results.items():
                if move < self.action_space_size:
                    policy[move] = visits / total_visits
        else:
            legal_moves = root_state.get_legal_moves()
            if len(legal_moves) > 0:
                for move in legal_moves:
                    if move < self.action_space_size:
                        policy[move] = 1.0 / len(legal_moves)
        if temperature != 1.0 and temperature > 0:
            policy = policy ** (1.0 / temperature)
            policy = policy / np.sum(policy)
        return policy

    def get_value(self, root_state: IGameState) -> float:
        return np.random.uniform(-1, 1)

    def reset(self) -> None:
        self._tree_size = 0
        self.search_results = {}

    @property
    def tree_size(self) -> int:
        return self._tree_size