import math
import numpy as np
import time
import threading
import concurrent.futures
from typing import List, Dict, Tuple, Callable, Optional, Any, Union, Set
import random
import queue

from alphazero.python.games.game_base import GameWrapper
from alphazero.python.mcts.transposition_table import TranspositionTable, LRUTranspositionTable


class EnhancedMCTSNode:
    """
    Enhanced node in the Monte Carlo Tree Search with virtual loss.
    """
    def __init__(self, prior: float = 0.0, parent=None, move: int = -1):
        """
        Initialize a new MCTS node.
        
        Args:
            prior: Prior probability of selecting this node
            parent: Parent node
            move: The move that led to this state from parent
        """
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.parent = parent
        self.move = move  # The move that led to this state from parent
        self.children: Dict[int, 'EnhancedMCTSNode'] = {}  # Maps moves to child nodes
        
        # For virtual loss
        self.virtual_loss = 0
        self.lock = threading.RLock()
        
        # For RAVE (Rapid Action Value Estimation)
        self.rave_count = 0
        self.rave_value = 0.0
    
    def is_expanded(self) -> bool:
        """Check if the node has been expanded."""
        return len(self.children) > 0
    
    def value(self) -> float:
        """Calculate the mean value of this node."""
        with self.lock:
            if self.visit_count == 0:
                return 0.0
            return self.value_sum / self.visit_count
    
    def rave_value_estimate(self) -> float:
        """Calculate the RAVE value estimate."""
        with self.lock:
            if self.rave_count == 0:
                return 0.0
            return self.rave_value / self.rave_count
    
    def add_virtual_loss(self, amount: int = 1) -> None:
        """
        Add virtual loss to this node to discourage other threads from exploring this path.
        
        Args:
            amount: Amount of virtual loss to add
        """
        with self.lock:
            self.virtual_loss += amount
    
    def remove_virtual_loss(self, amount: int = 1) -> None:
        """
        Remove virtual loss from this node.
        
        Args:
            amount: Amount of virtual loss to remove
        """
        with self.lock:
            self.virtual_loss -= amount
    
    def ucb_score(self, parent_visit_count: int, c_puct: float, rave_weight: float = 0.0) -> float:
        """
        Calculate the Upper Confidence Bound (UCB) score for this node with virtual loss.
        
        Args:
            parent_visit_count: Number of visits to the parent node
            c_puct: Exploration constant
            rave_weight: Weight for RAVE score (0.0 to disable RAVE)
            
        Returns:
            The UCB score
        """
        with self.lock:
            # Effective visit count (including virtual loss)
            effective_visits = self.visit_count + self.virtual_loss
            
            if effective_visits == 0:
                # If no visits yet, use a high value to encourage exploration
                return float('inf')
            
            # Exploitation term: Q(s,a)
            exploitation = self.value_sum / effective_visits
            
            # Exploration term: c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            exploration = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + effective_visits)
            
            # Calculate a combined score with an optional RAVE component
            if rave_weight > 0.0 and self.rave_count > 0:
                rave_score = self.rave_value_estimate()
                rave_factor = rave_weight * self.rave_count / (self.rave_count + effective_visits + 1e-5)
                return (1 - rave_factor) * (exploitation + exploration) + rave_factor * rave_score
            else:
                return exploitation + exploration
    
    def select_child(self, c_puct: float, rave_weight: float = 0.0) -> Tuple[int, 'EnhancedMCTSNode']:
        """
        Select the child with the highest UCB score.
        
        Args:
            c_puct: Exploration constant
            rave_weight: Weight for RAVE score
            
        Returns:
            Tuple of (move, child_node)
        """
        with self.lock:
            # Find the move with the highest UCB score
            best_score = float('-inf')
            best_move = -1
            best_child = None
            
            for move, child in self.children.items():
                score = child.ucb_score(self.visit_count, c_puct, rave_weight)
                if score > best_score:
                    best_score = score
                    best_move = move
                    best_child = child
            
            return best_move, best_child
    
    def expand(self, moves: List[int], priors: Dict[int, float]) -> None:
        """
        Expand the node with the given moves and priors.
        
        Args:
            moves: List of legal moves from this state
            priors: Dictionary mapping moves to prior probabilities
        """
        with self.lock:
            for move in moves:
                # Get the prior probability for this move, default to 0 if not in priors
                prior = priors.get(move, 0.0)
                self.children[move] = EnhancedMCTSNode(prior=prior, parent=self, move=move)
    
    def backup(self, value: float, path: List[int] = None) -> None:
        """
        Update the node and its ancestors with the given value.
        
        Args:
            value: The value to back up
            path: List of moves in the path for RAVE updates (if None, RAVE not used)
        """
        with self.lock:
            # Update this node
            self.visit_count += 1
            self.value_sum += value
            
            # Update RAVE statistics if path is provided
            if path is not None:
                moves_set = set(path)
                for move, child in self.children.items():
                    if move in moves_set:
                        child.rave_count += 1
                        child.rave_value += value
            
            # Remove any virtual loss that was applied during selection
            self.remove_virtual_loss()
        
        # Update parent node with the negative of the value (for alternating players)
        if self.parent is not None:
            self.parent.backup(-value, path)


class EnhancedMCTS:
    """
    Enhanced Monte Carlo Tree Search algorithm with:
    - Transposition table
    - Virtual loss for parallelization
    - RAVE (Rapid Action Value Estimation)
    - Progressive widening for large branching factors
    """
    def __init__(self, 
                 game: GameWrapper, 
                 evaluator: Callable[[GameWrapper], Tuple[Dict[int, float], float]],
                 c_puct: float = 1.5,
                 num_simulations: int = 800,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_noise_weight: float = 0.25,
                 temperature: float = 1.0,
                 use_transposition_table: bool = True,
                 transposition_table_size: int = 1000000,
                 num_workers: int = 1,
                 rave_weight: float = 0.0,
                 use_progressive_widening: bool = False):
        """
        Initialize the MCTS algorithm.
        
        Args:
            game: The game to search for moves
            evaluator: Function that evaluates a game state and returns (move_priors, value)
            c_puct: Exploration constant for UCB
            num_simulations: Number of simulations to run per search
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_noise_weight: Weight of Dirichlet noise added to root prior probabilities
            temperature: Temperature parameter for move selection
            use_transposition_table: Whether to use a transposition table
            transposition_table_size: Maximum size of the transposition table
            num_workers: Number of worker threads for parallel MCTS
            rave_weight: Weight for RAVE (0.0 to disable)
            use_progressive_widening: Whether to use progressive widening for large branching factors
        """
        self.game = game
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_noise_weight = dirichlet_noise_weight
        self.temperature = temperature
        self.rave_weight = rave_weight
        self.use_progressive_widening = use_progressive_widening
        
        # Root node
        self.root = EnhancedMCTSNode()
        
        # Transposition table
        self.use_transposition_table = use_transposition_table
        if use_transposition_table:
            self.transposition_table = LRUTranspositionTable(transposition_table_size)
        else:
            self.transposition_table = None
        
        # Parallelization
        self.num_workers = num_workers
        self.simulations_queue = queue.Queue()
        self.executor = None
        if num_workers > 1:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    
    def search(self, game_state: Optional[GameWrapper] = None) -> Dict[int, float]:
        """
        Perform MCTS search from the given game state.
        
        Args:
            game_state: The game state to start the search from (uses self.game if None)
            
        Returns:
            Dictionary mapping moves to search probabilities
        """
        if game_state is not None:
            self.game = game_state
        
        # Ensure the root node is expanded
        if not self.root.is_expanded():
            self._expand_root()
        
        # Perform simulations
        if self.num_workers > 1:
            self._parallel_simulate()
        else:
            for _ in range(self.num_simulations):
                self._simulate()
        
        # Calculate the search probabilities
        search_probs = self._get_search_probabilities()
        
        return search_probs
    
    def _expand_root(self) -> None:
        """
        Expand the root node with priors and optionally add Dirichlet noise.
        This node represents the current game state.
        """
        # Get legal moves
        legal_moves = self.game.get_legal_moves()
        
        # Get prior probabilities from the evaluator
        priors, _ = self.evaluator(self.game)
        
        # Add Dirichlet noise to the priors at the root to encourage exploration
        if self.dirichlet_noise_weight > 0:
            noisy_priors = {}
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
            
            for i, move in enumerate(legal_moves):
                noisy_priors[move] = (1 - self.dirichlet_noise_weight) * priors.get(move, 0.0) + \
                                    self.dirichlet_noise_weight * noise[i]
            
            priors = noisy_priors
        
        # Progressive widening if enabled and many legal moves
        if self.use_progressive_widening and len(legal_moves) > 50:
            # Sort moves by prior probability
            sorted_moves = sorted(priors.items(), key=lambda x: x[1], reverse=True)
            # Take only the top 50 moves
            width = min(50, int(math.sqrt(len(legal_moves))))
            legal_moves = [move for move, _ in sorted_moves[:width]]
            # Renormalize priors
            total = sum(priors[move] for move in legal_moves)
            if total > 0:
                priors = {move: priors[move] / total for move in legal_moves}
        
        # Expand the root node
        self.root.expand(legal_moves, priors)
        
        # Store in transposition table if used
        if self.use_transposition_table:
            hash_value = hash(str(self.game.get_state_tensor().tobytes()))
            self.transposition_table.store(hash_value, self.root)
    
    def _simulate(self) -> None:
        """Perform a single MCTS simulation."""
        # Create a copy of the game for simulation
        game_copy = self.game.clone()
        
        # Start from the root node
        node = self.root
        path = [node]
        moves_in_path = []
        
        # Selection: Traverse the tree to a leaf node
        while node.is_expanded() and not game_copy.is_terminal():
            # Select the best child according to UCB
            move, node = node.select_child(self.c_puct, self.rave_weight)
            
            # Apply virtual loss if multiple workers
            if self.num_workers > 1:
                node.add_virtual_loss()
            
            # Apply the move to the game
            game_copy.apply_move(move)
            
            # Add node and move to the path
            path.append(node)
            moves_in_path.append(move)
            
            # Check transposition table if enabled
            if self.use_transposition_table and not game_copy.is_terminal():
                hash_value = hash(str(game_copy.get_state_tensor().tobytes()))
                transposition_node = self.transposition_table.lookup(hash_value)
                if transposition_node is not None:
                    node = transposition_node
                    path[-1] = node  # Replace the last node in path
        
        # Expansion and Evaluation: If we're not at a terminal state, expand the node
        if not game_copy.is_terminal():
            legal_moves = game_copy.get_legal_moves()
            priors, value = self.evaluator(game_copy)
            
            # Progressive widening if enabled and many legal moves
            if self.use_progressive_widening and len(legal_moves) > 50:
                # Sort moves by prior probability
                sorted_moves = sorted(priors.items(), key=lambda x: x[1], reverse=True)
                # Take only the top sqrt(N) moves
                width = min(50, int(math.sqrt(len(legal_moves))))
                legal_moves = [move for move, _ in sorted_moves[:width]]
                # Renormalize priors
                total = sum(priors[move] for move in legal_moves)
                if total > 0:
                    priors = {move: priors[move] / total for move in legal_moves}
            
            # Expand the node with the legal moves and prior probabilities
            node.expand(legal_moves, priors)
            
            # Store in transposition table if used
            if self.use_transposition_table:
                hash_value = hash(str(game_copy.get_state_tensor().tobytes()))
                self.transposition_table.store(hash_value, node)
        else:
            # If we reached a terminal state, get the value based on the winner
            winner = game_copy.get_winner()
            current_player = game_copy.get_current_player()
            
            if winner == 0:
                # Draw
                value = 0.0
            elif winner == current_player:
                # Current player wins
                value = 1.0
            else:
                # Current player loses
                value = -1.0
        
        # Backup: Update the values of all nodes in the path
        if self.rave_weight > 0.0:
            # If using RAVE, pass the path of moves
            node.backup(value, moves_in_path)
        else:
            # Without RAVE, just pass the value
            node.backup(value)
    
    def _parallel_simulate(self) -> None:
        """
        Perform multiple MCTS simulations in parallel using a thread pool.
        """
        # Submit simulations to the thread pool
        futures = []
        for _ in range(self.num_simulations):
            futures.append(self.executor.submit(self._simulate))
        
        # Wait for all simulations to complete
        concurrent.futures.wait(futures)
    
    def _get_search_probabilities(self) -> Dict[int, float]:
        """
        Calculate the search probabilities based on visit counts and temperature.
        
        Returns:
            Dictionary mapping moves to probabilities
        """
        visits = {move: child.visit_count for move, child in self.root.children.items()}
        total_visits = sum(visits.values())
        
        if total_visits == 0:
            # If no visits, return uniform probabilities
            legal_moves = self.game.get_legal_moves()
            return {move: 1.0 / len(legal_moves) for move in legal_moves}
        
        if self.temperature < 0.01:
            # For very small temperature, just select the move with the most visits
            best_move = max(visits.items(), key=lambda x: x[1])[0]
            probs = {move: 1.0 if move == best_move else 0.0 for move in visits}
        else:
            # Apply the temperature factor with a safer approach
            try:
                # visits ^ (1/temperature) / sum(visits ^ (1/temperature))
                temp_visits = {move: count ** (1 / self.temperature) for move, count in visits.items()}
                total_temp_visits = sum(temp_visits.values())
                probs = {move: count / total_temp_visits for move, count in temp_visits.items()}
            except OverflowError:
                # If overflow occurs, fall back to a safer approach
                # Just normalize the raw visit counts
                probs = {move: count / total_visits for move, count in visits.items()}
        
        return probs
    
    def select_move(self, return_probs: bool = False) -> Union[int, Tuple[int, Dict[int, float]]]:
        """
        Select a move to play based on the search probabilities.
        
        Args:
            return_probs: Whether to return the search probabilities as well
            
        Returns:
            The selected move, or a tuple of (move, probabilities) if return_probs is True
        """
        probs = self.search()
        
        # Select a move based on the probabilities
        moves = list(probs.keys())
        move_probs = list(probs.values())
        move = np.random.choice(moves, p=move_probs)
        
        if return_probs:
            return move, probs
        return move
    
    def update_with_move(self, move: int) -> None:
        """
        Update the tree with the given move.
        If the move exists in the tree, make the corresponding child the new root.
        Otherwise, reset the tree with a new root.
        
        Args:
            move: The move to update with
        """
        if move in self.root.children:
            # Make the child the new root
            self.root = self.root.children[move]
            self.root.parent = None  # Disconnect from the old parent
        else:
            # If the move doesn't exist in the tree, reset the tree
            self.root = EnhancedMCTSNode()
        
        # Clear the transposition table to free memory
        if self.use_transposition_table:
            self.transposition_table.clear()