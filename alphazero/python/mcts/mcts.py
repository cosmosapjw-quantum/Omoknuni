import math
import numpy as np
import time
from typing import List, Dict, Tuple, Callable, Optional, Any, Union
import random

from alphazero.python.games.game_base import GameWrapper


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search.
    """
    def __init__(self, prior: float = 0.0, parent=None, move: int = -1):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.parent = parent
        self.move = move  # The move that led to this state from parent
        self.children: Dict[int, 'MCTSNode'] = {}  # Maps moves to child nodes
        
    def is_expanded(self) -> bool:
        """Check if the node has been expanded."""
        return len(self.children) > 0
    
    def value(self) -> float:
        """Calculate the mean value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visit_count: int, c_puct: float) -> float:
        """
        Calculate the Upper Confidence Bound (UCB) score for this node.
        
        Args:
            parent_visit_count: Number of visits to the parent node
            c_puct: Exploration constant
            
        Returns:
            The UCB score
        """
        # Exploration term: c_puct * prior * sqrt(parent_visit_count) / (1 + visit_count)
        exploration = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        
        # Exploitation term: Q(s,a)
        exploitation = self.value()
        
        return exploitation + exploration
    
    def select_child(self, c_puct: float) -> Tuple[int, 'MCTSNode']:
        """
        Select the child with the highest UCB score.
        
        Args:
            c_puct: Exploration constant
            
        Returns:
            Tuple of (move, child_node)
        """
        # Find the move with the highest UCB score
        best_score = float('-inf')
        best_move = -1
        best_child = None
        
        for move, child in self.children.items():
            score = child.ucb_score(self.visit_count, c_puct)
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
        for move in moves:
            # Get the prior probability for this move, default to 0 if not in priors
            prior = priors.get(move, 0.0)
            self.children[move] = MCTSNode(prior=prior, parent=self, move=move)
    
    def backup(self, value: float) -> None:
        """
        Update the node and its ancestors with the given value.
        
        Args:
            value: The value to back up
        """
        # Update this node
        self.visit_count += 1
        self.value_sum += value
        
        # Update parent node with the negative of the value (for alternating players)
        if self.parent is not None:
            self.parent.backup(-value)


class MCTS:
    """
    Monte Carlo Tree Search algorithm.
    """
    def __init__(self, 
                 game: GameWrapper, 
                 evaluator: Callable[[GameWrapper], Tuple[Dict[int, float], float]],
                 c_puct: float = 1.0,
                 num_simulations: int = 800,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_noise_weight: float = 0.25,
                 temperature: float = 1.0):
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
        """
        self.game = game
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_noise_weight = dirichlet_noise_weight
        self.temperature = temperature
        self.root = MCTSNode()
    
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
        for _ in range(self.num_simulations):
            self._simulate()
        
        # Calculate the search probabilities
        search_probs = self._get_search_probabilities()
        
        return search_probs
    
    def _expand_root(self) -> None:
        """Expand the root node with priors and optionally add Dirichlet noise."""
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
        
        # Expand the root node
        self.root.expand(legal_moves, priors)
    
    def _simulate(self) -> None:
        """Perform a single MCTS simulation."""
        # Create a copy of the game for simulation
        game_copy = self.game.clone()
        
        # Start from the root node
        node = self.root
        path = [node]
        
        # Selection: Traverse the tree to a leaf node
        while node.is_expanded() and not game_copy.is_terminal():
            # Select the best child according to UCB
            move, node = node.select_child(self.c_puct)
            
            # Apply the move to the game
            game_copy.apply_move(move)
            
            # Add node to the path
            path.append(node)
        
        # Expansion and Evaluation: If we're not at a terminal state, expand the node
        if not game_copy.is_terminal():
            legal_moves = game_copy.get_legal_moves()
            priors, value = self.evaluator(game_copy)
            
            # Expand the node with the legal moves and prior probabilities
            node.expand(legal_moves, priors)
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
        node.backup(value)
    
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
            self.root = MCTSNode()