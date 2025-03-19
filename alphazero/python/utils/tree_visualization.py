import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

from alphazero.python.mcts.mcts import MCTSNode


def visualize_mcts_tree(root: MCTSNode, 
                        board_size: int, 
                        max_nodes: int = 30, 
                        node_size: int = 500, 
                        figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Visualize the MCTS tree from the given root node.
    
    Args:
        root: The root node of the MCTS tree
        board_size: Size of the game board (to convert move indices to coordinates)
        max_nodes: Maximum number of nodes to display (to avoid cluttering)
        node_size: Size of the nodes in the visualization
        figsize: Figure size (width, height) in inches
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges to the graph via BFS
    queue = [(root, -1, 0)]  # (node, move, depth)
    visited = set()
    node_count = 0
    node_labels = {}
    node_colors = []
    node_sizes = []
    
    while queue and node_count < max_nodes:
        node, move, depth = queue.pop(0)
        
        # Create a unique identifier for this node
        node_id = id(node)
        
        if node_id in visited:
            continue
        
        visited.add(node_id)
        node_count += 1
        
        # Add node to the graph
        G.add_node(node_id)
        
        # Add edge to the parent
        if move != -1:
            parent_id = id(node.parent)
            G.add_edge(parent_id, node_id)
        
        # Create node label
        if move == -1:
            move_str = "Root"
        else:
            row, col = divmod(move, board_size)
            move_str = f"({row},{col})"
        
        value = node.value() if node.visit_count > 0 else 0.0
        node_labels[node_id] = f"{move_str}\nN={node.visit_count}\nV={value:.2f}"
        
        # Calculate node color based on value (red for negative, blue for positive)
        color = (1.0 - (value + 1) / 2, 0.5, (value + 1) / 2)  # (r, g, b)
        node_colors.append(color)
        
        # Calculate node size based on visit count
        if node.visit_count > 0:
            size = node_size * (1 + np.log(node.visit_count) / 5)
        else:
            size = node_size / 2
        node_sizes.append(size)
        
        # Add children to the queue, prioritizing nodes with more visits
        children = sorted(node.children.items(), key=lambda x: x[1].visit_count, reverse=True)
        for child_move, child_node in children:
            queue.append((child_node, child_move, depth + 1))
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Use a hierarchical layout
    pos = nx.spring_layout(G, k=1.5/np.sqrt(len(G.nodes())), iterations=50)
    
    # Draw the graph
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=node_sizes, 
            alpha=0.8, linewidths=1, font_size=10, font_weight='bold',
            edge_color='gray', arrows=True)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, node_labels, font_size=8)
    
    plt.title(f"MCTS Tree Visualization (showing {node_count} nodes)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_policy_board(policy: Dict[int, float], 
                          board_size: int, 
                          figsize: Tuple[int, int] = (8, 8)) -> None:
    """
    Visualize the policy as a heatmap on the board.
    
    Args:
        policy: Dictionary mapping moves to probabilities
        board_size: Size of the game board
        figsize: Figure size (width, height) in inches
    """
    # Create a 2D array for the policy
    policy_array = np.zeros((board_size, board_size))
    
    # Fill the array with probabilities
    for move, prob in policy.items():
        row, col = divmod(move, board_size)
        policy_array[row, col] = prob
    
    # Create the plot
    plt.figure(figsize=figsize)
    plt.imshow(policy_array, cmap='hot', interpolation='nearest', vmin=0, vmax=max(policy.values()))
    plt.colorbar(label='Probability')
    
    # Add grid
    plt.grid(color='black', linestyle='-', linewidth=0.5)
    
    # Set ticks
    plt.xticks(np.arange(board_size))
    plt.yticks(np.arange(board_size))
    
    # Add text annotations for probabilities
    for row in range(board_size):
        for col in range(board_size):
            move = row * board_size + col
            if move in policy and policy[move] > 0.01:
                plt.text(col, row, f"{policy[move]:.2f}", 
                         ha="center", va="center", color="w" if policy_array[row, col] > 0.3 else "k")
    
    plt.title("Move Probabilities")
    plt.tight_layout()
    plt.show()