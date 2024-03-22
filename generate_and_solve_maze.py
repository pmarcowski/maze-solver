# -*- coding: utf-8 -*-

__author__ = "Przemyslaw Marcowski"
__copyright__ = "Copyright 2024 Przemyslaw Marcowski"
__license__ = "MIT"
__email__ = "p.marcowski@gmail.com"

"""
Maze generation and solving

This script generates a random maze using the Recursive Backtracking algorithm,
solves the maze using the selected path finding algorithm (BFS, DFS, A*, or Dijkstra),
and visualizes the maze and the solution path using Matplotlib animation.

The maze is represented as a 2D NumPy array, where walls are denoted by 1 and
paths are denoted by 0. The entry point is at the top-left corner, and the exit
point is at the bottom-right corner.

The selected path finding algorithm finds a path from the entry point to the exit point.
The solution path is then animated on the visualized maze.

The script accepts command-line arguments to specify the path finding algorithm
and the dimension of the maze.

# This code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""

import numpy as np
import random
from queue import Queue, PriorityQueue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse


def generate_maze(dim):
    """
    Generates a random maze using the Recursive Backtracking algorithm.

    Args:
        dim: The dimension of the maze (dim x dim).

    Returns:
        A 2D NumPy array representing the maze.
    """
    # Create a grid filled with walls
    maze = np.ones((dim * 2 + 1, dim * 2 + 1))
    
    # Define the starting point
    x, y = (0, 0)
    maze[2 * x + 1, 2 * y + 1] = 0
    
    # Initialize the stack with the starting point
    stack = [(x, y)]
    
    while len(stack) > 0:
        x, y = stack[-1]
        
        # Define possible directions
        directions = [(dx, dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < dim and 0 <= ny < dim and maze[2 * nx + 1, 2 * ny + 1] == 1:
                maze[2 * nx + 1, 2 * ny + 1] = 0
                maze[2 * x + 1 + dx, 2 * y + 1 + dy] = 0
                stack.append((nx, ny))
                break
        else:
            stack.pop()
    
    # Create an entrance and an exit
    maze[1, 0] = 0
    maze[-2, -1] = 0
    
    return maze


def solve_maze(maze, algorithm):
    """
    Solves the maze using the selected path finding algorithm.

    Args:
        maze: A 2D NumPy array representing the maze.
        algorithm: The path finding algorithm to use ('bfs', 'dfs', 'astar', or 'dijkstra').

    Returns:
        A list of coordinates representing the solution path from the entry to the exit.
    """
    # Call appropriate algorithm function based on selection
    if algorithm == 'bfs':
        return solve_maze_bfs(maze)
    elif algorithm == 'dfs':
        return solve_maze_dfs(maze)
    elif algorithm == 'astar':
        return solve_maze_astar(maze)
    elif algorithm == 'dijkstra':
        return solve_maze_dijkstra(maze)


def solve_maze_bfs(maze):
    """
    Solves the maze using the Breadth-First Search (BFS) algorithm.

    Args:
        maze: A 2D NumPy array representing the maze.

    Returns:
        A list of coordinates representing the solution path from the entry to the exit.
    """
    # Define possible directions
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Define starting and ending points
    start = (1, 1)
    end = (maze.shape[0] - 2, maze.shape[1] - 2)
    
    # Initialize visited array and queue
    visited = np.zeros_like(maze, dtype=bool)
    visited[start] = True
    queue = Queue()
    queue.put((start, []))
    
    while not queue.empty():
        (node, path) = queue.get()
        for dx, dy in directions:
            next_node = (node[0] + dx, node[1] + dy)
            if next_node == end:
                return path + [next_node]
            if (0 <= next_node[0] < maze.shape[0] and 0 <= next_node[1] < maze.shape[1]
                    and maze[next_node] == 0 and not visited[next_node]):
                visited[next_node] = True
                queue.put((next_node, path + [next_node]))


def solve_maze_dfs(maze):
    """
    Solves the maze using the Depth-First Search (DFS) algorithm.

    Args:
        maze: A 2D NumPy array representing the maze.

    Returns:
        A list of coordinates representing the solution path from the entry to the exit.
    """
    # Define possible directions
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Define starting and ending points
    start = (1, 1)
    end = (maze.shape[0] - 2, maze.shape[1] - 2)
    
    # Initialize visited array and stack
    visited = np.zeros_like(maze, dtype=bool)
    stack = [(start, [])]
    
    while len(stack) > 0:
        (node, path) = stack.pop()
        if node == end:
            return path + [node]
        if not visited[node]:
            visited[node] = True
            for dx, dy in directions:
                next_node = (node[0] + dx, node[1] + dy)
                if (0 <= next_node[0] < maze.shape[0] and 0 <= next_node[1] < maze.shape[1]
                        and maze[next_node] == 0):
                    stack.append((next_node, path + [node]))


def solve_maze_astar(maze):
    """
    Solves the maze using the A* Search algorithm.

    Args:
        maze: A 2D NumPy array representing the maze.

    Returns:
        A list of coordinates representing the solution path from the entry to the exit.
    """
    # Define possible directions
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Define starting and ending points
    start = (1, 1)
    end = (maze.shape[0] - 2, maze.shape[1] - 2)
    
    # Define heuristic function (Manhattan distance)
    def heuristic(node):
        return abs(node[0] - end[0]) + abs(node[1] - end[1])
    
    # Initialize visited array, cost array, and priority queue
    visited = np.zeros_like(maze, dtype=bool)
    cost = np.full_like(maze, np.inf)
    cost[start] = 0
    queue = PriorityQueue()
    queue.put((0, start, []))
    
    while not queue.empty():
        (_, node, path) = queue.get()
        if node == end:
            return path + [node]
        if not visited[node]:
            visited[node] = True
            for dx, dy in directions:
                next_node = (node[0] + dx, node[1] + dy)
                if (0 <= next_node[0] < maze.shape[0] and 0 <= next_node[1] < maze.shape[1]
                        and maze[next_node] == 0):
                    new_cost = cost[node] + 1
                    if new_cost < cost[next_node]:
                        cost[next_node] = new_cost
                        queue.put((new_cost + heuristic(next_node), next_node, path + [node]))


def solve_maze_dijkstra(maze):
    """
    Solves the maze using Dijkstra's algorithm.

    Args:
        maze: A 2D NumPy array representing the maze.

    Returns:
        A list of coordinates representing the solution path from the entry to the exit.
    """
    # Define possible directions
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Define starting and ending points
    start = (1, 1)
    end = (maze.shape[0] - 2, maze.shape[1] - 2)
    
    # Initialize visited array, cost array, and priority queue
    visited = np.zeros_like(maze, dtype=bool)
    cost = np.full_like(maze, np.inf)
    cost[start] = 0
    queue = PriorityQueue()
    queue.put((0, start, []))
    
    while not queue.empty():
        (_, node, path) = queue.get()
        if node == end:
            return path + [node]
        if not visited[node]:
            visited[node] = True
            for dx, dy in directions:
                next_node = (node[0] + dx, node[1] + dy)
                if (0 <= next_node[0] < maze.shape[0] and 0 <= next_node[1] < maze.shape[1]
                        and maze[next_node] == 0):
                    new_cost = cost[node] + 1
                    if new_cost < cost[next_node]:
                        cost[next_node] = new_cost
                        queue.put((new_cost, next_node, path + [node]))


def animate_maze(maze, path=None, algorithm=None):
    """
    Draws the maze and animates the solution path if provided. 
    Solution path is drawn in color depending on the algorithm selected.
    
    Algorithm Colors:
        - BFS: Red
        - DFS: Blue
        - A*: Green
        - Dijkstra: Orange

    Args:
        maze: A 2D NumPy array representing the maze.
        path: A list of coordinates representing the solution path (optional).
        algorithm: The path finding algorithm used (optional).
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set border color to white
    fig.patch.set_edgecolor('white')
    fig.patch.set_linewidth(0)

    # Display maze
    ax.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])

    # Set title with algorithm name
    if algorithm:
        fig.text(0.5, 0.90, f"Solving maze using {algorithm.upper()} algorithm", fontsize=24, ha='center')

    # Add text for displaying current step and total steps
    step_text = fig.text(0.5, 0.07, "", fontsize=18, ha='center')

    # Assign colors to each algorithm
    algorithm_colors = {
        'bfs': 'red',
        'dfs': 'blue',
        'astar': 'green',
        'dijkstra': 'orange'
    }

    # Get color based on selected algorithm
    path_color = algorithm_colors.get(algorithm.lower(), 'red')

    # Prepare for path animation
    if path is not None:
        line, = ax.plot([], [], color=path_color, linewidth=2)

        def init():
            line.set_data([], [])
            step_text.set_text("")
            return line, step_text

        def update(frame):
            x, y = path[frame]
            line.set_data(*zip(*[(p[1], p[0]) for p in path[:frame + 1]]))
            step_text.set_text(f"Time: {frame + 1}/{len(path)}")
            return line, step_text

        ani = animation.FuncAnimation(fig, update, frames=range(len(path)),
                                      init_func=init, repeat=False, interval=50, blit=False)
        plt.show()

    # Draw entry and exit arrows
    ax.arrow(0, 1, .4, 0, fc='green', ec='green', head_width=0.3, head_length=0.3)
    ax.arrow(maze.shape[1] - 1, maze.shape[0] - 2, 0.4, 0, fc='blue', ec='blue',
             head_width=0.3, head_length=0.3)


# Parse arguments
parser = argparse.ArgumentParser(description='Maze Generation and Solving')

parser.add_argument('--algorithm', '-a', default='bfs', choices=['bfs', 'dfs', 'astar', 'dijkstra'],
                    help='Path finding algorithm to use (default: bfs)')

parser.add_argument('--dimension', '-d', type=int, default=50,
                    help='Dimension of the maze (default: 50)')

args = parser.parse_args()

# Get maze dimension
dim = args.dimension

# Generate the maze
maze = generate_maze(dim)

# Get path finding algorithm to use
algorithm = args.algorithm

# Solve maze using selected algorithm
solution = solve_maze(maze, algorithm)

# Draw maze and animate solution path
animate_maze(maze, solution, algorithm)