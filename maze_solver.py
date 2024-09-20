# -*- coding: utf-8 -*-

__author__ = "Przemyslaw Marcowski"
__email__ = "p.marcowski@gmail.com"
__license__ = "MIT"

"""
Maze Generation and Solving Module.

This module provides functionalities to generate random mazes using the Recursive
Backtracking algorithm and solve them using various path-finding algorithms
such as Breadth-First Search (BFS), Depth-First Search (DFS), A* Search, and
Dijkstra's Algorithm.
"""

import yaml
import numpy as np
import random
from heapq import heappush, heappop
from collections import deque
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load configuration from YAML file
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Extract configuration values
FIGURE_SIZE = (config['figure_size']['width'], config['figure_size']['height'])
ALGORITHM_COLORS = config['algorithm_colors']
DEFAULT_MAZE_DIMENSION = config['default_maze_dimension']
DEFAULT_CYCLE_PROBABILITY = config['default_cycle_probability']
ANIMATION_INTERVAL = config['animation']['interval']


def generate_perfect_maze(dim: int) -> np.ndarray:
    """
    Generate a perfect maze using the Recursive Backtracking algorithm.

    A perfect maze has exactly one unique path between any two points.

    Args:
        dim (int): The dimension of the maze (maze will be dim x dim).

    Returns:
        np.ndarray: A 2D NumPy array representing the maze.
                    1 represents walls, 0 represents paths.
    """
    # Initialize maze with all walls
    maze = np.ones((dim * 2 + 1, dim * 2 + 1), dtype=int)

    # Starting point at (0, 0)
    start_x, start_y = 0, 0
    maze[2 * start_x + 1, 2 * start_y + 1] = 0  # Carve starting cell
    stack = [(start_x, start_y)]  # Initialize stack with starting cell

    while stack:
        x, y = stack[-1]  # Current cell
        # Possible directions: right, down, left, up
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)  # Randomize directions for maze randomness

        for dx, dy in directions:
            nx, ny = x + dx, y + dy  # Neighbor cell coordinates
            # Check if neighbor is within bounds and unvisited
            if 0 <= nx < dim and 0 <= ny < dim:
                if maze[2 * nx + 1, 2 * ny + 1] == 1:
                    # Carve path between current cell and neighbor
                    maze[2 * nx + 1, 2 * ny + 1] = 0
                    maze[2 * x + 1 + dx, 2 * y + 1 + dy] = 0
                    stack.append((nx, ny))  # Move to neighbor
                    break  # Continue with new cell
        else:
            # No unvisited neighbors, backtrack
            stack.pop()

    # Create entrance and exit (optional)
    # Uncomment the following lines to add entrance and exit
    # maze[1, 0] = 0    # Entrance
    # maze[-2, -1] = 0  # Exit

    return maze


def introduce_cycles(maze: np.ndarray, cycle_probability: float = 0.1) -> np.ndarray:
    """
    Introduce cycles into a perfect maze by removing walls between adjacent paths.

    This converts the maze from being perfect (no cycles) to imperfect (with cycles),
    allowing multiple paths between points.

    Args:
        maze (np.ndarray): The perfect maze array.
        cycle_probability (float): Probability of removing a wall to create a cycle (0 <= p <= 1).

    Returns:
        np.ndarray: The maze array with cycles.
    """
    dim_y, dim_x = maze.shape  # Dimensions of maze

    # Iterate through each cell in maze grid
    for y in range(1, dim_y - 1, 2):
        for x in range(1, dim_x - 1, 2):
            # Potential walls to remove: East and South
            directions = [(0, 2), (2, 0)]  # East, South
            random.shuffle(directions)  # Randomize directions

            for dy, dx in directions:
                ny, nx = y + dy, x + dx  # Adjacent cell coordinates
                # Skip if adjacent cell is out of bounds
                if ny >= dim_y - 1 or nx >= dim_x - 1:
                    continue

                # Check if wall between cells exists (indicating no connection)
                if maze[y + dy//2, x + dx//2] == 1:
                    # With given probability, remove the wall to create cycle
                    if random.random() < cycle_probability:
                        maze[y + dy//2, x + dx//2] = 0  # Remove wall

    return maze


def generate_imperfect_maze(dim: int, cycle_probability: float = 0.1) -> np.ndarray:
    """
    Generate an imperfect maze with multiple paths by first creating a perfect maze
    and then introducing cycles.

    Args:
        dim (int): The dimension of the maze (maze will be dim x dim).
        cycle_probability (float): Probability of removing a wall to create a cycle (0 <= p <= 1).

    Returns:
        np.ndarray: A 2D NumPy array representing the imperfect maze.
                    1 represents walls, 0 represents paths.
    """
    maze = generate_perfect_maze(dim)  # Generate perfect maze
    maze = introduce_cycles(maze, cycle_probability)  # Introduce cycles
    return maze


class MazeSolver:
    """
    Base class for maze solving algorithms.
    """

    def __init__(self, maze: np.ndarray, start: tuple, end: tuple):
        """
        Initialize the MazeSolver.

        Args:
            maze (np.ndarray): The maze to solve.
            start (tuple): The starting position (row, col).
            end (tuple): The ending position (row, col).
        """
        self.maze = maze
        self.start = start
        self.end = end
        self.visited = set()  # Track visited positions to avoid cycles
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Possible movement directions

    def is_valid_move(self, position: tuple) -> bool:
        """
        Check if a move to the given position is valid.

        A move is valid if it's within maze boundaries, not a wall, and not already visited.

        Args:
            position (tuple): The position to check (row, col).

        Returns:
            bool: True if the move is valid, False otherwise.
        """
        row, col = position
        return (
            0 <= row < self.maze.shape[0]
            and 0 <= col < self.maze.shape[1]
            and self.maze[position] == 0
            and position not in self.visited
        )

    def solve(self):
        """
        Solve the maze. To be implemented by subclasses.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")


class BFSSolver(MazeSolver):
    """
    Breadth-First Search (BFS) maze solver.
    """

    def solve(self):
        """
        Solve the maze using BFS.

        BFS explores the maze level by level, ensuring the shortest path is found.

        Returns:
            list: The path from start to end, or None if no path is found.
        """
        queue = deque([(self.start, [self.start])])  # Initialize queue with start position and path
        self.visited.add(self.start)  # Mark start as visited

        while queue:
            current_position, path = queue.popleft()  # Dequeue the next position and path

            if current_position == self.end:
                return path  # Found end; return path

            for dy, dx in self.directions:
                next_position = (current_position[0] + dy, current_position[1] + dx)
                if self.is_valid_move(next_position):
                    self.visited.add(next_position)  # Mark new position as visited
                    queue.append((next_position, path + [next_position]))  # Enqueue new position and updated path

        return None  # No path found


class DFSSolver(MazeSolver):
    """
    Depth-First Search (DFS) maze solver.
    """

    def solve(self):
        """
        Solve the maze using DFS.

        DFS explores as far as possible along each branch before backtracking.

        Returns:
            list: The path from start to end, or None if no path is found.
        """
        stack = [(self.start, [self.start])]  # Initialize stack with start position and path

        while stack:
            current_position, path = stack.pop()  # Pop top position and path from stack

            if current_position == self.end:
                return path  # Found end; return path

            if current_position not in self.visited:
                self.visited.add(current_position)  # Mark current position as visited

                for dy, dx in self.directions:
                    next_position = (current_position[0] + dy, current_position[1] + dx)
                    if self.is_valid_move(next_position):
                        stack.append((next_position, path + [next_position]))  # Push new position and updated path

        return None  # No path found


class AStarSolver(MazeSolver):
    """
    A* Search maze solver.
    """

    def heuristic(self, a: tuple, b: tuple) -> int:
        """
        Calculate the Manhattan distance between two points.

        Manhattan distance is used as the heuristic for A*.

        Args:
            a (tuple): First point (row, col).
            b (tuple): Second point (row, col).

        Returns:
            int: The Manhattan distance between the two points.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def solve(self):
        """
        Solve the maze using A* Search.

        A* uses heuristics to find the most promising path efficiently.

        Returns:
            list: The path from start to end, or None if no path is found.
        """
        g_score = {self.start: 0}  # Cost from start to current node
        f_score = {self.start: self.heuristic(self.start, self.end)}  # Estimated total cost
        open_set = [(f_score[self.start], self.start, [self.start])]  # Priority queue with (f_score, position, path)

        while open_set:
            _, current_position, path = heappop(open_set)  # Pop position with lowest f_score

            if current_position == self.end:
                return path  # Found end; return path

            self.visited.add(current_position)  # Mark current position as visited

            for dy, dx in self.directions:
                next_position = (current_position[0] + dy, current_position[1] + dx)
                if self.is_valid_move(next_position):
                    tentative_g_score = g_score[current_position] + 1  # Assume cost between nodes is 1

                    # If better path to next_position is found
                    if tentative_g_score < g_score.get(next_position, float('inf')):
                        g_score[next_position] = tentative_g_score
                        f_score[next_position] = tentative_g_score + self.heuristic(next_position, self.end)
                        heappush(open_set, (f_score[next_position], next_position, path + [next_position]))

        return None  # No path found


class DijkstraSolver(MazeSolver):
    """
    Dijkstra's Algorithm maze solver.
    """

    def solve(self):
        """
        Solve the maze using Dijkstra's Algorithm.

        Dijkstra's explores the maze uniformly, ensuring the shortest path is found.

        Returns:
            list: The path from start to end, or None if no path is found.
        """
        distances = {self.start: 0}  # Cost from start to current node
        open_set = [(0, self.start, [self.start])]  # Priority queue with (distance, position, path)

        while open_set:
            current_distance, current_position, path = heappop(open_set)  # Pop position with lowest distance

            if current_position == self.end:
                return path  # Found end; return path

            self.visited.add(current_position)  # Mark current position as visited

            for dy, dx in self.directions:
                next_position = (current_position[0] + dy, current_position[1] + dx)
                if self.is_valid_move(next_position):
                    tentative_distance = current_distance + 1  # Assume cost between nodes is 1

                    # If better path to next_position is found
                    if tentative_distance < distances.get(next_position, float('inf')):
                        distances[next_position] = tentative_distance
                        heappush(open_set, (tentative_distance, next_position, path + [next_position]))

        return None  # No path found


def solve_maze(maze: np.ndarray, algorithm: str, start: tuple, end: tuple) -> list:
    """
    Solve the maze using the specified path-finding algorithm.

    Args:
        maze (np.ndarray): A 2D NumPy array representing the maze.
        algorithm (str): The path-finding algorithm to use ('bfs', 'dfs', 'astar', or 'dijkstra').
        start (tuple): The starting position as (row, column).
        end (tuple): The ending position as (row, column).

    Returns:
        list: A list of positions representing the solution path from start to end.

    Raises:
        ValueError: If an unknown algorithm is specified.
    """
    # Mapping of algorithm names to their corresponding solver classes
    solver_classes = {
        'bfs': BFSSolver,
        'dfs': DFSSolver,
        'astar': AStarSolver,
        'dijkstra': DijkstraSolver
    }

    solver_class = solver_classes.get(algorithm.lower())
    if not solver_class:
        raise ValueError(f"Unknown algorithm: {algorithm}")  # Handle invalid algorithm input

    solver = solver_class(maze, start, end)  # Instantiate solver
    return solver.solve()  # Execute solve method


def get_start_end_positions(maze: np.ndarray) -> tuple:
    """
    Display the maze and allow the user to select start and end positions via mouse clicks.

    Args:
        maze (np.ndarray): A 2D NumPy array representing the maze.

    Returns:
        tuple: A tuple containing start and end positions as (row, column).

    Raises:
        Exception: If two valid positions are not selected.
    """
    positions = []  # List to store selected positions

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)  # Create matplotlib figure and axis
    ax.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')  # Display maze
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_title("Click on two points to select start and end positions")  # Set plot title

    def onclick(event):
        """
        Handle mouse click events to select start and end positions.

        Args:
            event: The mouse event.
        """
        if event.inaxes != ax:
            return  # Ignore clicks outside maze plot
        if event.xdata is None or event.ydata is None:
            return  # Ignore clicks with no data
        x, y = int(round(event.xdata)), int(round(event.ydata))  # Get integer coordinates
        # Check if click is within maze bounds
        if 0 <= x < maze.shape[1] and 0 <= y < maze.shape[0]:
            if maze[y, x] == 0:
                positions.append((y, x))  # Add position to list
                color = 'go' if len(positions) == 1 else 'ro'  # Green for start, red for end
                ax.plot(x, y, color, markersize=10)  # Mark the position on the maze
                fig.canvas.draw()  # Redraw figure to show marker
                if len(positions) == 2:
                    plt.close(fig)  # Close figure after selecting points
            else:
                print("Clicked on a wall. Please select a path cell.")  # Invalid selection
        else:
            print("Clicked outside the maze. Please click inside the maze.")  # Out of bounds

    # Connect onclick function to mouse click events
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)  # Display plot and wait for user interaction

    # Ensure that two valid positions have been selected
    if len(positions) != 2:
        raise Exception("Two valid positions were not selected.")

    start, end = positions[0], positions[1]  # Unpack start and end positions
    return start, end


def animate_maze(maze: np.ndarray, path: list = None, algorithm: str = None, 
                start: tuple = None, end: tuple = None, callback: callable = None):
    """
    Draw the maze and animate the solution path if provided.

    Args:
        maze (np.ndarray): The maze array.
        path (list, optional): The solution path as a list of positions.
        algorithm (str, optional): The algorithm used.
        start (tuple, optional): Starting position.
        end (tuple, optional): Ending position.
        callback (callable, optional): Function to call when the animation ends.

    Returns:
        tuple: A tuple containing the animation object and the timer object.
    """
    fig = plt.figure(figsize=FIGURE_SIZE)  # Create matplotlib figure
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 3])  # Define grid layout
    ax_info = fig.add_subplot(gs[0])  # Axis for algorithm info
    ax_maze = fig.add_subplot(gs[1])  # Axis for maze visualization

    ax_maze.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')  # Display maze
    ax_maze.set_xticks([])  # Remove x-axis ticks
    ax_maze.set_yticks([])  # Remove y-axis ticks

    # Configure info axis
    ax_info.axis('off')  # Hide axes
    ax_info.text(0.1, 0.9, "Solving maze...", fontsize=14, fontweight='bold', transform=ax_info.transAxes)
    ax_info.text(0.1, 0.85, f"Algorithm: {algorithm.upper()}", fontsize=14, transform=ax_info.transAxes)
    solution_length_text = ax_info.text(0.1, 0.8, "", fontsize=14, transform=ax_info.transAxes)
    step_text = ax_info.text(0.1, 0.75, "", fontsize=14, transform=ax_info.transAxes)

    # Determine color for path based on algorithm used
    path_color = ALGORITHM_COLORS.get(algorithm.lower(), 'red')

    # Plot start and end points if provided
    if start is not None:
        ax_maze.plot(start[1], start[0], 'go', markersize=10)  # Green marker for start
    if end is not None:
        ax_maze.plot(end[1], end[0], 'ro', markersize=10)  # Red marker for end

    if path is not None:
        # Initialize line object for path animation
        line, = ax_maze.plot([], [], color=path_color, linewidth=2)

        def init():
            """
            Initialize the animation by clearing the path line and resetting texts.
            """
            line.set_data([], [])  # Clear line data
            step_text.set_text("")  # Clear step text
            solution_length_text.set_text("")  # Clear solution length text
            return line, step_text, solution_length_text

        def update(frame):
            """
            Update function for each frame of the animation.

            Args:
                frame (int): The current frame number.
            """
            # Extract path up to current frame
            y_coords = [p[0] for p in path[:frame + 1]]
            x_coords = [p[1] for p in path[:frame + 1]]
            line.set_data(x_coords, y_coords)  # Update line data
            step_text.set_text(f"Step: {frame + 1}")  # Update step count
            solution_length_text.set_text(f"Solution length: {len(path)}")  # Update solution length
            return line, step_text, solution_length_text

        # Create animation object
        ani = animation.FuncAnimation(
            fig, update, frames=range(len(path)),
            init_func=init, repeat=False, interval=ANIMATION_INTERVAL, blit=False
        )

        plt.show(block=False)  # Display plot without blocking

        # Calculate total duration for animation
        total_duration = len(path) * ANIMATION_INTERVAL
        timer = fig.canvas.new_timer(interval=total_duration)
        timer.single_shot = True  # Ensure timer fires only once

        if callback is not None:
            timer.add_callback(callback)  # Add callback to be executed after animation

        timer.start()  # Start timer

        return ani, timer  # Return animation and timer objects
    else:
        plt.show(block=False)  # Display plot without animation
        return None, None


def main():
    """
    Main function to parse arguments, generate the maze, and handle the user interaction loop.
    """
    # Initialize argument parser for command-line options
    parser = argparse.ArgumentParser(description='Maze Generation and Solving')
    parser.add_argument('--algorithm', '-a', default='bfs',
                        choices=['bfs', 'dfs', 'astar', 'dijkstra'],
                        help='Path-finding algorithm to use (default: bfs)')
    parser.add_argument('--dimension', '-d', type=int, default=DEFAULT_MAZE_DIMENSION,
                        help=f'Dimension of the maze (default: {DEFAULT_MAZE_DIMENSION})')
    parser.add_argument('--cycles', '-c', type=float, default=DEFAULT_CYCLE_PROBABILITY,
                        help=f'Probability of cycles in the maze (default: {DEFAULT_CYCLE_PROBABILITY})')
    args = parser.parse_args()  # Parse arguments

    dim = args.dimension  # Maze dimension
    algorithm = args.algorithm  # Selected algorithm
    cycle_probability = args.cycles  # Cycle probability

    # Generate imperfect maze based on provided dimensions and cycle probability
    maze = generate_imperfect_maze(dim, cycle_probability)

    continue_running = True  # Control variable for interaction loop
    animations = []  # List to store animation objects
    timers = []  # List to store timer objects

    while continue_running:
        try:
            # Prompt user to select start and end positions via mouse clicks
            start, end = get_start_end_positions(maze)

            # Solve maze using selected algorithm
            solution = solve_maze(maze, algorithm, start, end)

            if solution is None:
                print("No path found between the selected points.")  # Inform user of no solution
            else:
                def animation_callback():
                    """
                    Callback function to handle user input after animation ends.
                    """
                    nonlocal continue_running  # Access outer scope variable
                    reset = input("Do you want to select new start and end positions? (y/n): ").strip().lower()
                    if reset != 'y':
                        print("Exiting the program.")
                        plt.close('all')  # Close all matplotlib windows
                        continue_running = False  # Exit loop
                    else:
                        plt.close('all')  # Close current plot and continue

                # Animate solution path
                ani, timer = animate_maze(maze, solution, algorithm, start, end, callback=animation_callback)

                if ani is not None:
                    animations.append(ani)  # Keep reference to prevent garbage collection
                if timer is not None:
                    timers.append(timer)  # Keep reference to prevent garbage collection

                # Keep program running until animation is done or user decides to exit
                while continue_running and plt.get_fignums():
                    plt.pause(0.1)  # Pause briefly to allow animation to proceed
        except Exception as e:
            # Handle exceptions, such as invalid selections or user interruptions
            print(f"An error occurred: {e}")
            reset = input("Do you want to try selecting start and end positions again? (y/n): ").strip().lower()
            if reset != 'y':
                print("Exiting the program.")
                break  # Exit loop


if __name__ == "__main__":
    main()
