# -*- coding: utf-8 -*-

__author__ = "Przemyslaw Marcowski"
__email__ = "p.marcowski@gmail.com"
__license__ = "MIT"

"""
Maze Generation and Solving Module

This module provides functionalities to generate random mazes using the Growing Tree
algorithm and solve them using various path-finding algorithms such as Breadth-First
Search (BFS), Depth-First Search (DFS), A* Search, and Dijkstra's Algorithm.

The main class, MazeSolver, offers static methods for maze generation and solving,
as well as instance methods for specific solving algorithms.
"""

import numpy as np
import random
from typing import List, Tuple
from collections import deque
import heapq


class MazeSolver:
    """
    A class that provides maze generation and solving capabilities.

    This class contains static methods for generating mazes and solving them
    using various algorithms, as well as instance methods for specific
    solving algorithms.
    """

    @staticmethod
    def generate_maze(width: int, height: int, randomness: float = 0.1) -> np.ndarray:
        """
        Generate a maze using the Growing Tree algorithm.

        Args:
            width (int): The width of the maze.
            height (int): The height of the maze.
            randomness (float): Factor controlling the randomness of the maze (0 to 1).

        Returns:
            np.ndarray: A 2D NumPy array representing the maze (0 for paths, 1 for walls).
        """
        # Initialize maze with all walls
        maze = np.ones((height * 2 + 1, width * 2 + 1), dtype=int)
        
        # Start with random cell
        cells = [(random.randint(0, height - 1), random.randint(0, width - 1))]
        
        # Possible directions: right, down, left, up
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while cells:
            # Choose a cell from list
            index = random.randint(0, len(cells) - 1) if random.random() < randomness else -1
            y, x = cells[index]
            maze[y*2+1, x*2+1] = 0  # Carve current cell
            
            # Find unvisited neighbors
            neighbors = [(y+dy, x+dx) for dy, dx in directions 
                         if 0 <= y+dy < height and 0 <= x+dx < width and maze[(y+dy)*2+1, (x+dx)*2+1] == 1]
            
            if neighbors:
                # Choose a random unvisited neighbor
                ny, nx = random.choice(neighbors)
                maze[y+ny+1, x+nx+1] = 0  # Carve passage to neighbor
                cells.append((ny, nx))
            else:
                # No unvisited neighbors, remove cell from list
                cells.pop(index)

        # Introduce additional paths to create multiple solutions
        for _ in range(int((width * height) * randomness * 0.1)):
            y, x = random.randint(1, height*2-1), random.randint(1, width*2-1)
            if maze[y, x] == 1:
                maze[y, x] = 0

        return maze

    @staticmethod
    def solve_maze(maze: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], algorithm: str = 'bfs') -> List[Tuple[int, int]]:
        """
        Solve the maze using the specified algorithm.

        Args:
            maze (np.ndarray): A 2D NumPy array representing the maze.
            start (Tuple[int, int]): The starting position (row, col).
            end (Tuple[int, int]): The ending position (row, col).
            algorithm (str): The solving algorithm to use ('bfs', 'dfs', 'astar', or 'dijkstra').

        Returns:
            List[Tuple[int, int]]: A list of positions representing the solution path.

        Raises:
            ValueError: If an unknown algorithm is specified.
        """
        solver = MazeSolver(maze, start, end)
        if algorithm == 'bfs':
            return solver._bfs()
        elif algorithm == 'dfs':
            return solver._dfs()
        elif algorithm == 'astar':
            return solver._astar()
        elif algorithm == 'dijkstra':
            return solver._dijkstra()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def __init__(self, maze: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]):
        """
        Initialize the MazeSolver instance.

        Args:
            maze (np.ndarray): A 2D NumPy array representing the maze.
            start (Tuple[int, int]): The starting position (row, col).
            end (Tuple[int, int]): The ending position (row, col).
        """
        self.maze = maze
        self.start = start
        self.end = end
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

    def is_valid_move(self, position: Tuple[int, int]) -> bool:
        """
        Check if a move to the given position is valid.

        A move is valid if it's within maze boundaries and not a wall.

        Args:
            position (Tuple[int, int]): The position to check (row, col).

        Returns:
            bool: True if the move is valid, False otherwise.
        """
        y, x = position
        return (0 <= y < self.maze.shape[0] and 0 <= x < self.maze.shape[1] and
                self.maze[y, x] == 0)

    def _bfs(self) -> List[Tuple[int, int]]:
        """
        Solve the maze using Breadth-First Search (BFS).

        Returns:
            List[Tuple[int, int]]: A list of positions representing the solution path.
        """
        queue = deque([(self.start, [self.start])])
        visited = set([self.start])

        while queue:
            current, path = queue.popleft()
            if current == self.end:
                return path

            for dy, dx in self.directions:
                next_pos = (current[0] + dy, current[1] + dx)
                if self.is_valid_move(next_pos) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))

        return []  # No path found

    def _dfs(self) -> List[Tuple[int, int]]:
        """
        Solve the maze using Depth-First Search (DFS).

        Returns:
            List[Tuple[int, int]]: A list of positions representing the solution path.
        """
        stack = [(self.start, [self.start])]
        visited = set()

        while stack:
            current, path = stack.pop()
            if current == self.end:
                return path

            if current not in visited:
                visited.add(current)
                for dy, dx in self.directions:
                    next_pos = (current[0] + dy, current[1] + dx)
                    if self.is_valid_move(next_pos) and next_pos not in visited:
                        stack.append((next_pos, path + [next_pos]))

        return []  # No path found

    def _astar(self) -> List[Tuple[int, int]]:
        """
        Solve the maze using A* Search.

        Returns:
            List[Tuple[int, int]]: A list of positions representing the solution path.
        """
        def heuristic(a, b):
            return abs(b[0] - a[0]) + abs(b[1] - a[1])

        open_set = [(0, self.start, [self.start])]
        g_score = {self.start: 0}
        f_score = {self.start: heuristic(self.start, self.end)}
        visited = set()

        while open_set:
            current_f, current, path = heapq.heappop(open_set)
            if current == self.end:
                return path

            if current in visited:
                continue

            visited.add(current)

            for dy, dx in self.directions:
                next_pos = (current[0] + dy, current[1] + dx)
                if self.is_valid_move(next_pos):
                    tentative_g = g_score[current] + 1
                    if tentative_g < g_score.get(next_pos, float('inf')):
                        g_score[next_pos] = tentative_g
                        f_score[next_pos] = tentative_g + heuristic(next_pos, self.end)
                        heapq.heappush(open_set, (f_score[next_pos], next_pos, path + [next_pos]))

        return []  # No path found

    def _dijkstra(self) -> List[Tuple[int, int]]:
        """
        Solve the maze using Dijkstra's Algorithm.

        Returns:
            List[Tuple[int, int]]: A list of positions representing the solution path.
        """
        queue = [(0, self.start, [self.start])]
        visited = set()

        while queue:
            cost, current, path = heapq.heappop(queue)
            if current == self.end:
                return path

            if current in visited:
                continue

            visited.add(current)

            for dy, dx in self.directions:
                next_pos = (current[0] + dy, current[1] + dx)
                if self.is_valid_move(next_pos):
                    heapq.heappush(queue, (cost + 1, next_pos, path + [next_pos]))

        return []  # No path found


# Example usage
if __name__ == "__main__":
    # Generate maze
    maze = MazeSolver.generate_maze(20, 20, randomness=0.1)
    
    # Define start and end points
    start = (1, 1)
    end = (18, 18)
    
    # Solve maze using BFS
    path = MazeSolver.solve_maze(maze, start, end, algorithm='bfs')
    
    print(f"Maze generated with size 20x20")
    print(f"Path found from {start} to {end} with length: {len(path)}")
    