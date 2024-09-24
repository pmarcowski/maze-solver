# -*- coding: utf-8 -*-

__author__ = "Przemyslaw Marcowski"
__email__ = "p.marcowski@gmail.com"
__license__ = "MIT"

"""
Maze Solver CLI

This module provides a command-line interface for generating mazes,
solving them using various algorithms, and visualizing the results.
It allows users to interactively select start and end points, view
the solution path, and repeat the process with new points.

Usage:
    python main.py [options]

For a list of available options, use:
    python main.py --help
"""

import argparse
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from maze_solver import MazeSolver


def load_config():
    """
    Load configuration from YAML file.

    Returns:
        dict: Configuration settings loaded from the YAML file.
    """
    with open('config.yaml', 'r') as config_file:
        return yaml.safe_load(config_file)


class MazeVisualization:
    """
    A class to handle the visualization of the maze and its solution.

    This class manages the matplotlib figure and axes, handles user
    interactions for selecting start and end points, and animates
    the solution path.
    """

    def __init__(self, maze):
        """
        Initialize the MazeVisualization object.

        Args:
            maze (numpy.ndarray): The maze to visualize.
        """
        self.maze = maze
        self.fig, self.ax = plt.subplots(figsize=(CONFIG['visualization']['figure_size']['width'],
                                                  CONFIG['visualization']['figure_size']['height']))
        self.ax.imshow(maze, cmap='binary')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.start = None
        self.end = None
        self.path = None
        self.line = None

    def get_start_end_positions(self):
        """
        Allow user to select start and end positions on the maze.

        This method sets up the plot for user interaction and
        connects the onclick event to the plot.
        """
        self.ax.set_title("Click to select start (green) and end (red) positions")
        self.start = None
        self.end = None
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show(block=False)

    def onclick(self, event):
        """
        Handle mouse click events on the maze plot.

        This method is called when the user clicks on the plot to
        select start and end positions.

        Args:
            event (matplotlib.backend_bases.MouseEvent): The mouse click event.
        """
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < self.maze.shape[1] and 0 <= y < self.maze.shape[0] and self.maze[y, x] == 0:
                if self.start is None:
                    self.start = (y, x)
                    self.ax.plot(x, y, 'go', markersize=10)
                elif self.end is None:
                    self.end = (y, x)
                    self.ax.plot(x, y, 'ro', markersize=10)
                    # Disconnect onclick event after both points are selected
                    self.fig.canvas.mpl_disconnect(self.fig.canvas.mpl_connect('button_press_event', self.onclick))
                self.fig.canvas.draw()

    def animate_solution(self, path, algorithm):
        """
        Animate the solution path on the maze plot.

        Args:
            path (list): List of coordinates representing the solution path.
            algorithm (str): Name of the algorithm used to solve the maze.
        """
        self.path = path
        self.ax.set_title(f"Solution using {algorithm.upper()} algorithm")
        if self.line:
            self.line.remove()
        self.line, = self.ax.plot([], [], color=CONFIG['visualization']['colors']['solution'][algorithm], linewidth=2)

        def init():
            """Initialize the animation."""
            self.line.set_data([], [])
            return self.line,

        def update(frame):
            """Update the animation for each frame."""
            x = [p[1] for p in self.path[:frame+1]]
            y = [p[0] for p in self.path[:frame+1]]
            self.line.set_data(x, y)
            return self.line,

        self.ani = animation.FuncAnimation(self.fig, update, frames=len(self.path),
                                           init_func=init, blit=True, 
                                           interval=CONFIG['visualization']['animation']['interval'],
                                           repeat=False)
        self.fig.canvas.draw()

    def clear(self):
        """
        Clear the current plot and reset it for a new maze solving attempt.
        """
        self.ax.clear()
        self.ax.imshow(self.maze, cmap='binary')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.start = None
        self.end = None
        self.path = None
        self.line = None
        self.fig.canvas.draw()


def main():
    """
    Main function to handle command-line arguments and run the maze solver.

    This function parses command-line arguments, generates a maze,
    and enters a loop allowing the user to repeatedly solve the maze
    with different start and end points.
    """
    parser = argparse.ArgumentParser(description=CONFIG['cli']['description'],
                                     epilog=CONFIG['cli']['epilog'])
    parser.add_argument('--algorithm', '-a', 
                        default=CONFIG['algorithms']['default'],
                        choices=CONFIG['algorithms']['available'],
                        help=f"Path-finding algorithm to use (default: {CONFIG['algorithms']['default']})")
    parser.add_argument('--width', '-w', 
                        type=int, 
                        default=CONFIG['maze']['default_width'],
                        help=f"Width of the maze (default: {CONFIG['maze']['default_width']})")
    parser.add_argument('--height', '-ht', 
                        type=int, 
                        default=CONFIG['maze']['default_height'],
                        help=f"Height of the maze (default: {CONFIG['maze']['default_height']})")
    parser.add_argument('--randomness', '-r', 
                        type=float, 
                        default=CONFIG['maze']['default_randomness'],
                        help=f"Randomness factor for maze generation (default: {CONFIG['maze']['default_randomness']})")
    args = parser.parse_args()

    # Generate maze
    print(f"Generating a {args.width}x{args.height} maze with randomness {args.randomness}...")
    maze = MazeSolver.generate_maze(args.width, args.height, args.randomness)

    viz = MazeVisualization(maze)

    while True:
        # Get start and end positions
        print("Please select start (green) and end (red) positions on the maze...")
        viz.get_start_end_positions()
        
        # Wait for user to select both start and end points
        while viz.start is None or viz.end is None:
            plt.pause(0.1)
        
        print(f"Start position: {viz.start}")
        print(f"End position: {viz.end}")

        # Solve maze
        print(f"Solving maze using {args.algorithm.upper()} algorithm...")
        path = MazeSolver.solve_maze(maze, viz.start, viz.end, args.algorithm)

        if not path:
            print("No solution found.")
        else:
            print(f"Solution found with path length: {len(path)}")
            print("Animating solution...")
            viz.animate_solution(path, args.algorithm)

        print("You can now select new start/end positions.")
        user_input = input("Do you want to select new start/end points? (y/n): ").strip().lower()
        if user_input != 'y':
            break
        viz.clear()

    plt.close(viz.fig)
    print("Thank you for using the Maze Solver!")


if __name__ == "__main__":
    CONFIG = load_config()
    main()
