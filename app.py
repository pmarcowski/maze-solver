# -*- coding: utf-8 -*-

__author__ = "Przemyslaw Marcowski"
__email__ = "p.marcowski@gmail.com"
__license__ = "MIT"

"""
Interactive Maze Solver Dash Application.

This Dash web application allows users to generate random mazes, select start and end
points, choose a path-finding algorithm, and visualize the solution path interactively.

Features:
- Algorithm Selection: Choose between BFS, DFS, A*, and Dijkstra's algorithms.
- Maze Generation: Adjust maze dimensions and generate new mazes.
- Interactive Selection: Click on the maze visualization to set start and end points.
- Solution Visualization: Display the solving path based on the selected algorithm.
- Cycle Probability Control: Adjust the number of cycles to create multiple paths.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import numpy as np
import random
from maze_solver import generate_perfect_maze, introduce_cycles, generate_imperfect_maze
from maze_solver import solve_maze
import logging

# Configure logging for debugging purposes
logging.basicConfig(level=logging.DEBUG)


def create_empty_figure():
    """
    Creates an empty Plotly figure with no data.

    Returns:
        dict: A dictionary representing an empty Plotly figure.
    """
    return {
        'data': [],
        'layout': go.Layout(
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[0, 1],
                fixedrange=True
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[0, 1],
                scaleanchor='x',
                scaleratio=1,
                fixedrange=True
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            hovermode='closest'  # Ensures hoverData is captured
        )
    }


def create_maze_figure(maze, points=None, path=None):
    """
    Creates a Plotly figure for the maze with optional start/end points and solution path.

    Args:
        maze (np.ndarray): 2D NumPy array representing the maze.
        points (dict, optional): Dictionary containing 'start' and 'end' points as (row, col).
        path (list, optional): List of positions representing the solution path.

    Returns:
        dict: A dictionary representing the Plotly figure.
    """
    data = [
        go.Heatmap(
            z=maze,  # Maze grid: 1 for walls, 0 for paths
            x=list(range(maze.shape[1])),  # X-axis indices
            y=list(range(maze.shape[0])),  # Y-axis indices
            colorscale=[[0, 'white'], [1, 'black']],  # White for paths, black for walls
            showscale=False,  # Hide color scale
            hoverinfo='none'  # Disable hover tooltips for heatmap
        )
    ]
    
    # Add start and end points if provided
    if points:
        if 'start' in points:
            data.append(
                go.Scatter(
                    x=[points['start'][1]],  # X-coordinate of start
                    y=[points['start'][0]],  # Y-coordinate of start
                    mode='markers',
                    marker=dict(size=12, color='green'),  # Green marker for start
                    name='Start',
                    hoverinfo='none'  # Disable hover tooltips for start point
                )
            )
        if 'end' in points:
            data.append(
                go.Scatter(
                    x=[points['end'][1]],  # X-coordinate of end
                    y=[points['end'][0]],  # Y-coordinate of end
                    mode='markers',
                    marker=dict(size=12, color='red'),  # Red marker for end
                    name='End',
                    hoverinfo='none'  # Disable hover tooltips for end point
                )
            )
    
    # Add solution path if provided
    if path:
        path_x = [p[1] for p in path]  # Extract X-coordinates from path
        path_y = [p[0] for p in path]  # Extract Y-coordinates from path
        data.append(
            go.Scatter(
                x=path_x,
                y=path_y,
                mode='lines',
                line=dict(color='green', width=3),  # Green line for path
                name='Path',
                hoverinfo='none'  # Disable hover tooltips for path
            )
        )
    
    # Define figure layout
    layout = go.Layout(
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True,
            scaleanchor='x',  # Ensure square cells
            scaleratio=1,
            autorange='reversed'  # Reverse y-axis to match maze indexing
        ),
        margin=dict(l=0, r=0, t=0, b=0),  # No margins
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
        hovermode='closest',  # Hover mode set to closest point
        showlegend=False      # Disable legend
    )
    
    return {'data': data, 'layout': layout}


# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Define app layout
app.layout = html.Div([
    # App title
    html.H1(
        "Maze Solver Demonstration",
        style={
            'textAlign': 'center',
            'marginBottom': '40px',
            'fontSize': '36px',
            'fontWeight': 'bold',
            'fontFamily': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'
        }
    ),
    
    # Hidden divs for storing intermediate data
    dcc.Store(id='store-maze'),   # Stores maze data
    dcc.Store(id='store-points'), # Stores start and end points
    dcc.Store(id='store-path'),   # Stores solution path
    
    # Main content container
    html.Div([
        # Maze visualization graph
        dcc.Graph(
            id="maze-graph",
            style={
                'height': '50vh',   # Height of graph
                'marginBottom': '20px'   
            },
            figure=create_empty_figure(),  # Initialize with empty figure
            config={
                'staticPlot': False,  # Allow interactivity
                'scrollZoom': False,  # Disable scroll zoom
                'displayModeBar': True,  # Show mode bar
                'modeBarButtonsToRemove': [
                    'zoom2d', 'pan2d', 'select2d', 'lasso2d',
                    'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d',
                    'hoverClosestCartesian', 'hoverCompareCartesian',
                    'toggleSpikelines'
                ]  # Remove specific buttons from mode bar
            }
        ),
        
        # Controls for user interaction
        html.Div([
            # Sliders arranged in same row
            html.Div([
                # Maze Dimension Slider
                html.Div([
                    html.Label(
                        "Maze Dimension:",
                        style={'textAlign': 'center', 'display': 'block', 'marginBottom': '10px', 'fontSize': '18px'}
                    ),
                    dcc.Slider(
                        id="maze-dim-slider",
                        min=5,
                        max=50,
                        step=1,
                        value=20,  # Default value
                        marks={i: str(i) for i in range(5, 51, 5)},  # Marks at every 5 units
                        tooltip={"placement": "bottom", "always_visible": True}  # Always show tooltip
                    )
                ], style={'flex': 1, 'marginRight': '20px'}),  # Flex properties for layout
                
                # Cycle Probability Slider
                html.Div([
                    html.Label(
                        "Cycle Probability:",
                        style={'textAlign': 'center', 'display': 'block', 'marginBottom': '10px', 'fontSize': '18px'}
                    ),
                    dcc.Slider(
                        id="cycle-prob-slider",
                        min=0.0,
                        max=0.3,
                        step=0.01,
                        value=0.1,  # Default cycle probability
                        marks={i/100: f"{i}%" for i in range(0, 31, 5)},  # Marks at every 5%
                        tooltip={"placement": "bottom", "always_visible": True}  # Always show tooltip
                    )
                ], style={'flex': 1})  # Flex properties for layout
            ], style={'display': 'flex', 'flexDirection': 'row', 'marginBottom': '20px'}),  # Parent Div with flex layout

            # Algorithm Selection Dropdown
            html.Div([
                html.Label(
                    "Select Algorithm:",
                    style={'textAlign': 'center', 'display': 'block', 'marginBottom': '10px', 'fontSize': '18px'}
                ),
                dcc.Dropdown(
                    id="algorithm-dropdown",
                    options=[
                        {'label': 'Breadth-First Search', 'value': 'bfs'},
                        {'label': 'Depth-First Search', 'value': 'dfs'},
                        {'label': 'A* Search', 'value': 'astar'},
                        {'label': "Dijkstra's Algorithm", 'value': 'dijkstra'}
                    ],
                    value='bfs',  # Default selected algorithm
                    clearable=False,  # Prevent clearing selection
                    style={'width': '50%', 'margin': '0 auto'}  # Center dropdown
                )
            ], style={'marginBottom': '20px'}),  # Bottom margin for spacing
            
            # Action Buttons: Generate and Solve Maze
            html.Div([
                html.Button(
                    "Generate New Maze",
                    id="generate-button",
                    className="btn btn-primary me-2",
                    style={'marginRight': '10px', 'padding': '10px 20px', 'fontSize': '16px'}
                ),
                html.Button(
                    "Solve Maze",
                    id="solve-button",
                    className="btn btn-success me-2",
                    style={'marginRight': '10px', 'padding': '10px 20px', 'fontSize': '16px'}
                )
            ], style={'textAlign': 'center', 'marginBottom': '20px'})  # Center buttons and add spacing
        ], style={'padding': '0 20px'}),  # Padding for controls
        
        # Message Window to Display Information to Users
        html.Div([
            html.H4(
                "Messages:",
                style={'textAlign': 'center', 'marginBottom': '10px', 'fontSize': '20px'}
            ),
            html.Div(
                id="solution-info",
                className="p-3 bg-light border",  # Bootstrap classes for styling
                style={'textAlign': 'center', 'fontSize': '18px'}
            )
        ], style={'padding': '0 20px'})  # Padding for message window
    ], style={'maxWidth': '800px', 'margin': '0 auto', 'fontFamily': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'})  # Center content
], style={'padding': '20px'})  # Outer padding for entire app

# Callback to handle interactions and update stores and messages
@app.callback(
    Output('store-maze', 'data'),     # Store generated maze data
    Output('store-points', 'data'),   # Store start and end points
    Output('store-path', 'data'),     # Store solution path
    Output('solution-info', 'children'),  # Update message window
    Input('generate-button', 'n_clicks'),  # Triggered by Generate button
    Input('maze-graph', 'clickData'),      # Triggered by clicking on maze graph
    Input('solve-button', 'n_clicks'),     # Triggered by Solve button
    State('maze-dim-slider', 'value'),     # Current maze dimension from slider
    State('algorithm-dropdown', 'value'),  # Selected algorithm from dropdown
    State('cycle-prob-slider', 'value'),   # Current cycle probability from slider
    State('store-maze', 'data'),           # Current maze data
    State('store-points', 'data'),         # Current start and end points
    prevent_initial_call=True              # Prevent callback from running on initial load
)
def handle_interactions(n_generate, clickData, n_solve, dim, algorithm, cycle_prob, maze, points):
    """
    Handles all user interactions: generating maze, selecting start/end points, and solving the maze.
    Updates the stored data and displays relevant messages to the user.

    Args:
        n_generate (int): Number of times the Generate button has been clicked.
        clickData (dict): Data from clicking on the maze graph.
        n_solve (int): Number of times the Solve button has been clicked.
        dim (int): Maze dimension from the slider.
        algorithm (str): Selected path-finding algorithm.
        cycle_prob (float): Cycle probability from the slider.
        maze (list): Current maze data.
        points (dict): Current start and end points.

    Returns:
        tuple: Updated maze data, points, path, and message string.
    """
    triggered = callback_context.triggered  # Determine which input triggered callback
    if not triggered:
        logging.debug("No triggers detected.")
        return dash.no_update, dash.no_update, dash.no_update, "No action triggered."
    
    trigger_id = triggered[0]['prop_id'].split('.')[0]  # Get ID of triggering component
    logging.debug(f"Triggered by: {trigger_id}")
    
    # Initialize message
    message = ""
    
    # Handle Generate New Maze Button Click
    if trigger_id == 'generate-button' and n_generate:
        # Generate new imperfect maze with specified dimension and cycle probability
        maze = generate_imperfect_maze(dim, cycle_probability=cycle_prob)
        points = {}  # Reset start and end points
        path = None  # Reset solution path
        message = "New maze generated with multiple paths. Click on a white cell to set the start point."
        logging.debug(f"Maze generated with dimension {dim} and cycle probability {cycle_prob}.")
        return maze.tolist(), points, path, message
    
    # Handle Clicks on Maze Graph to Set Start/End Points
    if trigger_id == 'maze-graph' and clickData:
        if maze is None:
            # No maze has been generated yet
            message = "No maze generated. Please generate a maze first."
            logging.debug("Click detected but no maze is generated.")
            return dash.no_update, dash.no_update, dash.no_update, message
        
        # Extract clicked coordinates from clickData
        point = clickData['points'][0]
        x, y = point['x'], point['y']
        logging.debug(f"Clicked coordinates before rounding: x={x}, y={y}")
        
        # Convert floating point coordinates to integer indices
        try:
            x = int(round(x))
            y = int(round(y))
        except (TypeError, ValueError):
            message = f"Invalid click coordinates: x={x}, y={y}"
            logging.debug(message)
            return dash.no_update, dash.no_update, dash.no_update, message
        
        logging.debug(f"Clicked coordinates after rounding: x={x}, y={y}")
        
        # Validate that click is within maze boundaries
        if y < 0 or y >= len(maze) or x < 0 or x >= len(maze[0]):
            message = f"Click out of bounds at x: {x}, y: {y}"
            logging.debug(message)
            return dash.no_update, dash.no_update, dash.no_update, message
        
        # Validate that clicked cell is path (not a wall)
        if maze[y][x] == 1:
            message = f"Clicked on a wall at x: {x}, y: {y}"
            logging.debug(message)
            return dash.no_update, dash.no_update, dash.no_update, message
        
        # Initialize points dictionary if it doesn't exist
        if points is None:
            points = {}
        
        # Set Start and End Points
        if 'start' not in points:
            # Set start point
            points['start'] = [y, x]
            message = f"Start point set at ({x}, {y}). Click on another white cell to set the end point."
            logging.debug(message)
        elif 'end' not in points:
            # Set end point
            points['end'] = [y, x]
            message = f"End point set at ({x}, {y}). Click 'Solve Maze' to find the path or click again to select new start/end points."
            logging.debug(message)
        else:
            # Both start and end points are already set; update start point and reset end point
            points['start'] = [y, x]
            points.pop('end', None)  # Remove existing end point
            path = None  # Reset solution path
            message = f"Start point updated to ({x}, {y}). Click on another white cell to set the end point."
            logging.debug(message)
            return dash.no_update, points, path, message
        
        # Reset path when points are updated
        return dash.no_update, points, None, message
    
    # Handle Solve Maze Button Click
    if trigger_id == 'solve-button' and n_solve:
        # Ensure maze, start, and end points are set
        if maze is None or points is None or 'start' not in points or 'end' not in points:
            message = "Please generate a maze and set both start and end points before solving."
            logging.debug(message)
            return dash.no_update, dash.no_update, dash.no_update, message
        
        # Extract start and end positions as tuples
        start = tuple(points['start'])
        end = tuple(points['end'])
        logging.debug(f"Solving maze from {start} to {end} using algorithm {algorithm}.")
        
        # Solve maze using selected algorithm
        path = solve_maze(np.array(maze), algorithm, start, end)
        
        if path is None:
            # No solution found
            message = "No solution found for the selected maze."
            logging.debug(message)
            return dash.no_update, dash.no_update, dash.no_update, message
        
        # Solution found; display path length and prompt for next action
        path_length = len(path)
        message = f"Solution found using {algorithm.upper()}. Path length: {path_length}. Click on the maze to select a new start point."
        logging.debug(message)
        return dash.no_update, dash.no_update, path, message
    
    # If no valid trigger detected, do not update anything
    logging.debug("No valid trigger detected.")
    return dash.no_update, dash.no_update, dash.no_update, "No valid action performed."


# Callback to update maze visualization based on stored data
@app.callback(
    Output('maze-graph', 'figure'),  # Output: Updated maze figure
    Input('store-maze', 'data'),     # Input: Maze data
    Input('store-points', 'data'),   # Input: Start and end points
    Input('store-path', 'data'),     # Input: Solution path
)
def update_figure(maze, points, path):
    """
    Updates the maze visualization based on stored maze data, points, and solution path.

    Args:
        maze (list): Maze data as a list of lists.
        points (dict): Dictionary containing 'start' and 'end' points.
        path (list): Solution path as a list of positions.

    Returns:
        dict: Updated Plotly figure for the maze visualization.
    """
    if maze is None:
        logging.debug("Maze is None, setting empty figure.")
        return create_empty_figure()
    
    # Convert maze data back to NumPy array for processing
    maze_np = np.array(maze)
    fig = create_maze_figure(maze_np, points, path)  # Create updated figure
    logging.debug("Maze figure updated.")
    return fig

# Run Dash app
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')
