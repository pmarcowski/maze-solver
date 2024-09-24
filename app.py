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
- Randomness Control: Adjust the randomness factor for maze generation.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import numpy as np
from maze_solver import MazeSolver
import logging

# Configure logging for debugging purposes
logging.basicConfig(level=logging.DEBUG)


def create_empty_figure():
    """Creates an empty Plotly figure with no data."""
    return {
        'data': [],
        'layout': go.Layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1], fixedrange=True),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1], scaleanchor='x', scaleratio=1, fixedrange=True),
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='closest'
        )
    }


def create_maze_figure(maze, points=None, path=None):
    """Creates a Plotly figure for the maze with optional start/end points and solution path."""
    data = [
        go.Heatmap(
            z=maze,
            x=list(range(maze.shape[1])),
            y=list(range(maze.shape[0])),
            colorscale=[[0, 'white'], [1, 'black']],
            showscale=False,
            hoverinfo='none'
        )
    ]
    
    if points:
        if 'start' in points:
            data.append(go.Scatter(x=[points['start'][1]], y=[points['start'][0]], mode='markers',
                                   marker=dict(size=12, color='green'), name='Start', hoverinfo='none'))
        if 'end' in points:
            data.append(go.Scatter(x=[points['end'][1]], y=[points['end'][0]], mode='markers',
                                   marker=dict(size=12, color='red'), name='End', hoverinfo='none'))
    
    if path:
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        data.append(go.Scatter(x=path_x, y=path_y, mode='lines', line=dict(color='green', width=3),
                               name='Path', hoverinfo='none'))
    
    layout = go.Layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, scaleanchor='x', scaleratio=1, autorange='reversed'),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='closest',
        showlegend=False
    )
    
    return {'data': data, 'layout': layout}


# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Define app layout
app.layout = html.Div([
    # App title
    html.H1(
        "Maze Solver",
        style={
            'textAlign': 'center',
            'marginBottom': '40px',
            'fontSize': '36px',
            'fontWeight': 'bold',
            'fontFamily': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'
        }
    ),
    
    # Info modal
    html.Div([
        html.Div([
            html.H2("Welcome to the Maze Solver!",
                    style={'textAlign': 'center', 'marginBottom': '20px'}),
            html.P(
                "I'm Przemek, a San Diego-based researcher and data scientist with a passion for using data to make things more interesting. "
                "This app showcases various pathfinding algorithms applied to randomly generated mazes.",
                style={'marginBottom': '15px'}
            ),
            html.P(
                "You can generate mazes of different sizes, adjust the randomness factor, "
                "and compare the performance of different pathfinding algorithms.",
                style={'marginBottom': '15px'}
            ),
            html.P([
                "You can explore my other work ",
                html.A("here", href="https://przemyslawmarcowski.com", target="_blank"),
                "."
            ], style={'marginBottom': '20px'}),
            html.Button('Close', id='close-modal', n_clicks=0,
                        style={'padding': '10px 20px', 'fontSize': '16px'})
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '5px',
            'maxWidth': '500px',
            'margin': '100px auto',
            'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)',
            'fontFamily': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'
        })
    ], id='modal', style={
        'display': 'none',
        'position': 'fixed',
        'zIndex': '1000',
        'left': '0',
        'top': '0',
        'width': '100%',
        'height': '100%',
        'overflow': 'auto',
        'backgroundColor': 'rgba(0,0,0,0.4)'
    }),
    
    dcc.Store(id='store-maze'),
    dcc.Store(id='store-points'),
    dcc.Store(id='store-path'),
    
    html.Div([
        dcc.Graph(
            id="maze-graph",
            style={'height': '50vh', 'marginBottom': '20px'},
            figure=create_empty_figure(),
            config={'staticPlot': False, 'scrollZoom': False, 'displayModeBar': True,
                    'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines']}
        ),
        
        html.Div([
            html.Div([
                html.Label("Maze Dimension:", style={'textAlign': 'center', 'display': 'block', 'marginBottom': '10px', 'fontSize': '18px'}),
                dcc.Slider(id="maze-dim-slider", min=5, max=50, step=1, value=20,
                           marks={i: str(i) for i in range(5, 51, 5)},
                           tooltip={"placement": "bottom", "always_visible": True})
            ], style={'flex': 1, 'marginRight': '20px'}),
            
            html.Div([
                html.Label("Randomness:", style={'textAlign': 'center', 'display': 'block', 'marginBottom': '10px', 'fontSize': '18px'}),
                dcc.Slider(id="randomness-slider", min=0.0, max=0.5, step=0.01, value=0.1,
                           marks={i/10: f"{i/10:.1f}" for i in range(0, 6, 1)},
                           tooltip={"placement": "bottom", "always_visible": True})
            ], style={'flex': 1})
        ], style={'display': 'flex', 'flexDirection': 'row', 'marginBottom': '20px'}),

        html.Div([
            html.Label("Select Algorithm:", style={'textAlign': 'center', 'display': 'block', 'marginBottom': '10px', 'fontSize': '18px'}),
            dcc.Dropdown(
                id="algorithm-dropdown",
                options=[
                    {'label': 'Breadth-First Search', 'value': 'bfs'},
                    {'label': 'Depth-First Search', 'value': 'dfs'},
                    {'label': 'A* Search', 'value': 'astar'},
                    {'label': "Dijkstra's Algorithm", 'value': 'dijkstra'}
                ],
                value='bfs',
                clearable=False,
                style={'width': '50%', 'margin': '0 auto'}
            )
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Button("Generate New Maze", id="generate-button", 
                        style={'marginRight': '10px', 'padding': '10px 20px', 'fontSize': '16px'}),
            html.Button("Solve Maze", id="solve-button", 
                        style={'marginRight': '10px', 'padding': '10px 20px', 'fontSize': '16px'}),
            html.Button("About This App", id='open-modal', n_clicks=0,
                        style={'padding': '10px 20px', 'fontSize': '16px'})
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        html.Div([
            html.H4("Messages:", style={'textAlign': 'center', 'marginBottom': '10px', 'fontSize': '20px'}),
            html.Div(id="solution-info", 
                     style={'textAlign': 'center', 'fontSize': '18px'},
                     children="Welcome to the Maze Solver! Generate a new maze and select start and end points to find a solution.")
        ], style={'padding': '0 20px'})
    ], style={'maxWidth': '800px', 'margin': '0 auto', 'fontFamily': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'})
], style={'padding': '20px'})


@app.callback(
    Output('store-maze', 'data'),
    Output('store-points', 'data'),
    Output('store-path', 'data'),
    Output('solution-info', 'children'),
    Input('generate-button', 'n_clicks'),
    Input('maze-graph', 'clickData'),
    Input('solve-button', 'n_clicks'),
    State('maze-dim-slider', 'value'),
    State('algorithm-dropdown', 'value'),
    State('randomness-slider', 'value'),
    State('store-maze', 'data'),
    State('store-points', 'data'),
    prevent_initial_call=True
)
def handle_interactions(n_generate, clickData, n_solve, dim, algorithm, randomness, maze, points):
    triggered = callback_context.triggered
    if not triggered:
        return dash.no_update, dash.no_update, dash.no_update, "No action triggered."
    
    trigger_id = triggered[0]['prop_id'].split('.')[0]
    logging.debug(f"Triggered by: {trigger_id}")
    
    if trigger_id == 'generate-button' and n_generate:
        maze = MazeSolver.generate_maze(dim, dim, randomness)
        points = {}
        path = None
        message = "New maze generated. Click on a white cell to set the start point."
        return maze.tolist(), points, path, message
    
    if trigger_id == 'maze-graph' and clickData:
        if maze is None:
            return dash.no_update, dash.no_update, dash.no_update, "No maze generated. Please generate a maze first."
        
        point = clickData['points'][0]
        x, y = int(round(point['x'])), int(round(point['y']))
        
        if y < 0 or y >= len(maze) or x < 0 or x >= len(maze[0]) or maze[y][x] == 1:
            return dash.no_update, dash.no_update, dash.no_update, "Invalid selection. Please click on a white cell."
        
        if points is None:
            points = {}
        
        if 'start' not in points:
            points['start'] = [y, x]
            message = f"Start point set at ({x}, {y}). Click on another white cell to set the end point."
        elif 'end' not in points:
            points['end'] = [y, x]
            message = f"End point set at ({x}, {y}). Click 'Solve Maze' to find the path or click again to select new start/end points."
        else:
            points['start'] = [y, x]
            points.pop('end', None)
            message = f"Start point updated to ({x}, {y}). Click on another white cell to set the end point."
        
        return dash.no_update, points, None, message
    
    if trigger_id == 'solve-button' and n_solve:
        if maze is None or points is None or 'start' not in points or 'end' not in points:
            return dash.no_update, dash.no_update, dash.no_update, "Please generate a maze and set both start and end points before solving."
        
        start = tuple(points['start'])
        end = tuple(points['end'])
        path = MazeSolver.solve_maze(np.array(maze), start, end, algorithm)
        
        if path is None:
            return dash.no_update, dash.no_update, dash.no_update, "No solution found for the selected maze."
        
        path_length = len(path)
        message = f"Solution found using {algorithm.upper()}. Path length: {path_length}. Click on the maze to select a new start point."
        return dash.no_update, dash.no_update, path, message
    
    return dash.no_update, dash.no_update, dash.no_update, "No valid action performed."


@app.callback(
    Output('maze-graph', 'figure'),
    Input('store-maze', 'data'),
    Input('store-points', 'data'),
    Input('store-path', 'data'),
)
def update_figure(maze, points, path):
    if maze is None:
        return create_empty_figure()
    
    maze_np = np.array(maze)
    fig = create_maze_figure(maze_np, points, path)
    return fig


@app.callback(
    Output('modal', 'style'),
    [Input('open-modal', 'n_clicks'),
     Input('close-modal', 'n_clicks')],
    [State('modal', 'style')]
)
def toggle_modal(n_open, n_close, modal_style):
    ctx = dash.callback_context
    if not ctx.triggered:
        return modal_style
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'open-modal' and n_open:
        modal_style['display'] = 'block'
    elif button_id == 'close-modal' and n_close:
        modal_style['display'] = 'none'
    
    return modal_style


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')
