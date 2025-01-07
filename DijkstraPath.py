import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# Define the directions for moving in the grid (up, down, left, right)
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# Global variables
drawing = False
erasing = False
setting_target = False
start = (0, 0)
target = None
path = []
last_pos = None

def create_grid(rows, cols):
    return np.zeros((rows, cols))

def dijkstra(grid, start, target):
    rows, cols = grid.shape
    distances = { (i, j): float('inf') for i in range(rows) for j in range(cols) }
    distances[start] = 0
    priority_queue = [(0, start)]
    heapq.heapify(priority_queue)
    prev = {start: None}
    visited = set()

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_node == target:
            break
        if current_node in visited:
            continue
        visited.add(current_node)
        for direction in DIRECTIONS:
            neighbor = (current_node[0] + direction[0], current_node[1] + direction[1])
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] == 0:
                distance = current_distance + 1
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))
                    prev[neighbor] = current_node

    path = []
    node = target
    while node is not None and node in prev:
        path.append(node)
        node = prev[node]
    path.reverse()
    
    # If the path does not end at the target, return an empty list
    if not path or path[-1] != target:
        return []
    return path

def visualize(grid, path, start, target):
    ax.clear()
    ax.imshow(grid, cmap='gray', origin='upper')
    if path:
        for (x, y) in path:
            ax.scatter(y, x, color='red', s=10)
    ax.scatter(start[1], start[0], color='green')
    if target:
        ax.scatter(target[1], target[0], color='blue')
    ax.axis('off')
    plt.draw()

def on_mouse_press(event):
    global drawing, erasing, setting_target, last_pos
    if event.button == 1:  # Left button
        drawing = True
        erasing = False
        setting_target = False
        last_pos = (int(round(event.ydata)), int(round(event.xdata)))
        update_grid(event)
    elif event.button == 3:  # Right button
        setting_target = True
        drawing = False
        erasing = False
        update_grid(event)

def on_mouse_release(event):
    global drawing, erasing, setting_target, path, last_pos
    drawing = False
    erasing = False
    setting_target = False
    last_pos = None
    if target and grid[target] != 1:  # Check if target is not an obstacle
        path = dijkstra(grid, start, target)
        visualize(grid, path, start, target)

def on_mouse_move(event):
    if drawing or setting_target:
        update_grid(event)

def interpolate_line(start, end):
    x0, y0 = start
    x1, y1 = end
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

def update_grid(event):
    global target, last_pos
    if event.xdata is None or event.ydata is None:
        return
    x, y = int(round(event.ydata)), int(round(event.xdata))
    if 0 <= x < rows and 0 <= y < cols:
        if drawing:
            if last_pos:
                for i, j in interpolate_line(last_pos, (x, y)):
                    if 0 <= i < rows and 0 <= j < cols:
                        grid[i, j] = 1
            else:
                grid[x, y] = 1
        elif setting_target:
            target = (x, y)
        last_pos = (x, y)
        visualize(grid, path, start, target)

def clear_grid(event):
    global grid, path, target
    grid = create_grid(rows, cols)
    path = []
    target = None
    visualize(grid, path, start, target)

# Parameters
rows, cols = 20, 20

# Create grid
grid = create_grid(rows, cols)

# Create the plot
fig, ax = plt.subplots()
fig.canvas.manager.set_window_title('Dijkstra-Pathfinding Visualization')

# Add instructions as text on the plot
plt.figtext(0.5, 0.95, 'Left-click + drag for walls, right-click to set target', 
            ha='center', fontsize=10, color='blue')

# Add the CLEAR button
button_ax = plt.axes([0.4, 0.01, 0.2, 0.05])  # [left, bottom, width, height]
clear_button = Button(button_ax, 'CLEAR', color='lightgray', hovercolor='red')
clear_button.on_clicked(clear_grid)

fig.canvas.mpl_connect('button_press_event', on_mouse_press)
fig.canvas.mpl_connect('button_release_event', on_mouse_release)
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
visualize(grid, path, start, target)
plt.show()
