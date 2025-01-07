import random

def create_maze(width, height):
    # Initialization of the maze with walls
    maze = [[1] * width for _ in range(height)]
    stack = [(1, 1)]
    maze[1][1] = 0  # Starting point

    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nx, ny = x + dx, y + dy
            if 1 <= nx < width - 1 and 1 <= ny < height - 1 and maze[ny][nx] == 1:
                neighbors.append((nx, ny))

        if neighbors:
            nx, ny = random.choice(neighbors)
            maze[(y + ny) // 2][(x + nx) // 2] = 0  # Create a path
            maze[ny][nx] = 0
            stack.append((nx, ny))
        else:
            stack.pop()

    # Create an exit point
    maze[height-2][width-2] = 0  # End point
    return maze
