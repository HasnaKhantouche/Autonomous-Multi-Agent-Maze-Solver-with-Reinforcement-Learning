import pygame
from maze import create_maze
from agent import Agent

# Generate a larger maze
maze = create_maze(15, 15)

pygame.init()
cell_size = 40
screen = pygame.display.set_mode((cell_size * len(maze[0]), cell_size * len(maze)))
pygame.display.set_caption("Multi-Agent Maze Solver")
clock = pygame.time.Clock()

# Create multiple agents
start_positions = [(1, 1), (1, len(maze[0]) - 2), (len(maze) - 2, 1)]
goal_position = (len(maze) - 2, len(maze[0]) - 2)
agents = [Agent(start_pos, goal_position, agent_id=i) for i, start_pos in enumerate(start_positions)]

# Shared knowledge among agents
shared_knowledge = {"visited": set(), "obstacles": set(), "communication": []}

running = True
finished_agents = set()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))  # White background

    # Draw the maze
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            color = (0, 0, 0) if maze[i][j] == 1 else (255, 255, 255)
            pygame.draw.rect(screen, color, pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size))

    # Highlight start positions and goal
    for start_position in start_positions:
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(start_position[1] * cell_size, start_position[0] * cell_size, cell_size, cell_size))  # Start
    pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(goal_position[1] * cell_size, goal_position[0] * cell_size, cell_size, cell_size))  # Goal

    # Move agents and check their status
    other_agents_positions = {agent.position for agent in agents if agent.position != goal_position}

    for i, agent in enumerate(agents):
        if agent.position == goal_position:
            finished_agents.add(i)
        else:
            agent.perceive_and_act(maze, shared_knowledge, other_agents_positions)

        # Draw the agent's current position
        pygame.draw.circle(screen, (255, 0, 0), (agent.position[1] * cell_size + cell_size // 2, agent.position[0] * cell_size + cell_size // 2), cell_size // 4)

    # Check if all agents reached the goal
    if len(finished_agents) == len(agents):
        print("All agents reached the goal!")
        running = False

    pygame.display.flip()
    clock.tick(10)

pygame.quit()