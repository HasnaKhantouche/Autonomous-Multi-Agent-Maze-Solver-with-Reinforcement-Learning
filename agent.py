import random

class Agent:
    def __init__(self, position, goal, agent_id, epsilon_decay=0.995, min_epsilon=0.01):
        self.position = position
        self.previous_position = None
        self.q_table = {}  # State-action value table
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Epsilon decay factor
        self.min_epsilon = min_epsilon  # Minimum epsilon value
        self.goal = goal
        self.agent_id = agent_id  # Unique identifier for communication
        self.communication = []  # Communication log for coordination

    def manhattan_distance(self, pos):
        """Calculate Manhattan distance to the goal."""
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

    def perceive_and_act(self, maze, shared_knowledge, other_agents_positions):
        x, y = self.position
        moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]  # Down, Up, Left, Right
        state = self.position

        # Initialize Q-values for the current state if not already initialized
        if state not in self.q_table:
            self.q_table[state] = {move: 0 for move in moves}

        # Avoid collisions with walls and other agents
        valid_moves = []
        for move in moves:
            nx, ny = x + move[0], y + move[1]
            if (
                0 <= nx < len(maze)
                and 0 <= ny < len(maze[0])
                and maze[nx][ny] == 0  # Not a wall
                and (nx, ny) not in shared_knowledge["visited"]
                and (nx, ny) not in other_agents_positions
            ):
                valid_moves.append(move)

        # If no valid moves, try to stay or move randomly
        if not valid_moves:
            random_moves = [
                move
                for move in moves
                if 0 <= x + move[0] < len(maze)
                and 0 <= y + move[1] < len(maze[0])
                and maze[x + move[0]][y + move[1]] == 0
            ]
            if random_moves:
                move = random.choice(random_moves)
            else:
                return  # No valid moves, stay in place

        else:
            # Epsilon-greedy strategy with decay for exploration and exploitation
            if random.random() < self.epsilon:
                move = random.choice(valid_moves)  # Explore (choose random move)
            else:
                # Evaluate the Manhattan distance for each valid move and prioritize those closer to the goal
                move_distances = {m: self.manhattan_distance((x + m[0], y + m[1])) for m in valid_moves}
                closest_moves = [m for m in valid_moves if move_distances[m] == min(move_distances.values())]

                # If there are multiple closest moves, use Q-values to break ties
                if len(closest_moves) > 1:
                    move_q_values = {m: self.q_table[state][m] for m in closest_moves}
                    move = max(move_q_values, key=move_q_values.get)
                else:
                    move = closest_moves[0]

        # Calculate the next position
        nx, ny = x + move[0], y + move[1]
        new_state = (nx, ny)

        # Enhanced Reward Calculation:
        old_distance = self.manhattan_distance(state)
        new_distance = self.manhattan_distance(new_state)

        if new_state == self.goal:
            reward = 20  # High reward for reaching the goal (make this large enough to prioritize goal-reaching)
        elif new_distance < old_distance:
            reward = 2  # Reward for getting closer to the goal
        elif new_distance == old_distance:
            reward = -0.5  # Slight penalty for staying in place
        else:
            reward = -2  # Strong penalty for moving away from the goal

        # Penalize backtracking (oscillation)
        if new_state == self.previous_position:
            reward -= 1  # Avoid oscillation

        # Avoid obstacles
        if new_state in shared_knowledge["obstacles"]:
            reward -= 2  # Strong penalty for bumping into obstacles

        # Update Q-values based on the reward
        if new_state not in self.q_table:
            self.q_table[new_state] = {move: 0 for move in moves}
        max_future_q = max(self.q_table[new_state].values())
        current_q = self.q_table[state][move]
        self.q_table[state][move] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

        # Update the agent's state (position) and previous position
        self.previous_position = self.position
        self.position = new_state
        shared_knowledge["visited"].add(new_state)  # Share the visited position with other agents

        # Update epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

        # Agent communication (simple example of sharing knowledge)
        self.communication.append(f"Agent {self.agent_id} at {self.position}")
        shared_knowledge["communication"].append(self.communication)
