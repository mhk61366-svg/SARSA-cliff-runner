# ---------------------- Cliff Runner Game logic ----------------------

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3                  # all posssible actions 
ACTION_DELTAS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}
ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]       

class CliffRunner:
    N_ACTIONS = 4
    def __init__(self, n_rows=6, n_cols=12, lava_penalty=-100, step_penalty=-1, max_steps=200):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.lava_penalty = lava_penalty
        self.step_penalty = step_penalty
        self.max_steps = max_steps
        self.start_state = (n_rows - 1, 0)
        self.goal_state = (n_rows - 1, n_cols - 1)
        self.lava = {(n_rows - 1, c) for c in range(1, n_cols - 1)}
        self.reset()

    def reset(self):                    # Reset the environment to the initial state
        self.pos = self.start_state
        self.steps = 0
        return self.pos

    def step(self, action):
        d_row, d_col = ACTION_DELTAS[action]
        new_row = min(max(self.pos[0] + d_row, 0), self.n_rows - 1)    # Ensure the new position is within the grid boundaries
        new_col = min(max(self.pos[1] + d_col, 0), self.n_cols - 1)  
        new_pos = (new_row, new_col)
        self.pos = new_pos
        self.steps += 1

        reward = self.step_penalty           # penalty -1 for each step taken to encourage the agent to reach the goal quickly
        fell = False
        terminated = False

        if self.pos in self.lava:           # check if the agent fell into lava
            reward += self.lava_penalty     # Reward penalty for falling into lava
            fell = True
            self.pos = self.start_state
        elif self.pos == self.goal_state:
            terminated = True               # check if the agent reached the goal state

        truncated = (not terminated) and self.steps >= self.max_steps

        return self.pos, reward, terminated, truncated, {"fell": fell}   

    def render_text(self):                              # Console/ Terminal view of the Game environment for testing
        """Quick ASCII view for debugging."""
        for r in range(self.n_rows):
            line = ""
            for c in range(self.n_cols):
                if (r, c) == self.pos:
                    line += "A "
                elif (r, c) == self.start_state:
                    line += "S "
                elif (r, c) == self.goal_state:
                    line += "G "
                elif (r, c) in self.lava:
                    line += "L "
                else:
                    line += ". "
            print(line)

if __name__ == "__main__":
    import random

    env = CliffRunner()
    env.reset()
    env.render_text()