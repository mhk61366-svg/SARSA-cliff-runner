import numpy as np
import pygame
from cliff_env import CliffRunner
from sarsa_lab import epsilon_greedy

TAG = "baseline"            
CELL = 48                   # pixels per grid cell
HEADER, FOOTER = 36, 30     # pixels reserved for the title bar and the help line
DIRS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}   # action -> (dx, dy) on screen

class Player:
    # One agent playing its own copy of the game, using a saved Q-table.
    def __init__(self, name, Q, seed):
        self.name, self.Q = name, Q
        self.env = CliffRunner()
        self.state = self.env.reset()
        self.rng = np.random.default_rng(seed)
        self.episodes, self.falls = 0, 0

    def tick(self, epsilon):
        a = epsilon_greedy(self.Q, self.state, epsilon, self.rng)
        self.state, r, terminated, truncated, info = self.env.step(a)
        self.falls += info["fell"]
        if terminated or truncated:
            self.episodes += 1
            self.state = self.env.reset()


def draw_arrow(screen, cx, cy, action, color=(235, 235, 235)):
    dx, dy = DIRS[action]
    tip = (cx + dx * 13, cy + dy * 13)
    tail = (cx - dx * 13, cy - dy * 13)
    px, py = -dy, dx                                        # perpendicular direction
    head = [tip,
            (tip[0] - dx * 9 + px * 6, tip[1] - dy * 9 + py * 6),
            (tip[0] - dx * 9 - px * 6, tip[1] - dy * 9 - py * 6)]
    pygame.draw.line(screen, color, tail, tip, 3)
    pygame.draw.polygon(screen, color, head)


def draw_panel(screen, player, top, epsilon, font):
    env, Q = player.env, player.Q
    values = Q.max(axis=2)                                  # V(s) = best Q-value of each cell
    seen = [values[r, c] for r in range(env.n_rows) for c in range(env.n_cols)
            if (r, c) not in env.lava and values[r, c] != 0]
    vmin, vmax = min(seen), max(seen)

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            rect = pygame.Rect(c * CELL, top + HEADER + r * CELL, CELL, CELL)
            if (r, c) in env.lava:
                color = (200, 60, 30)
            elif (r, c) == env.goal:
                color = (60, 180, 90)
            elif values[r, c] == 0:
                color = (60, 60, 70)                        # never updated
            else:
                t = (values[r, c] - vmin) / (vmax - vmin + 1e-9)   # 0 = bad, 1 = good
                color = (int(110 - 70 * t), int(50 + 70 * t), int(55 - 5 * t))
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (20, 20, 28), rect, 1)
            if (r, c) not in env.lava and (r, c) != env.goal:
                draw_arrow(screen, rect.centerx, rect.centery, int(np.argmax(Q[r, c])))
            if (r, c) == env.start:
                screen.blit(font.render("S", True, (255, 255, 255)), (rect.x + 4, rect.y + 2))
            if (r, c) == env.goal:
                screen.blit(font.render("G", True, (255, 255, 255)), (rect.x + 4, rect.y + 2))

    row, col = player.state                                 # the agent itself
    pygame.draw.circle(screen, (255, 220, 80),
                       (col * CELL + CELL // 2, top + HEADER + row * CELL + CELL // 2), CELL // 3)

    mode = f"exploring, eps={epsilon}" if epsilon > 0 else "greedy, eps=0"
    title = f"{player.name}  |  {mode}  |  games: {player.episodes}  |  lava falls: {player.falls}"
    screen.blit(font.render(title, True, (255, 255, 255)), (8, top + 9))

def main():
    pygame.init()
    probe = CliffRunner()
    panel_h = HEADER + probe.n_rows * CELL
    screen = pygame.display.set_mode((probe.n_cols * CELL, 2 * panel_h + FOOTER))
    pygame.display.set_caption("SARSA vs Q-learning on Cliff Runner")
    font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()

    players = [Player("SARSA", np.load(f"results/q_sarsa_{TAG}.npy"), seed=1),
               Player("Q-learning", np.load(f"results/q_qlearning_{TAG}.npy"), seed=2)]
    epsilon, speed = 0.1, 6

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    epsilon = 0.0 if epsilon > 0 else 0.1
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    speed = min(60, speed + 2)
                elif event.key == pygame.K_MINUS:
                    speed = max(2, speed - 2)

        for p in players:
            p.tick(epsilon)

        screen.fill((15, 15, 25))
        for i, p in enumerate(players):
            draw_panel(screen, p, i * panel_h, epsilon, font)
        help_text = "SPACE: explore / greedy     + / - : speed     ESC: quit"
        screen.blit(font.render(help_text, True, (170, 170, 190)), (8, 2 * panel_h + 7))
        pygame.display.flip()
        clock.tick(speed)
    pygame.quit()

if __name__ == "__main__":
    main()