import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from cliff_env import CliffRunner, ACTION_NAMES

# ---------- SETTINGS: the only place to edit for experiments ----------
SETTINGS = dict(
    episodes=1000,        # training episodes per run
    n_runs=20,            # independent runs (different random seeds), averaged in the plots
    alpha=0.1,            # learning rate
    gamma=0.99,           # discount factor
    epsilon=0.1,          # exploration rate (start value)
    eps_decay=1.0,        # 1.0 = epsilon stays constant; e.g. 0.99 = shrinks 1% per episode
    eps_min=0.0,          # epsilon never goes below this
    lava_penalty=-100,    # reward for stepping into lava
    tag="baseline",       # label added to saved file names, so experiments don't overwrite each other
)
# -----------------------------------------------------------------------

def epsilon_greedy(Q, state, epsilon, rng):
    # With probability epsilon pick a random action, otherwise the best-known one.
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[-1]))
    q = Q[state]                                         # the 4 Q-values of this cell
    return int(rng.choice(np.flatnonzero(q == q.max())))  # random tie-break

def train(env, algo="sarsa", episodes=1000, alpha=0.1, gamma=0.99,
          epsilon=0.1, eps_decay=1.0, eps_min=0.0, seed=0):
    # Train one agent. algo is "sarsa" or "qlearning". Returns a dict of results.
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_rows, env.n_cols, env.N_ACTIONS))
    returns = np.zeros(episodes)              # total reward of each episode
    falls = np.zeros(episodes)                # lava falls in each episode
    visits = np.zeros((env.n_rows, env.n_cols))   # where the agent spent its last 100 episodes

    for ep in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, epsilon, rng)            # choose A0 BEFORE the loop
        done = False

        while not done:
            s2, r, terminated, truncated, info = env.step(a)
            a2 = epsilon_greedy(Q, s2, epsilon, rng)      # choose A' NOW, before updating

            if terminated:
                target = r                                # no future after the goal
            elif algo == "sarsa":
                target = r + gamma * Q[s2 + (a2,)]        # <-- SARSA: the action I will REALLY take
            else:
                target = r + gamma * Q[s2].max()          # <-- Q-learning: the BEST action there

            Q[s + (a,)] += alpha * (target - Q[s + (a,)])

            returns[ep] += r
            falls[ep] += info["fell"]
            if ep >= episodes - 100:
                visits[s] += 1

            s, a = s2, a2                                 # A' becomes the action we actually execute
            done = terminated or truncated

        epsilon = max(eps_min, epsilon * eps_decay)       # decay once per episode (no-op if eps_decay=1)

    return {"Q": Q, "returns": returns, "falls": falls, "visits": visits}

def greedy_rollout(env, Q, max_steps=100):
    # Play one game with epsilon = 0 and NO learning. Returns (path, total_return).
    s = env.reset()
    path, total = [s], 0
    for _ in range(max_steps):
        s, r, terminated, truncated, info = env.step(int(np.argmax(Q[s])))
        path.append(s)
        total += r
        if terminated or truncated:
            break
    return path, total

def print_policy(env, Q, path=None):
    # Print the greedy action of every cell as an arrow. Cells on `path` are shown in [brackets].
    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}
    on_path = set(path) if path else set()
    for r in range(env.n_rows):
        line = ""
        for c in range(env.n_cols):
            if (r, c) in env.lava:
                ch = "L"
            elif (r, c) == env.goal:
                ch = "G"
            elif not Q[r, c].any():
                ch = "?"                                  # never updated = no opinion yet
            else:
                ch = arrows[int(np.argmax(Q[r, c]))]
            line += f"[{ch}]" if (r, c) in on_path else f" {ch} "
        print(line)

def demo():
    # Lab 5: train ONE SARSA agent and inspect it.
    S = SETTINGS
    env = CliffRunner(lava_penalty=S["lava_penalty"])
    out = train(env, "sarsa", S["episodes"], S["alpha"], S["gamma"],
                S["epsilon"], S["eps_decay"], S["eps_min"], seed=0)
    for start in range(0, S["episodes"], 100):               # average each block of 100 episodes
        block = slice(start, start + 100)
        print(f"episodes {start+1:4d}-{start+100:4d}: avg return {out['returns'][block].mean():7.1f}"
              f" | avg lava falls {out['falls'][block].mean():.2f}")
    path, total = greedy_rollout(env, out["Q"])
    reached = path[-1] == env.goal
    print(f"\ngreedy path: {len(path)-1} steps, return {total}, reached goal: {reached}")
    print_policy(env, out["Q"], path)

def smooth(x, k=10):
    # Moving average, to make noisy curves readable.
    return np.convolve(x, np.ones(k) / k, mode="valid")

def compare():
    # SARSA vs Q-learning: many runs each, a results table, two figures, saved Q-tables.
    S = SETTINGS
    os.makedirs("results", exist_ok=True)
    algos = {"sarsa": "SARSA", "qlearning": "Q-learning"}
    goal = CliffRunner().goal
    res = {}
    # --- 1. train n_runs independent agents of each kind ---
    for algo in algos:
        res[algo] = []
        for seed in range(S["n_runs"]):
            env = CliffRunner(lava_penalty=S["lava_penalty"])
            res[algo].append(train(env, algo, S["episodes"], S["alpha"], S["gamma"],
                                   S["epsilon"], S["eps_decay"], S["eps_min"], seed))

    # --- 2. the numbers ---
    print(f"{'':12s}| ONLINE (last 100 ep, exploring) | GREEDY (eps=0, after training)")
    print(f"{'':12s}| avg return | lava falls/ep      | reached goal | path length | return")
    for algo, name in algos.items():
        runs = res[algo]
        online_ret = np.mean([r["returns"][-100:].mean() for r in runs])
        online_fall = np.mean([r["falls"][-100:].mean() for r in runs])
        rolls = [greedy_rollout(CliffRunner(lava_penalty=S["lava_penalty"]), r["Q"]) for r in runs]
        good = [(path, total) for path, total in rolls if path[-1] == goal]   # runs that got there
        reached = len(good) / len(rolls)
        g_len = np.mean([len(path) - 1 for path, _ in good]) if good else float("nan")
        g_ret = np.mean([total for _, total in good]) if good else float("nan")
        print(f"{name:12s}| {online_ret:10.1f} | {online_fall:18.2f} | {reached:12.0%} | {g_len:11.1f} | {g_ret:6.1f}")

    # --- 3. figure 1: learning curves ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for algo, name in algos.items():
        ret = np.mean([r["returns"] for r in res[algo]], axis=0)
        fal = np.mean([r["falls"] for r in res[algo]], axis=0)
        ax[0].plot(smooth(ret), label=name)
        ax[1].plot(smooth(fal), label=name)
    ax[0].set_title("Return per episode (while still exploring)")
    ax[0].set_xlabel("Episode"); ax[0].set_ylabel("Total reward"); ax[0].set_ylim(-150, 0)
    ax[1].set_title("Lava falls per episode")
    ax[1].set_xlabel("Episode"); ax[1].set_ylabel("Falls")
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/curves_{S['tag']}.png", dpi=150)

    # --- 4. figure 2: where did each agent actually spend its time? ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.2))
    for i, (algo, name) in enumerate(algos.items()):
        v = np.mean([r["visits"] for r in res[algo]], axis=0)
        im = ax[i].imshow(v, cmap="magma")
        ax[i].set_title(f"{name}: where it walked (last 100 episodes)")
        ax[i].set_xticks([]); ax[i].set_yticks([])
        plt.colorbar(im, ax=ax[i], fraction=0.025)
    plt.tight_layout()
    plt.savefig(f"results/visits_{S['tag']}.png", dpi=150)

    # --- 5. save the first run's Q-table of each algorithm, for the Pygame viewer ---
    for algo in algos:
        np.save(f"results/q_{algo}_{S['tag']}.npy", res[algo][0]["Q"])
    print(f"\nsaved plots + Q-tables in results/ (tag = {S['tag']})")
    plt.show()


# ===== ADD NEW CODE ABOVE THIS LINE =====
if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command == "demo":
        demo()
    elif command == "compare":
        compare()
    else:
        print("unknown command:", command, "(use: demo or compare)")