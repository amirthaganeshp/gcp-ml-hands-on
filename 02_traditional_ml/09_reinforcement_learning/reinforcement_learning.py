"""Reinforcement learning: learning from consequences instead of answers.

Supervised learning is given the right answer for every example. RL is given
only a REWARD, often long after the action that earned it, and has to work out
which decisions were responsible. That credit-assignment problem is the whole
subject.

  1. The environment - a 5x5 grid world, written here in ~40 lines so there
     is no black box and no dependency
  2. Q-learning - the table, the update rule, and watching it converge
  3. Exploration vs exploitation - the same agent at four epsilon settings,
     which is the single most important hyperparameter here
  4. The discount factor - what gamma actually changes about the policy
  5. Q-learning vs SARSA - off-policy and on-policy, and the cliff problem
     where they visibly disagree

Run:
    python 02_traditional_ml/09_reinforcement_learning/reinforcement_learning.py

Concept reference: Foundations volume, Part A.18 (reinforcement learning).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"

ACTIONS = ["up", "down", "left", "right"]
MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
ARROWS = {0: "^", 1: "v", 2: "<", 3: ">"}


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


# ----------------------------------------------------------- 1. the world
class GridWorld:
    """A 5x5 grid. Start bottom-left, goal top-right, two penalty squares.

    Deliberately written out rather than imported: an RL example where the
    environment is a black box teaches you nothing about what the agent is
    actually solving. Everything an agent can perceive is here.

    Rewards:  +10 reaching the goal, -10 stepping on a penalty square,
              -1 for every other step (so dawdling is costly).
    """

    def __init__(self, size: int = 5, penalty: float = -10.0) -> None:
        self.size = size
        self.start = (size - 1, 0)
        self.goal = (0, size - 1)
        self.penalties = {(1, 1), (2, 3)}
        self.penalty = penalty
        self.n_states = size * size
        self.n_actions = 4

    def state_id(self, pos: tuple[int, int]) -> int:
        return pos[0] * self.size + pos[1]

    def reset(self) -> int:
        self.pos = self.start
        return self.state_id(self.pos)

    def step(self, action: int) -> tuple[int, float, bool]:
        dr, dc = MOVES[action]
        r = min(max(self.pos[0] + dr, 0), self.size - 1)   # walls block, not wrap
        c = min(max(self.pos[1] + dc, 0), self.size - 1)
        self.pos = (r, c)
        if self.pos == self.goal:
            return self.state_id(self.pos), 10.0, True
        if self.pos in self.penalties:
            return self.state_id(self.pos), self.penalty, True
        return self.state_id(self.pos), -1.0, False

    def render_policy(self, Q: np.ndarray) -> str:
        rows = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                if (r, c) == self.goal:
                    row.append(" G ")
                elif (r, c) in self.penalties:
                    row.append(" X ")
                else:
                    row.append(f" {ARROWS[int(np.argmax(Q[self.state_id((r, c))]))]} ")
            rows.append("".join(row))
        return "\n".join("    " + r for r in rows)


def train(env: GridWorld, episodes: int = 500, alpha: float = 0.1,
          gamma: float = 0.95, epsilon: float = 0.1, sarsa: bool = False,
          seed: int = SEED) -> tuple[np.ndarray, list[float]]:
    """Tabular Q-learning, or SARSA when sarsa=True.

    The only difference between the two is one line - see the comment at the
    update. That single line is the whole off-policy / on-policy distinction.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    history = []

    def choose(s: int) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(env.n_actions))       # explore
        return int(np.argmax(Q[s]))                        # exploit

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        total, done, steps = 0.0, False, 0
        while not done and steps < 100:                    # cap runaway episodes
            s2, r, done = env.step(a)
            a2 = choose(s2)
            if sarsa:
                # ON-POLICY: bootstrap from the action actually taken next,
                # so the agent's own exploration is priced into the value.
                target = r + gamma * Q[s2, a2] * (not done)
            else:
                # OFF-POLICY: bootstrap from the BEST next action, whether or
                # not the agent will actually take it.
                target = r + gamma * Q[s2].max() * (not done)
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = s2, a2
            total += r
            steps += 1
        history.append(total)
    return Q, history


def smooth(x: list[float], w: int = 25) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode="valid")


def figure_learning(env: GridWorld) -> np.ndarray:
    Q, hist = train(env, episodes=500)
    print(f"  A {env.size}x{env.size} grid. Start {env.start}, goal {env.goal}, "
          f"penalties {sorted(env.penalties)}.")
    print("  Rewards: +10 goal, -10 penalty, -1 per step.")
    print(f"\n  The Q-table is {env.n_states} states x {env.n_actions} actions = "
          f"{Q.size} numbers.")
    print("  Every one of them starts at zero. Nothing is told to the agent -")
    print("  not the layout, not where the goal is, not that walls exist.")

    print(f"\n  Mean reward, first 50 episodes:  {np.mean(hist[:50]):7.2f}")
    print(f"  Mean reward, last 50 episodes:   {np.mean(hist[-50:]):7.2f}")

    print("\n  The learned policy (greedy action in each square):\n")
    print(env.render_policy(Q))
    print("\n    G = goal, X = penalty. Arrows point the way the agent would go.")
    print("  Note the squares next to a penalty: the arrows steer around them,")
    print("  which the agent learned only by stepping on them and being")
    print("  punished. There was no rule saying 'X is bad'.")

    print("\n  The update rule, which is the whole algorithm:")
    print("      Q[s,a] <- Q[s,a] + alpha * (r + gamma * max Q[s',:] - Q[s,a])")
    print("  The bracket is the TEMPORAL DIFFERENCE ERROR: the gap between what")
    print("  we thought this state-action was worth and what it looks like now")
    print("  that we have seen the reward and the next state. Value propagates")
    print("  BACKWARDS from the goal, one square per visit - which is why early")
    print("  episodes look random and progress is sudden rather than smooth.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].plot(hist, color=PALETTE["grey"], lw=0.7, alpha=0.5, label="episode")
    axes[0].plot(np.arange(len(smooth(hist))) + 12, smooth(hist),
                 color=PALETTE["blue"], lw=2, label="25-episode mean")
    axes[0].set_xlabel("episode"); axes[0].set_ylabel("total reward")
    axes[0].set_title("Learning curve")
    axes[0].legend(fontsize=9)

    V = Q.max(axis=1).reshape(env.size, env.size)
    im = axes[1].imshow(V, cmap="viridis", aspect="equal")
    for r in range(env.size):
        for c in range(env.size):
            mark = ("G" if (r, c) == env.goal
                    else "X" if (r, c) in env.penalties
                    else ARROWS[int(np.argmax(Q[env.state_id((r, c))]))])
            axes[1].text(c, r, mark, ha="center", va="center",
                         color="white", fontweight="bold", fontsize=13)
    axes[1].set_title("State value  max_a Q(s,a),  with the greedy policy")
    axes[1].set_xticks([]); axes[1].set_yticks([]); axes[1].grid(False)
    fig.colorbar(im, ax=axes[1], shrink=0.8)

    fig.suptitle("Tabular Q-learning on a 5x5 grid",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    print(f"\n  saved  {save_figure(fig, OUT / '1_qlearning.png').name}")
    return Q


# ------------------------------------------ 3. exploration vs exploitation
def figure_epsilon(env: GridWorld) -> None:
    """The defining trade of RL. Too little exploration and the agent locks
    onto the first adequate route it finds; too much and it never commits.
    """
    settings = [0.0, 0.05, 0.2, 0.5]
    print(f"  {'epsilon':>8} {'final 50 reward':>17} {'states visited':>16} "
          f"{'reached goal':>14}")
    curves = {}
    for eps in settings:
        rewards, visited, solved = [], set(), 0
        for seed in range(5):                  # several seeds - one run is noise
            Q, hist = train(env, episodes=500, epsilon=eps, seed=SEED + seed)
            rewards.append(np.mean(hist[-50:]))
            visited.update(np.where(Q.any(axis=1))[0].tolist())
            solved += int(np.mean(hist[-50:]) > 0)
        curves[eps] = (float(np.mean(rewards)), len(visited), solved)
        print(f"  {eps:>8.2f} {curves[eps][0]:>17.2f} "
              f"{curves[eps][1]:>12}/{env.n_states} {solved:>11}/5")

    print("\n  Read that table carefully, because it does NOT say what the")
    print("  textbook version says. epsilon = 0 - pure exploitation, never a")
    print("  random action - gets the BEST reward here. So why does anyone")
    print("  bother with exploration?")
    print("\n  Because this environment explores for free, and it is worth")
    print("  seeing exactly why. Q starts at ZERO and every step earns -1, so")
    print("  any action the agent has already tried has a NEGATIVE value while")
    print("  every untried action still sits at 0. argmax therefore prefers the")
    print("  untried one. Zero initialisation plus negative rewards IS")
    print("  OPTIMISTIC INITIALISATION - a systematic, non-random exploration")
    print("  strategy hiding in the reward design.")

    # Controlled test of that claim: remove the step cost so zero-init is no
    # longer optimistic, and rerun the same agent.
    class NoStepCost(GridWorld):
        def step(self, a):                                   # noqa: D102
            s, r, d = super().step(a)
            return s, (0.0 if r == -1.0 else r), d

    flat = NoStepCost(env.size)
    print("\n  Testing it: remove the -1 step cost so an untried action is no")
    print("  longer more attractive than a tried one. Same agent, same grid.\n")
    print(f"  {'epsilon':>8} {'reward':>9} {'states visited':>16}")
    for eps in (0.0, 0.05, 0.2):
        rs, vis = [], []
        for s in range(5):
            Q, h = train(flat, episodes=500, epsilon=eps, seed=SEED + s)
            rs.append(np.mean(h[-50:]))
            vis.append(int(Q.any(axis=1).sum()))
        print(f"  {eps:>8.2f} {np.mean(rs):>9.2f} "
              f"{np.mean(vis):>12.1f}/{env.n_states}")

    print("\n  Now epsilon = 0 collapses completely: 0.00 reward and ZERO states")
    print("  ever updated. With no step cost and no randomness the agent walks")
    print("  into the same wall forever, learns nothing, and never finds the")
    print("  goal. A tiny epsilon = 0.05 rescues it entirely.")
    print("\n  THAT is the real lesson, and it is more useful than 'always")
    print("  explore': exploration has to come from SOMEWHERE, but epsilon-")
    print("  greedy randomness is only one source. Optimistic initialisation,")
    print("  count-based bonuses and entropy regularisation are others, and a")
    print("  reward structure can supply it accidentally.")
    print("\n  Back on the original grid, high epsilon hurts for a different")
    print("  reason: a random step can land on a penalty square, which ends the")
    print("  episode at -10. Exploration damages the reward you COLLECT while")
    print("  learning without necessarily damaging what you have LEARNED - all")
    print("  four settings visited 22 of 25 states.")
    print("\n  Hence DECAY: explore early, exploit later.")

    # decaying epsilon, implemented inline so the mechanism is visible
    rng = np.random.default_rng(SEED)
    Q = np.zeros((env.n_states, env.n_actions))
    alpha, gamma = 0.1, 0.95
    decay_hist, eps_track = [], []
    for ep in range(500):
        eps = max(0.02, 0.9 * (0.99 ** ep))
        eps_track.append(eps)
        s, done, total, steps = env.reset(), False, 0.0, 0
        while not done and steps < 100:
            a = (int(rng.integers(4)) if rng.random() < eps
                 else int(np.argmax(Q[s])))
            s2, r, done = env.step(a)
            Q[s, a] += alpha * (r + gamma * Q[s2].max() * (not done) - Q[s, a])
            s, total, steps = s2, total + r, steps + 1
        decay_hist.append(total)

    print(f"\n  Decaying epsilon (0.90 -> 0.02): final 50 reward "
          f"{np.mean(decay_hist[-50:]):.2f}")
    best_fixed = max(curves, key=lambda e: curves[e][0])
    print(f"  Best FIXED epsilon ({best_fixed}):            "
          f"{curves[best_fixed][0]:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for eps, colour in zip(settings, ("vermillion", "blue", "green", "orange")):
        _, h = train(env, episodes=500, epsilon=eps, seed=SEED)
        axes[0].plot(np.arange(len(smooth(h))) + 12, smooth(h),
                     color=PALETTE[colour], lw=1.7, label=f"epsilon = {eps}")
    axes[0].plot(np.arange(len(smooth(decay_hist))) + 12, smooth(decay_hist),
                 color=PALETTE["black"], lw=2.2, ls="--", label="decaying")
    axes[0].set_xlabel("episode"); axes[0].set_ylabel("reward (25-ep mean)")
    axes[0].set_title("Exploration rate decides everything")
    axes[0].legend(fontsize=9, loc="lower right")

    axes[1].plot(eps_track, color=PALETTE["vermillion"], lw=2)
    axes[1].set_xlabel("episode"); axes[1].set_ylabel("epsilon")
    axes[1].set_title("The decay schedule:\nexplore early, exploit late")

    fig.suptitle("Exploration vs exploitation", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    print(f"\n  saved  {save_figure(fig, OUT / '2_exploration.png').name}")


# ---------------------------------------------------- 4. the discount factor
def figure_gamma(env: GridWorld) -> None:
    """gamma says how much a reward one step later is worth. It is not a
    tuning knob - it changes what problem the agent is solving.
    """
    print(f"  {'gamma':>7} {'final 50 reward':>17} {'effective horizon':>19}")
    for g in (0.0, 0.5, 0.9, 0.99):
        rs = [np.mean(train(env, episodes=500, gamma=g, seed=SEED + s)[1][-50:])
              for s in range(5)]
        horizon = "1 step" if g == 0 else f"~{1 / (1 - g):.0f} steps"
        print(f"  {g:>7.2f} {np.mean(rs):>17.2f} {horizon:>19}")

    print("\n  gamma = 0 makes the agent COMPLETELY MYOPIC: it only values the")
    print("  immediate reward, so every step looks like -1 and it has no reason")
    print("  to prefer any direction. It cannot solve a problem whose payoff is")
    print("  several steps away, which is most problems worth solving.")
    print("\n  As gamma rises the agent sees further - 1/(1-gamma) is the rough")
    print("  effective horizon, so gamma=0.99 means it is planning about 100")
    print("  steps ahead. On a 5x5 grid anything above ~0.9 is plenty.")
    print("\n  gamma is part of the PROBLEM DEFINITION, not a hyperparameter to")
    print("  tune for score. Choosing it is choosing how much you care about")
    print("  the future, which is a business decision - a recommender optimising")
    print("  gamma=0 maximises the next click and burns long-term engagement.")


# ------------------------------------------------- 5. Q-learning vs SARSA
def figure_sarsa(env: GridWorld) -> None:
    """The cliff problem, which is where the two algorithms visibly differ.

    A corridor of penalty squares sits along the shortest route. The optimal
    path hugs it; the SAFE path detours. Q-learning learns the optimal path
    because it bootstraps from the best next action. SARSA learns the safe
    path because it bootstraps from what it will ACTUALLY do, including the
    occasional random step into the cliff.
    """
    # The Sutton & Barto cliff-walking layout. The cliff must sit BETWEEN the
    # start and the goal on the bottom row, so that the shortest route runs
    # along the row directly above it. An earlier version put the cliff one row
    # up, which left a safe path along the bottom and the two algorithms
    # produced identical policies - no trade-off to observe.
    # penalty -100, as in Sutton & Barto. At -10 the detour costs more than
    # the risk is worth and BOTH algorithms hug the cliff - no trade-off to see.
    cliff = GridWorld(size=5, penalty=-100.0)
    cliff.penalties = {(4, 1), (4, 2), (4, 3)}
    cliff.start, cliff.goal = (4, 0), (4, 4)

    print("  Cliff walking, the Sutton & Barto setup. Cliff reward is -100 and")
    print("  exploration is high (epsilon = 0.25), so a slip is both likely and")
    print("  expensive. The bottom row between start and goal is the cliff:")
    print("      .  .  .  .  .")
    print("      .  .  .  .  .")
    print("      .  .  .  .  .")
    print("      .  .  .  .  .     <- the RISKY route, one slip from the cliff")
    print("      S  X  X  X  G")
    print("\n  The shortest path runs along row 3, directly above the cliff. Any")
    print(f"  random step downwards ends the episode at {cliff.penalty:.0f}.\n")

    results = {}
    for name, is_sarsa in (("Q-learning (off-policy)", False),
                           ("SARSA (on-policy)", True)):
        rs, Qs = [], None
        for s in range(5):
            Q, h = train(cliff, episodes=800, epsilon=0.25,
                         sarsa=is_sarsa, seed=SEED + s)
            rs.append(np.mean(h[-100:]))
            if s == 0:
                Qs = Q
        results[name] = (float(np.mean(rs)), Qs)
        print(f"  {name:<26} mean reward over last 100 episodes: "
              f"{results[name][0]:6.2f}")

    print("\n  The policies each one learned:\n")
    for name, (_, Q) in results.items():
        print(f"    {name}")
        print(cliff.render_policy(Q))
        print()

    print("  The single line of difference:")
    print("    Q-learning   target = r + gamma * MAX Q[s2]        (best action)")
    print("    SARSA        target = r + gamma * Q[s2, a2]        (actual action)")
    print("\n  Q-learning learns the value of the OPTIMAL policy while behaving")
    print("  exploratorily - that is what 'off-policy' means. SARSA learns the")
    print("  value of the policy it is ACTUALLY FOLLOWING, exploration included,")
    print("  so a route that is only safe if you never slip gets marked down.")
    print("\n  Which you want depends on whether exploration is free. In a")
    print("  simulator, take Q-learning and the optimal path. On real hardware")
    print("  or a live system where a random action has a real cost, SARSA's")
    print("  caution is the correct objective, not a limitation.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, (name, (_, Q)) in zip(axes, results.items()):
        V = Q.max(axis=1).reshape(cliff.size, cliff.size)
        ax.imshow(V, cmap="viridis", aspect="equal")
        for r in range(cliff.size):
            for c in range(cliff.size):
                mark = ("G" if (r, c) == cliff.goal
                        else "X" if (r, c) in cliff.penalties
                        else ARROWS[int(np.argmax(Q[cliff.state_id((r, c))]))])
                ax.text(c, r, mark, ha="center", va="center", color="white",
                        fontweight="bold", fontsize=14)
        ax.set_title(name, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Same world, same rewards, different objective - "
                 "X marks the cliff", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    print(f"  saved  {save_figure(fig, OUT / '3_qlearning_vs_sarsa.png').name}")


def main() -> None:
    use_house_style()
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print("  REINFORCEMENT LEARNING")
    print("=" * 72)
    print("\n  Supervised learning is handed the right answer for every example.")
    print("  RL gets only a REWARD, usually long after the action that earned")
    print("  it, and has to work out which decisions deserve the credit.")

    env = GridWorld(5)

    banner("1-2. The world, and Q-learning on it")
    figure_learning(env)

    banner("3. Exploration vs exploitation")
    figure_epsilon(env)

    banner("4. The discount factor, gamma")
    figure_gamma(env)

    banner("5. Q-learning vs SARSA: off-policy and on-policy")
    figure_sarsa(env)

    banner("What to take away")
    print("  - RL learns from CONSEQUENCES, not from labelled answers. The hard")
    print("    part is credit assignment: which of the last 20 actions earned")
    print("    the reward?")
    print("  - The Q-learning update is the whole algorithm:")
    print("      Q[s,a] += alpha * (r + gamma * max Q[s'] - Q[s,a])")
    print("    The bracket is the temporal-difference error, and value")
    print("    propagates backwards from the goal one step per visit.")
    print("  - EXPLORATION IS THE CRITICAL HYPERPARAMETER. epsilon=0 can get")
    print("    stuck forever; epsilon=0.5 never commits. Decay it.")
    print("  - Exploration hurts the reward you COLLECT while learning without")
    print("    necessarily hurting what you LEARN. Do not confuse the two.")
    print("  - gamma is part of the problem definition, not a tuning knob. It")
    print("    sets how far ahead the agent cares, roughly 1/(1-gamma) steps.")
    print("  - OFF-POLICY (Q-learning) learns the optimal policy's value while")
    print("    behaving exploratorily; ON-POLICY (SARSA) learns the value of")
    print("    what it actually does, so it avoids risky-but-optimal routes.")
    print("    Pick by whether a random action costs anything real.")
    print("  - A Q-TABLE ONLY WORKS FOR SMALL DISCRETE STATE SPACES. 25 states")
    print("    here; a chessboard has ~10^43. Replacing the table with a neural")
    print("    network is Deep Q-Networks, and that is where 03_neural_networks")
    print("    picks the thread up.")
    print(f"\n  Figures in  {OUT.resolve().relative_to(Path(__file__).resolve().parents[2])}")
    print("=" * 72)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        main()
