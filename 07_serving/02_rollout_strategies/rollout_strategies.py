"""Canary, A/B, blue-green and shadow: four rollouts that are not synonyms.

The exam separates canary from A/B deliberately, and the distinction is not
stylistic:

    CANARY is RISK MANAGEMENT  - expose a small slice, see if it breaks
    A/B is MEASUREMENT         - get statistical evidence which is better

They use the same mechanism (an endpoint traffic split) for opposite
purposes, and the traffic share that makes one safe makes the other
unusable. This file computes both sides of that.

  1. The four strategies
  2. Canary: blast radius, measured
  3. A/B: how many requests you actually need
  4. Why a 5% canary cannot answer an A/B question
  5. Shadow: real traffic, zero user risk

EXECUTED IN FULL. Every number below is computed or simulated, not quoted.

Run:
    python 07_serving/02_rollout_strategies/rollout_strategies.py

Concept reference: Success Guide, Part 4.4. Blueprint section 4 (~20%).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402
from scipy import stats  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"
rng = np.random.default_rng(SEED)

DAILY_REQUESTS = 200_000
BASELINE_CONV = 0.050        # 5% conversion on the current model


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ------------------------------------------------------------- 1. the four
def section_strategies() -> None:
    rule("1. THE FOUR STRATEGIES")
    rows = [
        ("Canary", "95/5 split, then ramp", "limit blast radius of a bad model"),
        ("A/B test", "50/50, measure a business metric", "decide which model is better"),
        ("Blue-green", "deploy, then flip 100% at once", "fast cutover, instant rollback"),
        ("Shadow", "mirror traffic, discard responses", "validate on real traffic, zero risk"),
    ]
    print(f"  {'strategy':<12} {'mechanism':<36} purpose")
    print("  " + "-" * 70)
    for name, mech, purpose in rows:
        print(f"  {name:<12} {mech:<36} {purpose}")

    print("\n  All four are ONE endpoint with a traffic split. You do not need two")
    print("  endpoints, a load balancer, or a Cloud Function routing requests -")
    print("  options proposing that infrastructure are over-engineered distractors.")


# ---------------------------------------------------------------- 2. canary
def section_canary() -> None:
    rule("2. CANARY - blast radius, measured")

    bad_rate = 0.20          # the new model is broken: 20% of responses are wrong
    print(f"  A new model is deployed and it is broken: {bad_rate:.0%} of its")
    print(f"  responses are wrong. Daily traffic is {DAILY_REQUESTS:,} requests.\n")
    print(f"  {'traffic to new':>15} {'users affected/day':>20} {'vs full rollout':>17}")
    print("  " + "-" * 56)

    full = DAILY_REQUESTS * bad_rate
    for share in (1.00, 0.50, 0.05, 0.01):
        affected = DAILY_REQUESTS * share * bad_rate
        print(f"  {share:>14.0%} {affected:>20,.0f} {affected / full:>16.0%}")

    print("\n  That is the entire point of a canary: the damage scales with the")
    print("  share, so 5% of traffic means 5% of the harm while you look at the")
    print("  error rate. It is a CONTAINMENT decision, and it needs no statistics.")


# ------------------------------------------------------------------- 3. A/B
def required_n(p1: float, lift_rel: float, alpha: float = 0.05,
               power: float = 0.80) -> int:
    """Requests PER ARM to detect a relative lift on a proportion."""
    p2 = p1 * (1 + lift_rel)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar)) +
           z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p2 - p1) ** 2))


def section_ab() -> dict[float, int]:
    rule("3. A/B - how many requests you actually need")

    print(f"  Baseline conversion {BASELINE_CONV:.1%}, alpha 0.05, power 0.80.")
    print("  How much traffic does each arm need to detect a given lift?\n")
    print(f"  {'relative lift':>14} {'n per arm':>14} {'days at 50/50':>16}")
    print("  " + "-" * 48)

    needs: dict[float, int] = {}
    for lift in (0.20, 0.10, 0.05, 0.02):
        n = required_n(BASELINE_CONV, lift)
        needs[lift] = n
        days = n / (DAILY_REQUESTS / 2)
        print(f"  {lift:>13.0%} {n:>14,} {days:>16.1f}")

    print("\n  Smaller effects cost quadratically more traffic. Halving the lift")
    print("  you want to detect roughly QUADRUPLES the sample you need - which is")
    print("  why 'we'll just watch it for a day' does not settle a 2% question.")
    return needs


# ------------------------------------------------- 4. the confusion, costed
def section_why_not_canary(needs: dict[float, int]) -> None:
    rule("4. WHY A 5% CANARY CANNOT ANSWER AN A/B QUESTION")

    lift = 0.05
    n = needs[lift]
    print(f"  Say you want to detect a {lift:.0%} relative lift "
          f"({n:,} requests per arm).\n")
    print(f"  {'split':>10} {'new-model traffic/day':>24} {'days to decide':>16}")
    print("  " + "-" * 54)
    for share, label in ((0.05, "95/5 canary"), (0.50, "50/50 A/B")):
        per_day = DAILY_REQUESTS * share
        days = n / per_day
        print(f"  {label:>10} {per_day:>24,.0f} {days:>16.1f}")

    canary_days = n / (DAILY_REQUESTS * 0.05)
    ab_days = n / (DAILY_REQUESTS * 0.50)
    print(f"\n  {canary_days / ab_days:.0f}x longer on the canary split, for the same answer.")
    print("\n  So the two trade the same quantity in opposite directions:")
    print(f"    - a 5% canary exposes 5% of users, and takes {canary_days:.1f} days to decide")
    print(f"    - a 50/50 A/B exposes half your users, and decides in {ab_days:.1f} days")
    print("\n  The canary is not USELESS for measurement, just slow - and slow is")
    print("  the point, because the traffic share you chose was a safety decision,")
    print("  not a statistical one.")
    print("\n  Read which the scenario is asking for. If it mentions a business")
    print("  metric or significance, it is A/B. If it mentions rolling out safely,")
    print("  it is canary. Answering with the other one is the trap.")


# ---------------------------------------------------------------- 5. shadow
def section_shadow() -> None:
    rule("5. SHADOW - real traffic, zero user risk")

    n = 20_000
    # The live model and a candidate that disagrees on a slice of inputs.
    live = rng.random(n) < BASELINE_CONV
    candidate = live.copy()
    flip = rng.random(n) < 0.04
    candidate[flip] = ~candidate[flip]

    disagree = int((live != candidate).sum())
    print(f"  Mirror {n:,} real requests to a candidate model and throw its")
    print("  responses away. Users never see them.\n")
    print(f"  Predictions that differ : {disagree:,} ({disagree / n:.1%})")
    print(f"  Users affected          : 0")
    print("\n  You learn how the candidate behaves on REAL traffic - latency,")
    print("  error rate, disagreement rate - with no user exposure at all.")
    print("  What you cannot learn is whether it converts better, because nobody")
    print("  ever saw its output. Shadow validates BEHAVIOUR, not OUTCOMES.")


# ---------------------------------------------------------------- the figure
def figure() -> None:
    use_house_style()
    lifts = np.linspace(0.02, 0.30, 60)
    ns = np.array([required_n(BASELINE_CONV, l) for l in lifts])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(lifts * 100, ns / 1e3, color=PALETTE["blue"], lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("relative lift you want to detect (%)")
    ax.set_ylabel("requests per arm (thousands, log scale)")
    ax.set_title("Detecting a smaller effect costs quadratically more traffic")
    for l in (0.05, 0.10, 0.20):
        ax.axvline(l * 100, color=PALETTE["vermillion"], ls="--", lw=1, alpha=0.7)
        ax.annotate(f"{required_n(BASELINE_CONV, l):,}", (l * 100, required_n(BASELINE_CONV, l) / 1e3),
                    textcoords="offset points", xytext=(6, 6), fontsize=8,
                    color=PALETTE["vermillion"])
    save_figure(fig, OUT / "ab_sample_size.png")
    print(f"\n  figure -> {OUT / 'ab_sample_size.png'}")


def main() -> None:
    print("ROLLOUT STRATEGIES - canary, A/B, blue-green, shadow")
    section_strategies()
    section_canary()
    needs = section_ab()
    section_why_not_canary(needs)
    section_shadow()
    figure()

    rule("THE EXAM ANGLE")
    print("""  the scenario says                          the answer
  ------------------------------------------------------------------
  "roll it out safely / limit exposure"      CANARY  (95/5, then ramp)
  "which model converts better", metric,
     significance                            A/B     (50/50)
  "fast cutover with instant rollback"       BLUE-GREEN
  "validate on real traffic, no user risk"   SHADOW
  "we need two endpoints and a load
     balancer to route traffic"              over-engineered distractor

  All four are one endpoint and a traffic split.""")


if __name__ == "__main__":
    main()
