"""Batch vs online inference: the decision, and the cost bug hiding inside it.

Serving is ~20% of the exam and it opens with one question: does this
prediction need to exist in milliseconds, or by morning?

Get it wrong in the expensive direction and you pay for an always-warm
endpoint to do work a nightly job could have done. Get it wrong in the other
direction and a user waits eight hours for a recommendation.

  1. The two shapes, and what actually differs
  2. Cost, computed - where the crossover is
  3. The anti-pattern: an endpoint in a loop
  4. Scale-to-zero vs min_replica_count, and the inverse trap

EXECUTED IN FULL. Every number printed below is computed from the rates at
the top of the file, not quoted.

NOTE ON PRICES: approximate published list prices, and they move. They are
printed so a stale figure is visible rather than hidden. The RATIOS are the
durable part - the crossover moves with the prices, the arithmetic does not.

Run:
    python 07_serving/01_batch_vs_online/batch_vs_online.py

Concept reference: Success Guide, Part 4.1-4.2. Blueprint section 4 (~20%).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))

import matplotlib.pyplot as plt  # noqa: E402
from plotting import PALETTE, save_figure, use_house_style  # noqa: E402

SEED = 42
OUT = Path(__file__).parent / "images"

# ---------------------------------------------------------------- the rates
# n1-standard-4-ish serving replica, and batch compute billed per job.
ONLINE_REPLICA_PER_HOUR = 0.25     # $/replica-hour, billed whether used or not
BATCH_NODE_PER_HOUR = 0.25         # $/node-hour, billed only while the job runs
BATCH_THROUGHPUT_PER_HOUR = 2_000_000   # rows one batch node scores per hour
ONLINE_QPS_PER_REPLICA = 50        # requests/second one replica sustains


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ------------------------------------------------------------------ 1. shape
def section_shapes() -> None:
    rule("1. THE TWO SHAPES")
    rows = [
        ("latency",            "minutes to hours",        "milliseconds"),
        ("endpoint needed?",   "NO",                      "yes"),
        ("cost shape",         "per job, while running",  "per replica-hour, always warm"),
        ("input / output",     "BigQuery or GCS",         "JSON request / response"),
        ("typical use",        "nightly scoring, backfill", "apps, real-time decisions"),
    ]
    print(f"  {'':<18} {'BATCH':<26} ONLINE")
    print("  " + "-" * 68)
    for label, batch, online in rows:
        print(f"  {label:<18} {batch:<26} {online}")

    print("\n  The row that decides most exam questions is the second one.")
    print("  BATCH PREDICTION DOES NOT REQUIRE AN ENDPOINT. It reads straight")
    print("  from BigQuery or GCS and writes results back.")


# ------------------------------------------------------------------- 2. cost
def section_cost() -> int:
    rule("2. COST - where the crossover actually is")

    daily_rows = np.array([1e4, 1e5, 1e6, 1e7, 1e8])

    print(f"  Online : ${ONLINE_REPLICA_PER_HOUR}/replica-hour, {ONLINE_QPS_PER_REPLICA} QPS per replica")
    print(f"  Batch  : ${BATCH_NODE_PER_HOUR}/node-hour, {BATCH_THROUGHPUT_PER_HOUR:,} rows/hour per node")
    print()
    print(f"  {'rows/day':>12} {'batch $/day':>13} {'online $/day':>14} {'ratio':>9}")
    print("  " + "-" * 52)

    batch_costs, online_costs = [], []
    for rows in daily_rows:
        # Batch: pay only for the hours the job runs.
        batch_hours = rows / BATCH_THROUGHPUT_PER_HOUR
        batch_cost = batch_hours * BATCH_NODE_PER_HOUR

        # Online: size the fleet for the PEAK, then pay for it all day.
        # Traffic is never flat, so assume peak is 3x the daily mean rate.
        mean_qps = rows / 86_400
        peak_qps = mean_qps * 3
        replicas = max(1, int(np.ceil(peak_qps / ONLINE_QPS_PER_REPLICA)))
        online_cost = replicas * 24 * ONLINE_REPLICA_PER_HOUR

        batch_costs.append(batch_cost)
        online_costs.append(online_cost)
        ratio = online_cost / batch_cost if batch_cost else float("inf")
        print(f"  {rows:>12,.0f} {batch_cost:>13,.2f} {online_cost:>14,.2f} {ratio:>8.0f}x")

    # Where does online stop being absurd? Where the fleet is busy enough that
    # its always-on cost approaches the batch job's.
    crossover = BATCH_THROUGHPUT_PER_HOUR * 24 * (BATCH_NODE_PER_HOUR / ONLINE_REPLICA_PER_HOUR)
    print(f"\n  A single always-on replica costs the same as batch-scoring")
    print(f"  {crossover:,.0f} rows/day. Below that, an endpoint is pure overhead.")
    print("\n  NOTE what the online column is really showing: the cost barely moves")
    print("  until traffic is high, because you pay for a warm replica either way.")
    print("  ONLINE COST IS DRIVEN BY UPTIME, NOT BY USAGE. That is the whole")
    print("  difference, and it is why 10,000 rows/day costs the same as 1 million.")
    return int(crossover)


# -------------------------------------------------------------- 3. the bug
def section_antipattern() -> None:
    rule("3. THE ANTI-PATTERN - an endpoint in a loop")

    rows = 50_000_000            # score the whole customer base
    rtt_ms = 40                  # per online request, round trip
    client_parallel = 32         # generous client-side concurrency
    online_replicas = 8          # enough fleet to absorb that concurrency

    serial_hours = rows * (rtt_ms / 1000) / 3600
    online_hours = serial_hours / client_parallel
    online_cost = online_hours * online_replicas * ONLINE_REPLICA_PER_HOUR

    # Batch node-hours are WORK, and work is fixed. Adding nodes buys wall
    # clock at no extra cost, which is the property that matters here.
    batch_node_hours = rows / BATCH_THROUGHPUT_PER_HOUR
    batch_cost = batch_node_hours * BATCH_NODE_PER_HOUR

    print(f"  Nightly scoring of {rows:,} customers.\n")
    print(f"  Endpoint in a loop : {online_hours:>6.1f} h wall clock   ${online_cost:>8,.2f}")
    print(f"                       ({client_parallel}-way client concurrency, "
          f"{online_replicas} replicas)\n")
    print(f"  Batch prediction   : {batch_node_hours:>6.1f} NODE-hours   ${batch_cost:>8,.2f}")
    print("                       wall clock is whatever you buy with nodes:")
    for nodes in (1, 5, 10, 25):
        print(f"                         {nodes:>3} nodes -> "
              f"{batch_node_hours / nodes:>5.1f} h   "
              f"${batch_node_hours * BATCH_NODE_PER_HOUR:>6,.2f}  (cost unchanged)")

    print(f"\n  Cost: {online_cost / batch_cost:.0f}x more expensive, for identical predictions.")
    print("\n  And note WHY the wall-clock column is not the point. Batch cost is")
    print("  node-HOURS - that is work, and work is fixed. Adding nodes buys speed")
    print("  at the same total price. The endpoint has no such lever: its cost is")
    print("  replicas x uptime, so going faster there means paying more.")
    print("\n  If an exam option says \"deploy the model to an endpoint and loop over")
    print("  rows\", it is the wrong tool: more expensive, an endpoint to operate")
    print("  that the job did not need, and a client loop to write and babysit.")
    print("  Batch prediction exists for exactly this, and needs no endpoint.")


# ------------------------------------------------------- 4. scale to zero
def section_scale_to_zero() -> None:
    rule("4. SCALE-TO-ZERO vs min_replica_count")

    # A spiky, low-traffic internal tool: busy 2 hours a day.
    busy_hours = 2
    idle_hours = 24 - busy_hours

    always_warm = 24 * ONLINE_REPLICA_PER_HOUR
    scale_to_zero = busy_hours * ONLINE_REPLICA_PER_HOUR

    print(f"  An internal tool that is genuinely used {busy_hours} hours a day.\n")
    print(f"  min_replica_count = 1 : ${always_warm:>6,.2f}/day   "
          f"({idle_hours} h of it idle)")
    print(f"  scale to zero         : ${scale_to_zero:>6,.2f}/day")
    print(f"  wasted                : ${always_warm - scale_to_zero:>6,.2f}/day "
          f"= {(1 - scale_to_zero / always_warm) * 100:.0f}% of the bill\n")

    print("  \"Traffic is intermittent and we must not pay when idle\"")
    print("      -> Cloud Run, which scales to zero.")
    print("      An Agent Platform endpoint with min_replica_count = 1 never")
    print("      scales to zero and bills continuously. That IS the cost bug.")
    print("\n  AND THE INVERSE TRAP, which the exam also asks:")
    print("  \"No cold starts, consistent p99 latency\"")
    print("      -> min_replica_count >= 1 is now CORRECT, and scale-to-zero is")
    print("      wrong. A cold start is a multi-second first request.")
    print("\n  Same two options. The requirement decides, not the cost.")


# ---------------------------------------------------------------- the figure
def figure(crossover: int) -> None:
    use_house_style()
    rows = np.logspace(3, 8, 200)

    batch = (rows / BATCH_THROUGHPUT_PER_HOUR) * BATCH_NODE_PER_HOUR
    mean_qps = rows / 86_400
    replicas = np.maximum(1, np.ceil(mean_qps * 3 / ONLINE_QPS_PER_REPLICA))
    online = replicas * 24 * ONLINE_REPLICA_PER_HOUR

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.loglog(rows, batch, color=PALETTE["blue"], lw=2, label="batch (per job)")
    ax.loglog(rows, online, color=PALETTE["vermillion"], lw=2,
              label="online (always warm)")
    ax.axvline(crossover, color=PALETTE["green"], ls="--", lw=1.4,
               label=f"parity ~{crossover:,.0f} rows/day")
    ax.set_xlabel("predictions per day")
    ax.set_ylabel("cost per day (USD)")
    ax.set_title("Online serving bills uptime; batch bills work")
    ax.legend(frameon=False, fontsize=9)
    save_figure(fig, OUT / "batch_vs_online_cost.png")
    print(f"\n  figure -> {OUT / 'batch_vs_online_cost.png'}")


def main() -> None:
    print("BATCH vs ONLINE INFERENCE")
    section_shapes()
    crossover = section_cost()
    section_antipattern()
    section_scale_to_zero()
    figure(crossover)

    rule("WHAT DECIDES IT")
    print("""  Not cost. The REQUIREMENT decides, and cost tells you how badly a
  mismatch hurts:

    a user is waiting                 -> online, and pay for warm replicas
    a nightly file / table            -> batch, and never create an endpoint
    intermittent internal tool        -> Cloud Run, scale to zero
    strict p99, no cold starts        -> online with min_replica_count >= 1
    scoring where the data already is -> BigQuery ML ML.PREDICT
    offline or on-device              -> edge, after quantization

  The exam's favourite wrong answer is an endpoint doing batch work.""")


if __name__ == "__main__":
    main()
