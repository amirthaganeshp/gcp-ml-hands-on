# Rollout strategies — canary, A/B, blue-green, shadow

> **Concept reference:** Success Guide, Part 4.4 · Blueprint section 4 (~20%)
> **Datasets:** none — simulated traffic and a power calculation
> **Runtime:** ~4 s on a laptop CPU

```bash
python 07_serving/02_rollout_strategies/rollout_strategies.py
```

---

## The idea

**Canary and A/B are not synonyms, and the exam separates them.**

| | mechanism | purpose |
|---|---|---|
| **Canary** | 95/5 split, then ramp | **limit blast radius** of a bad model |
| **A/B test** | 50/50, measure a business metric | **decide which model is better** |
| **Blue-green** | deploy, then flip 100% at once | fast cutover, instant rollback |
| **Shadow** | mirror traffic, discard responses | validate on real traffic, zero risk |

Canary, A/B and blue-green are **one endpoint with a traffic split**. You do not need two
endpoints, a load balancer, or a Cloud Function routing requests — options proposing that
infrastructure are over-engineered distractors.

**Shadow is the exception, and it is not a split.** A split sends each request to *one* model; a
shadow sends it to *both*, returns the live model's answer and discards the new one. That means
mirroring requests (or replaying logged ones) to the shadow model — a traffic split of 0% to the new
model sends it nothing at all.

## 1. Canary is containment, and needs no statistics

A new model is broken: 20% of its responses are wrong. Daily traffic 200,000.

| traffic to new | users affected/day | vs full rollout |
|---|---|---|
| 100% | 40,000 | 100% |
| 50% | 20,000 | 50% |
| **5%** | **2,000** | **5%** |
| 1% | 400 | 1% |

**The damage scales with the share.** That is the whole mechanism — 5% of traffic means 5% of the
harm while you watch the error rate. It is a containment decision.

## 2. A/B is measurement, and it has a price in traffic

Baseline conversion 5%, α = 0.05, power = 0.80:

| relative lift to detect | n per arm | days at 50/50 |
|---|---|---|
| 20% | 8,158 | 0.1 |
| 10% | 31,234 | 0.3 |
| 5% | 122,124 | 1.2 |
| **2%** | **752,703** | **7.5** |

**Halving the effect you want to detect roughly quadruples the sample you need.** That is why
*"we'll just watch it for a day"* does not settle a 2% question — it settles a 20% one.

## 3. The two trade the same quantity in opposite directions

To detect a 5% lift (122,124 requests per arm):

| split | new-model traffic/day | days to decide |
|---|---|---|
| 95/5 canary | 10,000 | **12.2** |
| 50/50 A/B | 100,000 | **1.2** |

**10× longer on the canary split, for the same answer.**

> The canary is not *useless* for measurement — just slow. And slow is the point: the traffic share
> you chose was a **safety** decision, not a statistical one. A 5% canary exposes 5% of users and
> takes 12.2 days; a 50/50 A/B exposes half your users and decides in 1.2.

## 4. Shadow validates behaviour, not outcomes

Mirror 20,000 real requests to a candidate and throw its responses away:

```
Predictions that differ : 826 (4.1%)
Users affected          : 0
```

You learn latency, error rate and disagreement rate **on real traffic with no user exposure**.
What you cannot learn is whether it converts better — nobody ever saw its output.

**Shadow validates behaviour. A/B validates outcomes.** Confusing them is the same class of error
as confusing canary with A/B.

## The exam angle

| the scenario says | the answer |
|---|---|
| "roll it out safely", "limit exposure" | **canary** (95/5, then ramp) |
| "which converts better", a metric, significance | **A/B** (50/50) |
| "fast cutover with instant rollback" | **blue-green** |
| "validate on real traffic with no user risk" | **shadow** |
| two endpoints + a load balancer to route | **over-engineered distractor** |

> If the question mentions a **business metric** or **significance**, it is A/B. If it mentions
> **rolling out safely**, it is canary. Answering with the other one is the trap the wording is
> built around.

## Things to try

- Drop `BASELINE_CONV` to 0.5%. Rare events need far more traffic — this is why conversion tests on
  low-converting funnels run for weeks.
- Set the canary share to 20% and see how the 12.2 days moves. There is a real trade here, and it
  is yours to choose rather than a rule to memorise.
