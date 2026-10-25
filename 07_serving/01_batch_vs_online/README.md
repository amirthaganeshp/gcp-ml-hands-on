# Batch vs online inference

> **Concept reference:** Success Guide, Part 4.1–4.2 · Blueprint section 4 (~20%)
> **Datasets:** none — this is arithmetic over published rates
> **Runtime:** ~3 s on a laptop CPU

```bash
python 07_serving/01_batch_vs_online/batch_vs_online.py
```

---

## The idea

Serving opens with one question: **does this prediction need to exist in milliseconds, or by
morning?**

Everything else follows. Get it wrong in the expensive direction and you pay for an always-warm
endpoint to do work a nightly job could have done. Get it wrong the other way and a user waits
eight hours for a recommendation.

| | **batch** | **online** |
|---|---|---|
| latency | minutes to hours | milliseconds |
| **endpoint needed?** | **no** | yes |
| cost shape | per job, while running | **per replica-hour, always warm** |
| input / output | BigQuery or GCS | JSON request / response |
| typical use | nightly scoring, backfills | apps, real-time decisions |

**The second row decides most exam questions.** Batch prediction does not require an endpoint — it
reads straight from BigQuery or GCS and writes the results back.

## 1. Online cost is driven by uptime, not usage

Computed across five workloads:

| rows/day | batch $/day | online $/day | ratio |
|---|---|---|---|
| 10,000 | 0.00 | **6.00** | 4800× |
| 100,000 | 0.01 | **6.00** | 480× |
| 1,000,000 | 0.12 | **6.00** | 48× |
| 10,000,000 | 1.25 | 42.00 | 34× |
| 100,000,000 | 12.50 | 420.00 | 34× |

**Look at the first three rows of the online column: identical.** Ten thousand predictions a day
costs exactly what a million does, because you are paying for a warm replica either way. The batch
column tracks the work; the online column tracks the clock. Only past ~1M/day does traffic force a
second replica and the online cost start to climb.

A single always-on replica costs the same as batch-scoring **48 million rows/day**. Below that, the
endpoint is overhead — and the ratio column shows how badly: **4800× at ten thousand rows a day.**

## 2. The anti-pattern — an endpoint in a loop

Nightly scoring of 50,000,000 customers:

```
Endpoint in a loop :   17.4 h wall clock   $  34.72   (32-way concurrency, 8 replicas)

Batch prediction   :   25.0 NODE-hours     $   6.25
                       1 node  -> 25.0 h   $ 6.25
                       5 nodes ->  5.0 h   $ 6.25
                      25 nodes ->  1.0 h   $ 6.25     <- cost unchanged
```

**5.6× more expensive for identical predictions** — and note what the wall-clock column shows.
Batch cost is node-*hours*, which is **work**, and work is fixed. Adding nodes buys speed at the
same total price. The endpoint has no such lever: its cost is replicas × uptime, so going faster
there means paying more.

> If an option says *"deploy the model to an endpoint and loop over rows"*, it is the wrong tool:
> more expensive, an endpoint to operate that the job never needed, and a client loop to write and
> babysit.

## 3. Scale-to-zero vs `min_replica_count` — and the inverse trap

An internal tool genuinely used 2 hours a day:

| | $/day |
|---|---|
| `min_replica_count = 1` | **6.00** (22 h of it idle) |
| scale to zero | 0.50 |
| **wasted** | **5.50 — 92% of the bill** |

> **"Traffic is intermittent and we must not pay when idle"** → **Cloud Run**, which scales to
> zero. An Agent Platform endpoint with `min_replica_count = 1` never scales to zero and bills
> continuously. That *is* the cost bug the question describes.

**And the inverse, which the exam also asks:**

> **"No cold starts, consistent p99 latency"** → `min_replica_count >= 1` is now **correct**, and
> scale-to-zero is wrong. A cold start is a multi-second first request.

Same two options, opposite answers. **The requirement decides, not the cost.**

## The exam angle

| the scenario says | the answer |
|---|---|
| a user is waiting | online, and pay for warm replicas |
| a nightly file or table | **batch — and never create an endpoint** |
| intermittent internal tool, no idle cost | **Cloud Run** (scale to zero) |
| strict p99, no cold starts | online, `min_replica_count >= 1` |
| scoring where the data already lives | **BigQuery ML `ML.PREDICT`** |
| offline, on-device, or privacy at the edge | edge, after quantization |

## On the prices

Approximate published list rates, and **they move** — the script prints them so a stale figure is
visible rather than hidden. The **ratios** are the durable part: the crossover moves with the
prices, the arithmetic does not.

## Things to try

- Change `ONLINE_QPS_PER_REPLICA` and watch where the crossover goes. Throughput per replica is
  the number you most often get wrong when sizing a fleet.
- Set peak traffic to 1× the mean instead of 3×. Real traffic is never flat, and sizing for the
  mean is how endpoints fall over at 09:00.
