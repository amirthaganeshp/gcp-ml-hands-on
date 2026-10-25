# Running this section for real, on your own project

Everything in `05_gcp` is **write-only against the cloud** — defined, never submitted. This file is
the runbook for actually doing it, on a **personal** Google Cloud project.

It is deliberately separate from the rest of the section, because following it **spends money**.

---

## ⚠ Before anything: do not do this on a work machine

`gcloud` has a default project, and on a corporate workstation that default is very often a real
production project. A `docker push` or a training job submitted against it is a real change to a
real system.

Use a personal Google account on a personal machine. If you must keep both on one machine, isolate
them so a default can never leak:

```bash
gcloud config configurations create personal
gcloud config configurations activate personal
gcloud auth login                      # personal account
gcloud config set project YOUR-PROJECT-ID
gcloud config configurations list      # confirm which is ACTIVE before every session
```

Switching back is `gcloud config configurations activate default`.

---

## 1. Set up the project

A new Google Cloud account gets **$300 of credit for 90 days**, which is far more than this needs.
Billing must be enabled even while on credit.

```bash
gcloud services enable artifactregistry.googleapis.com aiplatform.googleapis.com
```

**Set a budget alert first, not later** — Billing → Budgets & alerts. Something small like $5 is
plenty to catch a mistake early. This is the single most useful five minutes in the whole runbook.

## 2. Create an Artifact Registry repository

This is the storage locker the image gets pushed to. It is a one-off.

```bash
gcloud artifacts repositories create ml-repo \
  --repository-format=docker \
  --location=us-central1 \
  --description="Training containers"
```

Then let Docker authenticate to it:

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

## 3. Build, tag and push

Run from `05_gcp/04_custom_training/`. The tag **is** the destination — Docker pushes to whatever
registry path the tag names, so this is not cosmetic.

```bash
export PROJECT=YOUR-PROJECT-ID
export REGION=us-central1
export IMAGE=$REGION-docker.pkg.dev/$PROJECT/ml-repo/trainer:latest

docker build -t $IMAGE .
docker push $IMAGE
```

The build is verified — it completes in ~265 s, and the base image is the slow part. The **push**
is the new step, and it uploads ~3 GB the first time. Later pushes only send changed layers, so
editing `trainer/task.py` pushes a few kilobytes.

## 4. Submit a training job

`submit.py` already contains the job definitions; it defines them rather than submitting. With a
project set and the image pushed, the calls inside it become live.

Start with the plain custom job, not the hyperparameter tuning job — a tuning job multiplies cost
by its trial count.

## 5. Costs, honestly

Approximate, and **check current pricing** — these change, and this was written in Sept 2026.

| thing | rough cost | notes |
|---|---|---|
| Artifact Registry storage | ~$0.10 / GB / month | a 3 GB image ≈ **$0.30/month**, billed while it exists |
| Vertex custom training, small CPU machine | ~$0.20 / hour | this job trains in seconds, but you pay for provisioning too — **cents per run** |
| Hyperparameter tuning | above × trials | 10 trials is 10 training runs |
| **Deployed endpoint** | **~$0.20 / hour, continuously** | **the trap — see below** |

**The endpoint is the only thing that bills while you are asleep.** A Vertex prediction endpoint
charges per node-hour whether or not anyone calls it, and `min_replica_count` cannot be 0. Deploy
one, forget it, and it quietly bills for a month. Everything else here is priced per use.

That asymmetry is not just a billing footnote — it is the reason batch prediction is the right
answer whenever predictions are not needed instantly, and the exam tests exactly that.

## 6. Cleanup

Do this the same day. In order of how much they cost you:

```bash
# 1. Endpoints FIRST - the only resource billing at rest
gcloud ai endpoints list --region=$REGION
gcloud ai endpoints undeploy-model ENDPOINT_ID --region=$REGION --deployed-model-id=ID
gcloud ai endpoints delete ENDPOINT_ID --region=$REGION

# 2. Images
gcloud artifacts docker images list $REGION-docker.pkg.dev/$PROJECT/ml-repo
gcloud artifacts docker images delete $IMAGE --delete-tags
```

**Simplest and safest: delete the whole project** when you are done. It removes every resource in
one action, with no chance of missing one.

```bash
gcloud projects delete YOUR-PROJECT-ID
```

---

## What you will have proved

- The image builds **and pushes** — the half this repository could not verify.
- A Vertex custom training job runs your own container end to end.
- `AIP_MODEL_DIR` is populated by the service and the model lands in Cloud Storage, which is the
  one part of `trainer/task.py` that cannot be exercised locally (it falls back to a local path).

That last one is the real reason to bother. Everything else here is already confirmed locally.
