# A custom training container for Vertex AI.
#
# THIS BUILDS. Verified with `docker build` (Rancher Desktop, moby): 265 s,
# image assembled, layers exported. It is NOT pushed - that needs a real
# project and an Artifact Registry repo, so the build stops locally.
#
# Note when you read the pip layers below: on this base image both are
# no-ops. The Vertex prebuilt already ships keras AND cloudml-hypertune.
#
# Build locally:
#   docker build -t ml-repo-trainer:local .
#
# Build and push for real:
#   REGION=us-central1; PROJECT=your-project-id
#   docker build -t $REGION-docker.pkg.dev/$PROJECT/ml-repo/trainer:latest .
#   docker push    $REGION-docker.pkg.dev/$PROJECT/ml-repo/trainer:latest

# Start from a Vertex AI prebuilt training image rather than bare python.
# It already contains the framework, CUDA where relevant, and the agent
# Vertex uses to report job status - all of which you would otherwise have
# to assemble and keep in step with the service.
FROM us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest

WORKDIR /app

# trainer/task.py defaults KERAS_BACKEND to torch, which is what the local
# environment has. This image ships TensorFlow and no torch, so without this
# line `import keras` fails on the first call. task.py uses setdefault, so
# the value set here wins.
ENV KERAS_BACKEND=tensorflow

# Dependencies BEFORE source. Docker caches layers, and requirements change
# far less often than code - so this ordering means editing task.py rebuilds
# one small layer instead of reinstalling every package.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# cloudml-hypertune is what report_metric() in trainer/task.py uses to send
# the objective back to the tuning service. Without it the trainer still
# runs - it degrades to a print - but a tuning job learns nothing.
RUN pip install --no-cache-dir cloudml-hypertune

COPY trainer/ ./trainer/

# ENTRYPOINT, not CMD. Vertex AI APPENDS the job's args to the entrypoint,
# so with CMD they would replace the command instead of extending it and
# every hyperparameter would be silently dropped.
ENTRYPOINT ["python", "-m", "trainer.task"]
