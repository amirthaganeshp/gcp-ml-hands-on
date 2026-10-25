#!/usr/bin/env bash
# Run every runnable example and report its exit code.
#
#   bash .github/scripts/run_list.sh fast    # core stack only (numpy/sklearn/xgboost)
#   bash .github/scripts/run_list.sh full    # + Keras/PyTorch/TF/JAX + pipelines
#
# Deliberately does NOT abort on the first failure: every item runs, so one
# broken script cannot mask the state of the rest. Exits non-zero at the end
# if anything failed.
#
# Runs locally as well as in CI - a CI-only script is one you cannot test.
set -uo pipefail

MODE="${1:-fast}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

# Find a python that has this repo's dependencies.
#   - honour $PYTHON if set
#   - otherwise the project venv, Windows layout then POSIX layout
#   - otherwise whatever is on PATH (the CI case)
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if   [ -x ".venv/Scripts/python.exe" ]; then PY=".venv/Scripts/python.exe"
  elif [ -x ".venv/bin/python" ];        then PY=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then PY="python3"
  else PY="python"; fi
fi

# ---------------------------------------------------------------- the lists
# NOT listed, on purpose: loaders.py, plotting.py, 04_frameworks/common.py and
# trainer/__init__.py are imported by other scripts, not run on their own.
# 05_gcp/{01_pretrained_apis,02_bigquery_ml,03_automl,04_custom_training} need a
# real GCP project and are documented rather than executed.

SETUP=(
  "00_setup/check_environment.py"
  "00_setup/download_datasets.py"
)

CORE=(
  "01_foundations/01_train_test_split_and_cv/splitting.py"
  "01_foundations/02_bias_variance_tradeoff/bias_variance.py"
  "01_foundations/03_confusion_matrix_and_metrics/metrics.py"
  "01_foundations/04_roc_vs_pr_curves/roc_vs_pr.py"
  "01_foundations/05_data_leakage/leakage.py"
  "01_foundations/06_feature_engineering/feature_engineering.py"
  "02_traditional_ml/01_regression/regression.py"
  "02_traditional_ml/02_classification/classification.py"
  "02_traditional_ml/03_clustering/clustering.py"
  "02_traditional_ml/04_dimensionality_reduction/dimensionality_reduction.py"
  "02_traditional_ml/05_anomaly_detection/anomaly_detection.py"
  "02_traditional_ml/06_time_series/time_series.py"
  "02_traditional_ml/07_recommenders/recommenders.py"
  "02_traditional_ml/08_self_supervised/self_supervised.py"
  "02_traditional_ml/09_reinforcement_learning/reinforcement_learning.py"
)

DEEP=(
  "03_neural_networks/01_perceptron_from_scratch/perceptron.py"
  "03_neural_networks/02_mlp_tabular/mlp_tabular.py"
  "03_neural_networks/03_cnn_images/cnn_images.py"
  "03_neural_networks/04_rnn_lstm_gru/rnn_lstm_gru.py"
  "03_neural_networks/05_transformer_attention/transformer_attention.py"
  "03_neural_networks/06_embeddings/embeddings.py"
  "03_neural_networks/07_autoencoders/autoencoders.py"
  "03_neural_networks/08_wide_and_deep/wide_and_deep.py"
  "03_neural_networks/09_two_tower_retrieval/two_tower.py"
  "04_frameworks/00_same_model_three_ways/compare.py"
  "04_frameworks/01_tensorflow/tf_version.py"
  "04_frameworks/02_pytorch/torch_version.py"
  "04_frameworks/03_jax/jax_version.py"
  "05_gcp/05_adk_agents/adk_agents.py"
  "05_gcp/06_hardware/hardware_and_scaling.py"
  "06_pipelines/pipeline.py"
  "06_pipelines/render_dag.py"
  "07_serving/01_batch_vs_online/batch_vs_online.py"
  "07_serving/02_rollout_strategies/rollout_strategies.py"
  "07_serving/03_model_optimization/model_optimization.py"
  "08_monitoring/01_skew_and_drift/skew_and_drift.py"
  "08_monitoring/02_fairness_and_slices/fairness_and_slices.py"
  "09_capstone/capstone.py"
  "10_summary/cheatsheet.py"
)

case "$MODE" in
  fast) SCRIPTS=("${SETUP[@]}" "${CORE[@]}") ;;
  full) SCRIPTS=("${SETUP[@]}" "${CORE[@]}" "${DEEP[@]}") ;;
  *) echo "usage: $0 [fast|full]" >&2; exit 2 ;;
esac

echo "python: $PY"
"$PY" -c "import sklearn" >/dev/null 2>&1 || {
  echo
  echo "WARNING: '$PY' cannot import scikit-learn, so most examples will fail."
  echo "         Install requirements.txt, or set PYTHON=/path/to/python."
  echo
}

LOGDIR="$(mktemp -d)"
RESULTS="$LOGDIR/results.tsv"
: > "$RESULTS"
FAILED=0
TOTAL_START=$SECONDS

echo "=== run-list ($MODE): ${#SCRIPTS[@]} scripts ==="
for s in "${SCRIPTS[@]}"; do
  printf '  %-62s ' "$s"
  START=$SECONDS
  log="$LOGDIR/$(echo "$s" | tr '/' '_').log"
  timeout 1800 "$PY" "$s" > "$log" 2>&1
  RC=$?
  DUR=$((SECONDS - START))
  printf 'rc=%-3s %4ss\n' "$RC" "$DUR"
  printf '%s\t%s\t%s\n' "$RC" "$DUR" "$s" >> "$RESULTS"
  if [ "$RC" -ne 0 ]; then
    FAILED=$((FAILED + 1))
    echo "----- output of failing $s -----"
    tail -25 "$log"
    echo "----- end -----"
  fi
done

TOTAL=$((SECONDS - TOTAL_START))
COUNT=$(wc -l < "$RESULTS")

echo
echo "=== $((COUNT - FAILED)) of $COUNT passed in ${TOTAL}s ==="

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### run-list (\`$MODE\`)"
    echo
    echo "**$((COUNT - FAILED)) of $COUNT passed in ${TOTAL}s**"
    echo
    echo "| | secs | script |"
    echo "|---|---|---|"
    while IFS=$'\t' read -r rc secs name; do
      if [ "$rc" -eq 0 ]; then mark="ok"; else mark="**FAIL ($rc)**"; fi
      echo "| $mark | $secs | \`$name\` |"
    done < "$RESULTS"
  } >> "$GITHUB_STEP_SUMMARY"
fi

exit $(( FAILED > 0 ? 1 : 0 ))
