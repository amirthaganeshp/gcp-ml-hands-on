-- BigQuery ML: train a model with CREATE MODEL, predict with ML.PREDICT.
--
-- Every statement here is real and runnable. Replace `your-project.your_dataset`
-- and point the FROM clauses at your own tables. Nothing in this file has been
-- executed - see the note in the folder README.
--
-- The whole appeal of BQML is that the data never moves. If your training set is
-- already a BigQuery table, the alternative is exporting it, standing up a
-- training job, and shipping predictions back. BQML skips all three.

-- ============================================================================
-- 1. LINEAR AND LOGISTIC REGRESSION
-- ============================================================================
-- The everyday cases. Note what is NOT here: no train/test split, no feature
-- scaling, no one-hot encoding. BQML does all of it automatically - it
-- standardises numerics, one-hot encodes strings, and splits the data.

CREATE OR REPLACE MODEL `your-project.your_dataset.trip_duration`
OPTIONS (
  model_type              = 'LINEAR_REG',
  input_label_cols        = ['duration_minutes'],
  -- AUTO_SPLIT holds out ~20% when the table is large, less when it is small.
  -- Be explicit if you care: a silent split is a silent assumption.
  data_split_method       = 'AUTO_SPLIT',
  l2_reg                  = 0.1,
  enable_global_explain   = TRUE          -- needed for ML.GLOBAL_EXPLAIN later
) AS
SELECT
  duration_minutes,
  distance_km,
  passenger_count,
  EXTRACT(HOUR FROM pickup_time)      AS pickup_hour,
  FORMAT_DATE('%A', DATE(pickup_time)) AS day_of_week   -- a STRING is fine
FROM `your-project.your_dataset.trips`
WHERE duration_minutes BETWEEN 1 AND 180;               -- drop the impossible


-- Binary classification. input_label_cols must be BOOL, INT64 or STRING.
CREATE OR REPLACE MODEL `your-project.your_dataset.churn`
OPTIONS (
  model_type        = 'LOGISTIC_REG',
  input_label_cols  = ['churned'],
  -- The single most useful option on an imbalanced problem, and the one
  -- people miss. Without it a 3%-positive dataset trains a model that
  -- predicts 'no' for everything and reports 97% accuracy.
  auto_class_weights = TRUE,
  data_split_method  = 'SEQ',            -- for time-ordered data...
  data_split_col     = 'signup_date'     -- ...split on TIME, never at random
) AS
SELECT churned, tenure_months, monthly_spend, support_tickets,
       plan_type, signup_date
FROM `your-project.your_dataset.customers`;


-- ============================================================================
-- 2. BOOSTED TREES - the tabular workhorse
-- ============================================================================
-- Usually the best accuracy on tabular data, for the reasons measured in
-- 02_traditional_ml/01. BQML wraps XGBoost.

CREATE OR REPLACE MODEL `your-project.your_dataset.churn_boosted`
OPTIONS (
  model_type        = 'BOOSTED_TREE_CLASSIFIER',
  input_label_cols  = ['churned'],
  auto_class_weights = TRUE,
  max_iterations    = 50,
  learn_rate        = 0.1,
  early_stop        = TRUE,
  enable_global_explain = TRUE,         -- needed for ML.GLOBAL_EXPLAIN in section 5
  -- HYPERPARAMETER TUNING, in SQL. num_trials runs that many configurations
  -- and keeps the best. This is Vertex AI Vizier under the covers.
  num_trials        = 10,
  hparam_tuning_objectives = ['roc_auc']
) AS
SELECT churned, tenure_months, monthly_spend, support_tickets, plan_type
FROM `your-project.your_dataset.customers`;


-- ============================================================================
-- 3. TIME SERIES - ARIMA_PLUS
-- ============================================================================
-- The managed answer for univariate seasonal forecasting, and the one the
-- exam reaches for. It handles holidays, multiple seasonalities, missing
-- values and anomalies without being asked.

CREATE OR REPLACE MODEL `your-project.your_dataset.demand_forecast`
OPTIONS (
  model_type            = 'ARIMA_PLUS',
  time_series_timestamp_col = 'order_date',
  time_series_data_col      = 'units_sold',
  time_series_id_col        = 'product_id',   -- fits ONE MODEL PER PRODUCT
  holiday_region            = 'US',
  auto_arima                = TRUE,
  data_frequency            = 'AUTO_FREQUENCY'
) AS
SELECT order_date, product_id, units_sold
FROM `your-project.your_dataset.orders`;

-- time_series_id_col is the feature that makes this worth knowing: one
-- statement fits thousands of independent series in parallel. Doing that by
-- hand is a distributed-systems problem; here it is one column.

SELECT * FROM ML.FORECAST(
  MODEL `your-project.your_dataset.demand_forecast`,
  STRUCT(30 AS horizon, 0.9 AS confidence_level));

-- ARIMA_PLUS_XREG is the same thing with external regressors - use it when
-- price or promotion drives the series and you know their future values.


-- ============================================================================
-- 4. CLUSTERING, ANOMALIES, AND MATRIX FACTORISATION
-- ============================================================================
CREATE OR REPLACE MODEL `your-project.your_dataset.segments`
OPTIONS (
  model_type       = 'KMEANS',
  num_clusters     = HPARAM_CANDIDATES([3, 4, 5, 6, 8]),   -- sweep it
  standardize_features = TRUE,        -- see 02_traditional_ml/03: mandatory
  num_trials       = 5,
  hparam_tuning_objectives = ['davies_bouldin_index']
) AS
SELECT recency_days, frequency, monetary_value
FROM `your-project.your_dataset.customers`;


-- Anomaly detection, the autoencoder route from 03_neural_networks/07.
CREATE OR REPLACE MODEL `your-project.your_dataset.txn_anomaly`
OPTIONS (model_type = 'AUTOENCODER', hidden_units = [32, 8, 32]) AS
SELECT amount, hour_of_day, merchant_category, days_since_last
FROM `your-project.your_dataset.transactions`
WHERE is_known_good;                   -- train on NORMAL data only

SELECT * FROM ML.DETECT_ANOMALIES(
  MODEL `your-project.your_dataset.txn_anomaly`,
  STRUCT(0.02 AS contamination),       -- the prior from 02_traditional_ml/05
  TABLE `your-project.your_dataset.transactions`);


-- Recommendations. Note this needs a reservation - see the README.
CREATE OR REPLACE MODEL `your-project.your_dataset.recs`
OPTIONS (
  model_type      = 'MATRIX_FACTORIZATION',
  user_col        = 'user_id',
  item_col        = 'item_id',
  rating_col      = 'rating',
  feedback_type   = 'EXPLICIT',        -- ratings. Use IMPLICIT for clicks.
  num_factors     = 16
) AS
SELECT user_id, item_id, rating FROM `your-project.your_dataset.ratings`;


-- ============================================================================
-- 5. EVALUATE, EXPLAIN, PREDICT
-- ============================================================================
SELECT * FROM ML.EVALUATE(MODEL `your-project.your_dataset.churn_boosted`);
-- classification -> precision, recall, accuracy, f1_score, log_loss, roc_auc
-- regression     -> MAE, MSE, MSLE, median_absolute_error, r2_score

SELECT * FROM ML.ROC_CURVE(MODEL `your-project.your_dataset.churn_boosted`);
SELECT * FROM ML.CONFUSION_MATRIX(MODEL `your-project.your_dataset.churn_boosted`);

-- Which features mattered overall (needs enable_global_explain at training).
SELECT * FROM ML.GLOBAL_EXPLAIN(MODEL `your-project.your_dataset.churn_boosted`);

-- Why THIS row got THIS prediction - per-row Shapley attributions.
SELECT * FROM ML.EXPLAIN_PREDICT(
  MODEL `your-project.your_dataset.churn_boosted`,
  TABLE `your-project.your_dataset.new_customers`,
  STRUCT(3 AS top_k_features));

-- Prediction is just a query, which is the whole point: the result is a
-- table you can join, filter and write back without leaving BigQuery.
SELECT customer_id,
       predicted_churned,
       predicted_churned_probs[OFFSET(0)].prob AS churn_probability
FROM ML.PREDICT(MODEL `your-project.your_dataset.churn_boosted`,
                TABLE `your-project.your_dataset.new_customers`)
WHERE predicted_churned_probs[OFFSET(0)].prob > 0.7
ORDER BY churn_probability DESC;


-- ============================================================================
-- 6. THE PART THAT SURPRISES PEOPLE: BQML AS A FRONT END
-- ============================================================================
-- BQML is not limited to models it trains. It can call a Vertex AI endpoint,
-- or a foundation model, from SQL.

CREATE OR REPLACE MODEL `your-project.your_dataset.imported_model`
OPTIONS (model_type = 'TENSORFLOW',
         model_path = 'gs://your-bucket/saved_model/*');
-- A SavedModel from 04_frameworks/01_tensorflow, now queryable in SQL.

-- And a remote model backed by Vertex AI - this is how generative AI gets
-- into a SQL workflow.
--
-- MODEL VERSIONS MOVE - check before you run this. As of October 2026:
--   - Gemini 2.0 is gone and Gemini 2.5 retires on Vertex AI on 2026-10-16.
--   - Gemini 3.x models are served ONLY from the `global` endpoint. Named
--     bare, the model resolves against the connection's region and returns
--     404 "not found or your project does not have access" - which reads
--     like an IAM problem and is not one. Hence the full global-endpoint URL
--     below. Confirm the current model list and BigQuery's endpoint syntax
--     in the docs; both change faster than this file.
CREATE OR REPLACE MODEL `your-project.your_dataset.gemini`
REMOTE WITH CONNECTION `your-project.us.my-connection`
OPTIONS (endpoint = 'https://aiplatform.googleapis.com/v1/projects/your-project/locations/global/publishers/google/models/gemini-3.5-flash');

-- flatten_json_output = TRUE is what produces the ml_generate_text_llm_result
-- column; without it the result is a raw JSON column and this SELECT fails.
SELECT ml_generate_text_llm_result AS summary
FROM ML.GENERATE_TEXT(
  MODEL `your-project.your_dataset.gemini`,
  (SELECT CONCAT('Summarise in one sentence: ', review_text) AS prompt
   FROM `your-project.your_dataset.reviews` LIMIT 10),
  STRUCT(0.2 AS temperature, 200 AS max_output_tokens,
         TRUE AS flatten_json_output));

-- That last statement is the reason BQML keeps appearing in exam questions
-- about generative AI on structured data: the data never leaves BigQuery and
-- no serving infrastructure is involved.
