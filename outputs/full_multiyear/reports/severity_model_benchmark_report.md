# Severity Model Benchmark Report

Dataset: full_multiyear
Benchmark setup:
- Baseline model: sklearn_logistic
- Tree benchmark model: tree_random_forest
- Shared target: ANY_INJURY
- Shared train/test split source: sample_train_test_rows in train_baseline_severity_model.py
- Train rows used: 250000
- Test rows used: 62500

Logistic baseline metrics:
- accuracy: 0.704544
- precision: 0.396137
- recall: 0.413575
- f1: 0.404668
- tp: 6276.000000
- tn: 37758.000000
- fp: 9567.000000
- fn: 8899.000000
- roc_auc: 0.618781
- pr_auc: 0.349873

RandomForest benchmark metrics:
- accuracy: 0.679984
- precision: 0.371238
- recall: 0.458451
- f1: 0.410261
- tp: 6957.000000
- tn: 35542.000000
- fp: 11783.000000
- fn: 8218.000000
- roc_auc: 0.632519
- pr_auc: 0.354939

Delta (tree - logistic):
- accuracy: -0.024560
- precision: -0.024899
- recall: 0.044876
- f1: 0.005593
- tp: 681.000000
- tn: -2216.000000
- fp: 2216.000000
- fn: -681.000000
- roc_auc: 0.013739
- pr_auc: 0.005066

Tree plot status:
- Created ROC curve plot.
- Created precision-recall curve plot.
- Created calibration curve plot.

Tree output artifacts:
- severity_model_predictions_sample_tree_random_forest.csv
- severity_model_metrics_tree_random_forest.csv
- severity_model_benchmark_comparison.csv
- tree_random_forest_roc_curve.png
- tree_random_forest_precision_recall_curve.png
- tree_random_forest_calibration_curve.png