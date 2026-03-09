# Fatality Count Modeling Report

Dataset: full_multiyear
Modeling target: grouped fatality_count by year_month x hour x is_weekend.

Data construction:
- Feature rows scanned: 2246476
- Grouped rows generated: 7920
- Rows dropped for invalid grouping keys (MONTH/HOUR/IS_WEEKEND): 0

Descriptive summary:
- Mean fatality_count: 0.451263
- Variance fatality_count: 0.549209
- Overdispersion ratio (var/mean): 1.217050
- Zero-fatality group share: 0.662374
- Mean collision_count per group: 283.645960

Required-field missing/invalid share:
- MONTH: 0 (0.000000)
- HOUR: 0 (0.000000)
- IS_WEEKEND: 0 (0.000000)
- NUMBER OF PERSONS KILLED: 31 (0.000014)
- NUMBER OF PERSONS INJURED: 18 (0.000008)

Poisson vs Negative Binomial comparison:
- Model status: ok
- Model fit complete (train=6356, test=1564).
- Estimated NB overdispersion alpha: 0.143067
- Poisson: llf=-5356.176766, aic=10836.353531, test_mae=0.533861, test_rmse=0.720928
- NegativeBinomial: llf=-5348.371255, aic=10822.742511, test_mae=0.533969, test_rmse=0.720922
- Best model by AIC: NegativeBinomial

Plot status:
- Created hourly collision count comparison plot.
- Created hourly fatality count comparison plot.
- Created hourly fatality rate comparison plot.
- Created grouped fatality count histogram.
- Created Poisson vs NB metric comparison plot.
- Created observed vs predicted test-group plot.