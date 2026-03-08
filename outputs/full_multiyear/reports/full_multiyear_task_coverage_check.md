# Full Multi-year Task Coverage Check

Dataset used: `NYC_Collisions_Official_full.csv` (2,246,476 rows; 2012-07-01 to 2026-03-03).

## Implemented in this run

| Internal Task Ref | Status | Evidence Output |
|---|---|---|
| 10 Baseline KPI Generation | Done | `tables/yearly_kpi_multiyear.csv`, `core/yearly_collision_trend_multiyear.png` |
| 11 Vehicle vs Casualty Severity | Done | `tables/vehicle_severity_multiyear.csv`, `core/vehicle_killed_per_10k_multiyear.png` |
| 12 Lethality Matrix Construction | Done | `tables/lethality_matrix_multiyear.csv`, `heatmaps/lethality_matrix_multiyear_heatmap.png` |
| 13 Pandemic Aftermath Analysis | Done | `tables/era_kpi_multiyear.csv`, `core/unsafe_speed_share_yearly_multiyear.png` |
| 14 Pandemic Impact Report (quantitative core) | Done | `reports/deep_multiyear_analysis_report.md`, `tables/era_kpi_multiyear.csv` |
| 15 Frequency vs Severity | Done | `tables/rush_vs_nonrush_multiyear.csv`, `core/rush_nonrush_fatal_rate_multiyear.png` |
| 17 Weekend vs Weekday Factors | Done | `tables/weekend_weekday_focus_factors_multiyear.csv`, `core/factor_weekend_vs_weekday_multiyear.png` |
| 18-19 DST event-window analysis | Done | `tables/dst_event_windows_multiyear.csv`, `core/dst_window_comparison_multiyear.png` |
| 22 Spatial Hotspot Clustering | Done | `tables/top_spatial_clusters.csv`, `clusters/spatial_collision_clustered_colored.png` |
| 24 Hotspot Dynamics (partial via trend/era split) | Partial | `tables/yearly_kpi_multiyear.csv`, `tables/era_hour_collision_matrix_multiyear.csv` |

## Not implemented yet (still pending)

| Internal Task Ref | Why pending |
|---|---|
| 16 Fatality Count Modeling (Poisson/NB) | Dedicated count-model stage/script not yet added. |
| 20 Full DST DiD regression | Current output is event-window comparison, not full DiD estimator. |
| 23 Full spatio-temporal clustering | Current clustering is spatial KMeans; no temporal cluster tracking yet. |
| 28-30 Tree benchmark, calibration tuning, explainability | Only logistic baseline exists in current modeling stage. |

